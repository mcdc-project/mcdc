from __future__ import annotations

####

import numpy as np

from pathlib import Path
from mpi4py import MPI

####

import mcdc
import mcdc.code_factory.adapt as adapt
import mcdc.config as config
import mcdc.object_ as object_module
import mcdc.object_.base as base

from mcdc.object_.base import (
    ObjectBase,
    ObjectNonSingleton,
    ObjectPolymorphic,
    ObjectSingleton,
)
from mcdc.object_.particle import Particle, ParticleBank, ParticleData
from mcdc.object_.tally import TallyBase
from mcdc.print_ import print_error
from mcdc.util import flatten

type_map = {
    bool: "?",
    float: "f8",
    int: "i8",
    str: "U32",
    np.bool_: "?",
    np.float64: "f8",
    np.int64: "i8",
    np.uint64: "u8",
    np.str_: "U32",
}

bank_names = ["bank_active", "bank_census", "bank_source", "bank_future"]

# ======================================================================================
# Gather and group the classes
# ======================================================================================

base_classes = [
    getattr(base, x)
    for x in dir(base)
    if isinstance(getattr(base, x), type) and issubclass(getattr(base, x), ObjectBase)
]

all_classes = [ParticleData, Particle]
mcdc_classes = [ParticleData, Particle]
polymorphic_bases = []

file_names = [x for x in dir(object_module) if x[:2] != "__" and x != "base"]
for file_name in file_names:
    file = getattr(object_module, file_name)
    item_names = dir(file)
    for item_name in item_names:
        item = getattr(file, item_name)
        if (
            isinstance(item, type)
            and issubclass(item, ObjectBase)
            and item not in all_classes
        ):
            all_classes.append(item)

            if (
                item not in base_classes
                and "label" in dir(item)
                and item not in mcdc_classes
            ):
                mcdc_classes.append(item)

polymorphic_bases = [
    x for x in all_classes if x.__name__[-4:] == "Base" and "label" in dir(x)
]

# ======================================================================================
# Numba object creation
# ======================================================================================


def generate_numba_objects(simulation):
    # ==================================================================================
    # Allocate key items for the Numba object:
    #   - Python annotations
    #   - Numba structures
    #   - Records
    #   - Data: flattened vector to store arbitrary-size arrays
    #   - Accessor targets: to generate getter/setter helpers to easily access data
    # ==================================================================================

    annotations = {}
    structures = {}
    records = {}
    data = {"size": 0}
    accessor_targets = {}

    for mcdc_class in mcdc_classes:
        annotations[mcdc_class.label] = {}
        structures[mcdc_class.label] = []
        accessor_targets[mcdc_class.label] = []
        if issubclass(mcdc_class, ObjectNonSingleton):
            records[mcdc_class.label] = []
        else:
            records[mcdc_class.label] = {}

    # Particle banks
    for name in bank_names:
        annotations[name] = {}
        structures[name] = []
        accessor_targets[name] = []

    # Move simulation to last
    annotations["simulation"] = annotations.pop("simulation")
    structures["simulation"] = structures.pop("simulation")
    records["simulation"] = records.pop("simulation")
    accessor_targets["simulation"] = accessor_targets.pop("simulation")

    # ==================================================================================
    # Gather the annotations from the classes
    # ==================================================================================

    for mcdc_class in mcdc_classes:
        # Include all ancestors, but stop at the MC/DC base classes
        classes = []
        for item in mcdc_class.__mro__:
            if item in base_classes:
                break
            classes.append(item)

        # If polymorphic, don't include the polymorphic base
        if issubclass(mcdc_class, ObjectPolymorphic):
            classes = [mcdc_class]

        # Get the annotations
        for class_ in classes:
            new_annotations = {
                k: v
                for k, v in class_.__annotations__.items()
                if k not in ["label", "non_numba"]
                and (
                    "non_numba" not in dir(class_)
                    or ("non_numba" in dir(class_) and k not in class_.non_numba)
                )
            }
            # Evaluate stringified annotation
            if (
                len(new_annotations) > 0
                and type(next(iter(new_annotations.values()))) == str
            ):
                new_annotations = parse_annotations_dict(new_annotations)

            annotations[mcdc_class.label].update(new_annotations)

    # Particle banks
    for name in bank_names:
        annotations[name] = {
            k: v
            for k, v in ParticleBank.__annotations__.items()
            if k not in ["label", "non_numba"]
            and (
                "non_numba" not in dir(ParticleBank)
                or (
                    "non_numba" in dir(ParticleBank) and k not in ParticleBank.non_numba
                )
            )
        }

    # ==================================================================================
    # Set the structures and accessor targets based on the annotations
    # ==================================================================================

    # Temporary simulation object structure
    simulation_object_structure = []
    for field in annotations["simulation"]:
        hint = annotations["simulation"][field]
        hint_origin = get_origin(hint)
        hint_args = get_args(hint)

        if hint in all_classes:
            simulation_object_structure.append((field, hint))
            continue
        if hint_origin == list and hint_args[0] in all_classes:
            simulation_object_structure.append((field, list, hint_args[0]))
            continue

    # Set the structures and accessor targets
    for label in annotations.keys():
        set_structure(label, structures, accessor_targets, annotations)

    # Generate the accessor helper
    if MPI.COMM_WORLD.Get_rank() == 0:
        generate_mcdc_access(accessor_targets)

    # Add ID for non-singleton
    for class_ in mcdc_classes:
        if issubclass(class_, ObjectNonSingleton):
            structures[class_.label].append(("ID", "i8"))
        # Set parent and child ID and type if polymorphic
        if issubclass(class_, ObjectPolymorphic):
            if class_.__name__[-4:] == "Base":
                structures[class_.label].append(("child_type", "i8"))
                structures[class_.label].append(("child_ID", "i8"))
            else:
                structures[class_.label].append(("parent_ID", "i8"))

    # Add particles to particle banks and add particle banks to the simulation
    for name in bank_names:
        bank = getattr(simulation, name)
        size = int(bank.size[0])
        structures[name] += [
            ("particles", into_dtype(structures["particle_data"]), (size,))
        ]
        #
        structures["simulation"] = [(name, into_dtype(structures[name]))] + structures[
            "simulation"
        ]

    # ==================================================================================
    # Set records and data based on the simulation structures and objects
    # ==================================================================================

    # Allocate object containers
    objects = []

    # Gather the objects from the simulation
    attribute_names = [
        x
        for x in dir(simulation)
        if (
            not x.startswith("__")
            and (
                isinstance(getattr(simulation, x), ObjectBase)
                or not callable(getattr(simulation, x))
            )
            and x not in simulation.non_numba
        )
    ]
    for attribute_name in attribute_names:
        attribute = getattr(simulation, attribute_name)
        if type(attribute) in mcdc_classes:
            objects.append(attribute)
        if type(attribute) == list:
            for item in attribute:
                if type(item) in mcdc_classes:
                    objects.append(item)

    # Set the objects
    for object_ in objects:
        set_object(object_, annotations, structures, records, data)
    set_object(simulation, annotations, structures, records, data)

    # Allocate the flattened data and re-set the objects
    data["array"], data["pointer"] = create_data_array(data["size"], type_map[float])

    data["size"] = 0
    records = {}
    for mcdc_class in mcdc_classes:
        if issubclass(mcdc_class, ObjectNonSingleton):
            records[mcdc_class.label] = []
        else:
            records[mcdc_class.label] = {}
    records["simulation"] = records.pop("simulation")

    for object_ in objects:
        set_object(object_, annotations, structures, records, data, set_data=True)
    set_object(simulation, annotations, structures, records, data, set_data=True)

    # ==================================================================================
    # Finalize the simulation object structure and set record
    # ==================================================================================

    new_structure = []
    record = records["simulation"]
    for item in simulation_object_structure:
        field = item[0]
        type_1 = item[1]

        # List of objects
        if type_1 == list:
            type_2 = item[2]

            # List of non-polymorphics
            if item[2] not in polymorphic_bases:
                N = len(records[item[2].label])
                new_structure.append(
                    (field, into_dtype(structures[item[2].label]), (N,))
                )
                new_structure.append((f"N_{plural_to_singular(field)}", "i8"))
                record[f"N_{plural_to_singular(field)}"] = N

            # List of polymorphics
            else:
                for class_ in mcdc_classes:
                    if issubclass(class_, type_2):
                        N = len(records[class_.label])
                        new_structure.append(
                            (
                                singular_to_plural(class_.label),
                                into_dtype(structures[class_.label]),
                                (N,),
                            )
                        )
                        new_structure.append((f"N_{class_.label}", "i8"))
                        record[f"N_{class_.label}"] = N

        # Singleton
        elif item[1] in mcdc_classes and issubclass(item[1], ObjectSingleton):
            new_structure.append((field, into_dtype(structures[item[1].label])))

        else:
            print_error(f"Unknown type: {item}")

    structures["simulation"] = new_structure + structures["simulation"]

    # Print the fields
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(f"{Path(mcdc.__file__).parent}/numba_types.py", "w") as f:
            text = "# The following is automatically generated by code_factory.py\n\n"
            text += (
                "from mcdc.code_factory.numba_objects_generator import into_dtype\n\n"
            )

            for label in structures.keys():
                text += f"{label} = into_dtype([\n"
                structure = structures[label]
                for item in structure:
                    if type(item[1]) != np.dtypes.VoidDType:
                        text += f"    {item},\n"
                    else:
                        if len(item) == 3:
                            text += f"    ('{item[0]}', {plural_to_singular(item[0])}, {item[2]}),\n"
                        else:
                            text += f"    ('{item[0]}', {item[0]}),\n"
                text += "])\n\n"

            f.write(text)

    # ==================================================================================
    # Set with records
    # ==================================================================================

    # The global structure/variable container
    mcdc_simulation_arr, mcdc_simulation_pointer = create_mcdc_array(
        into_dtype(structures["simulation"])
    )
    mcdc_simulation = mcdc_simulation_arr[0]

    record = records["simulation"]
    structure = structures["simulation"]
    for item in structure:
        field = item[0]
        field_type = item[1]
        size = -1
        if len(item) == 3:
            size = item[2][0]

        # Skip particular attributes
        if field in bank_names:
            continue

        # Simple attribute
        if type(field_type) != np.dtypes.VoidDType:
            mcdc_simulation[field] = record[field]

        # MC/DC objects
        else:
            # Singleton
            if size == -1:
                for sub_item in structures[field]:
                    mcdc_simulation[field][sub_item[0]] = records[field][sub_item[0]]
            # Non-singleton
            else:
                singular_field = plural_to_singular(field)
                for i in range(size):
                    for sub_item in structures[singular_field]:
                        mcdc_simulation[field][i][sub_item[0]] = records[
                            singular_field
                        ][i][sub_item[0]]

    # Manually set particle bank attributes
    for name in bank_names:
        mcdc_simulation[name]["tag"] = getattr(simulation, name).tag

    return mcdc_simulation_arr, data["array"]


def set_structure(label, structures, accessor_targets, annotations):
    structure = structures[label]
    annotation = annotations[label]
    accessor_target = accessor_targets[label]

    for field in annotation:
        hint = annotation[field]
        hint_origin = get_origin(hint)
        hint_args = get_args(hint)
        hint_origin_shape = None
        hint_inner_dtype = None
        fixed_size_array = False

        # Process annotation
        if hint_origin is Annotated:
            hint_decoded = decode_annotated_ndarray(hint)
            hint_origin = hint_decoded["origin"]
            hint_origin_shape = hint_decoded["shape"]
            hint_inner_dtype = get_args(hint_decoded["dtype"])[0]
            fixed_size_array = True

            # Mark as arbitrary size if string is used in shape
            for dim_size in hint_origin_shape:
                if type(dim_size) == str:
                    fixed_size_array = False
                    break

        # Skip simulation object structure
        if label == "simulation":
            if hint in all_classes:
                continue
            if hint_origin == list and hint_args[0] in all_classes:
                hint_origin = np.ndarray
                continue

        # ==========================================================================
        # Get the type
        # ==========================================================================

        # Basics
        simple_scalar = hint in type_map.keys()
        simple_list = hint_origin == list and hint_args[0] in type_map.keys()
        numpy_array = hint_origin == np.ndarray

        # MC/DC class
        def non_polymorphic(x):
            # Only treat real classes that inherit from ObjectNonSingleton
            return (
                isinstance(x, type)
                and issubclass(x, ObjectNonSingleton)
                and x not in polymorphic_bases
            )

        def polymorphic_base(x):
            # Only treat real classes that are registered polymorphic bases
            return isinstance(x, type) and x in polymorphic_bases

        # List of MC/DC classes
        list_of_non_polymorphics = hint_origin == list and non_polymorphic(hint_args[0])
        list_of_polymorphic_bases = hint_origin == list and polymorphic_base(
            hint_args[0]
        )

        # ==========================================================================
        # Set the structure
        # ==========================================================================

        # Basics
        if fixed_size_array:
            structure.append((field, type_map[hint_inner_dtype], hint_origin_shape))
        elif simple_scalar:
            structure.append((field, type_map[hint]))
        elif simple_list or numpy_array:
            structure.append((f"{field}_offset", "i8"))
            structure.append((f"{field}_length", "i8"))
            if hint_origin_shape is not None:
                accessor_target.append((f"{field}", hint_origin_shape))
            else:
                accessor_target.append((f"{field}", (f"{field}_length",)))

        # MC/DC classes
        elif non_polymorphic(hint) or polymorphic_base(hint):
            structure.append((f"{field}_ID", "i8"))

        # List of MC/DC classes
        elif list_of_non_polymorphics or list_of_polymorphic_bases:
            singular = plural_to_singular(field)
            structure.append((f"N_{singular}", "i8"))
            structure.append((f"{singular}_IDs_offset", "i8"))
            if hint_origin_shape is not None:
                accessor_target.append((f"{singular}_IDs", hint_origin_shape))
            else:
                accessor_target.append((f"{singular}_IDs", (f"N_{singular}",)))

        # Unknown type
        else:
            print_error(f"Unknown type hint for {label}/{field}: {hint}")


def set_object(
    object_, annotations, structures, records, data, class_=None, set_data=False
):
    if class_ == None:
        class_ = object_.__class__

    # Set the parent first if polymorphics
    if isinstance(object_, ObjectPolymorphic) and class_ not in polymorphic_bases:
        for parent_class in polymorphic_bases:
            if issubclass(class_, parent_class):
                set_object(
                    object_,
                    annotations,
                    structures,
                    records,
                    data,
                    parent_class,
                    set_data,
                )

    annotation = annotations[class_.label]
    structure = structures[class_.label]
    record = {}

    if class_.label == "simulation":
        record = records["simulation"]

    # Straightforwardly set up attributes
    for key in [x[0] for x in structure]:
        if key in dir(object_):
            # Skip if set already
            if key in record.keys():
                continue
            record[key] = getattr(object_, key)

    # Loop over the supported attributes
    attribute_names = [
        x for x in dir(object_) if (x[:2] != "__" and not callable(getattr(object_, x)))
    ]
    if "non_numba" in dir(object_):
        attribute_names = list(set(attribute_names) - set(object_.non_numba))
    for attribute_name in attribute_names:
        # Skip if set already
        if attribute_name in record.keys():
            continue

        # Skip if not in annotation
        if attribute_name not in annotation.keys():
            continue
        attribute = getattr(object_, attribute_name)

        # Convert list of supported types into Numpy array
        if type(attribute) == list:
            if get_args(annotation[attribute_name])[0] in type_map.keys():
                attribute = np.array(attribute)

        # Numpy array
        if type(attribute) == np.ndarray:
            attribute_flatten = attribute.flatten()
            record[f"{attribute_name}_offset"] = data["size"]
            record[f"{attribute_name}_length"] = len(attribute_flatten)
            if set_data:
                data["array"][data["size"] : data["size"] + len(attribute_flatten)] = (
                    attribute_flatten[:]
                )
            data["size"] += len(attribute_flatten)

        # Non-singleton object
        elif isinstance(attribute, ObjectNonSingleton):
            if (
                not isinstance(attribute, ObjectPolymorphic)
                or annotation[attribute_name] in polymorphic_bases
            ):
                record[f"{attribute_name}_ID"] = attribute.ID
            else:
                record[f"{attribute_name}_ID"] = attribute.child_ID

        # List of Non-singleton objects
        elif type(attribute) == list:
            inner_type = get_args(annotation[attribute_name])[0]

            # Flatten the list
            attribute_flatten = list(flatten(attribute))
            singular_name = plural_to_singular(attribute_name)

            if not issubclass(inner_type, ObjectNonSingleton):
                print_error(
                    f"[ERROR] Get a list of non-object for {attribute_name}: {attribute}"
                )

            record[f"N_{singular_name}"] = len(attribute_flatten)
            record[f"{singular_name}_IDs_offset"] = data["size"]
            if set_data:
                if (
                    not issubclass(inner_type, ObjectPolymorphic)
                    or inner_type in polymorphic_bases
                ):
                    data["array"][
                        data["size"] : data["size"] + len(attribute_flatten)
                    ] = [x.ID for x in attribute_flatten]
                else:
                    data["array"][
                        data["size"] : data["size"] + len(attribute_flatten)
                    ] = [x.child_ID for x in attribute_flatten]
            data["size"] += len(attribute_flatten)

    # Complete for simulation object
    if class_.label == "simulation":
        return

    # Set ID of non-singleton
    if isinstance(object_, ObjectNonSingleton):
        if not isinstance(object_, ObjectPolymorphic):
            record["ID"] = object_.ID

        # Set parent and child ID and type if polymorphic
        else:
            # Parent
            if class_ in polymorphic_bases:
                record["ID"] = object_.ID
                record["child_ID"] = object_.child_ID
                record["child_type"] = object_.type
            # Child
            else:
                record["ID"] = object_.child_ID
                record["parent_ID"] = object_.ID

    # Set tally bins
    if class_ == TallyBase:
        tally_size = np.prod(object_.bin_shape)
        record[f"bin_offset"] = data["size"]
        record[f"bin_sum_offset"] = data["size"] + tally_size
        record[f"bin_sum_square_offset"] = data["size"] + tally_size * 2
        record[f"bin_length"] = tally_size
        record[f"bin_sum_length"] = tally_size
        record[f"bin_sum_square_length"] = tally_size
        data["size"] += 3 * tally_size

    # Check structure-record compatibility
    missing = set([x[0] for x in structure]) - set(record.keys())
    if len(missing) > 0:
        print_error(f"Missing structure keys in record for {class_.label}: {missing}")

    # Register the record
    if isinstance(object_, ObjectSingleton):
        records[class_.label] = record
    elif isinstance(object_, ObjectNonSingleton):
        records[class_.label].append(record)


# =============================================================================
# Global GPU/CPU Array Variable Constructors
# =============================================================================


def create_data_array(size, dtype):
    if config.target == "gpu":
        import harmonize, numba

        if config.gpu_state_storage == "managed":
            data_tally_ptr = harmonize.alloc_managed_bytes(size)
        else:
            data_tally_ptr = harmonize.alloc_device_bytes(size)
        data_tally_uint = adapt.voidptr_to_uintp(data_tally_ptr)
        data_tally = numba.carray(data_tally_ptr, (size,), dtype)
        return data_tally, data_tally_uint
    else:
        data_tally = np.zeros(size, dtype=dtype)
        return data_tally, 0


def create_mcdc_array(dtype):
    if config.target == "gpu":
        import harmonize, numba

        if config.gpu_state_storage == "managed":
            mcdc_ptr = harmonize.alloc_managed_bytes(dtype.itemsize)
        else:
            mcdc_ptr = harmonize.alloc_device_bytes(dtype.itemsize)
        mcdc_uint = adapt.voidptr_to_uintp(mcdc_ptr)
        mcdc_array = numba.carray(mcdc_ptr, (1,), dtype)
        return mcdc_array, mcdc_uint
    else:
        mcdc_array = np.zeros((1,), dtype=dtype)
        return mcdc_array, 0


# ======================================================================================
# Alignment Logic
# ======================================================================================
# While CPU execution can robustly handle all sorts of Numba types, GPU
# execution requires structs to follow some of the basic properties expected of
# C-style structs with standard layout:
#
#      - Every primitive field is aligned by its size, and padding is inserted
#        between fields to ensure alignment in arrays and nested data structures
#
#      - Every field has a unique address
#
# If these rules are violated, memory accesses made in GPUs may encounter
# problems. For example, in cases where an access is not at an address aligned
# by their size, a segfault or similar fault will occur, or information will be
# lost. These issues were fixed by providing a function, align, which ensures the
# field lists fed to np.dtype fulfill these requirements.
#
# The align function does the following:
#
#      - Tracks the cumulative offset of fields as they appear in the input list.
#
#      - Inserts additional padding fields to ensure that primitive fields are
#        aligned by their size
#
#      - Re-sizes arrays to have at least one element in their array (this ensure
#        they have a non-zero size, and hence cannot overlap base addresses with
#        other fields.
#


def fixup_dims(dim_tuple):
    return tuple([max(d, 1) for d in dim_tuple])


def align(field_list):
    result = []
    offset = 0
    pad_id = 0
    for field in field_list:
        if len(field) > 3:
            print_error(
                "Unexpected struct field specification. Specifications \
                        usually only consist of 3 or fewer members"
            )
        multiplier = 1
        if len(field) == 3:
            field = (field[0], field[1], fixup_dims(field[2]))
            for d in field[2]:
                multiplier *= d
        kind = np.dtype(field[1])
        size = kind.itemsize

        if kind.isbuiltin == 0:
            alignment = 8
        elif kind.isbuiltin == 1:
            alignment = size
        else:
            print_error("Unexpected field item type")

        size *= multiplier

        if offset % alignment != 0:
            pad_size = alignment - (offset % alignment)
            result.append((f"padding_{pad_id}", np.uint8, (pad_size,)))
            pad_id += 1
            offset += pad_size

        result.append(field)
        offset += size

    if offset % 8 != 0:
        pad_size = 8 - (offset % 8)
        result.append((f"padding_{pad_id}", np.uint8, (pad_size,)))
        pad_id += 1

    return result


def into_dtype(field_list):
    result = np.dtype(align(field_list), align=True)
    return result


# ======================================================================================
# Type parser
# ======================================================================================

from typing import Annotated, Any, ForwardRef, Optional, Union, get_args, get_origin
import numpy as np
from numpy.typing import NDArray


# --- Safe locals for eval + ForwardRef fallback ---
class _FwdRefDict(dict):
    """If a symbol isn't in the whitelist, treat it as a ForwardRef('Symbol')."""

    def __missing__(self, key):
        for class_ in all_classes:
            if key == class_.__name__:
                return class_
        return ForwardRef(key)


_SAFE_GLOBALS = {}  # no builtins
_SAFE_LOCALS = _FwdRefDict(
    {
        # builtins
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "bytes": bytes,
        "object": object,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        # typing
        "Any": Any,
        "Annotated": Annotated,
        "Union": Union,
        "Optional": Optional,
        # numpy typing
        "NDArray": NDArray,
        # numpy dtypes (extend if you need more)
        "float64": np.float64,
        "float32": np.float32,
        "int64": np.int64,
        "int32": np.int32,
    }
)


def parse_type_hint_str(s: str):
    """
    Parse a stringified type hint into a runtime type/typing object.
    Unknown identifiers become ForwardRef('Name') so we don't import/resolve.
    """
    s = s.strip()
    # Special-case empty or 'None' if you ever pass those
    if s in {"None", "NoneType"}:
        return type(None)
    return eval(s, _SAFE_GLOBALS, _SAFE_LOCALS)


def parse_annotations_dict(ann: dict[str, str]) -> dict[str, object]:
    return {k: parse_type_hint_str(v) for k, v in ann.items()}


def decode_annotated_ndarray(hint):
    inner, metadata = get_args(hint)
    inner_origin = get_origin(inner)
    inner_args = get_args(inner)
    shape_type, dtype_type = inner_args
    return {
        "origin": inner_origin,
        "shape": metadata,
        "shape_type": shape_type,
        "dtype": dtype_type,
    }


# ======================================================================================
# Helpers for mcdc_get generators
# ======================================================================================


def generate_mcdc_access(targets):
    for object_name in targets.keys():
        path = f"{Path(mcdc.__file__).parent}"
        file_getter = open(f"{path}/mcdc_get/{object_name}.py", "w")
        file_setter = open(f"{path}/mcdc_set/{object_name}.py", "w")

        text_getter = (
            "# The following is automatically generated by code_factory.py\n\n"
        )
        text_setter = (
            "# The following is automatically generated by code_factory.py\n\n"
        )

        text_getter += "from numba import njit\n\n\n"
        text_setter += "from numba import njit\n\n\n"

        for attribute in targets[object_name]:
            attribute_name = attribute[0]
            shape = attribute[1]

            if len(shape) == 1:
                text_getter += _accessor_1d_element(object_name, attribute_name)
                text_getter += _accessor_1d_all(object_name, attribute_name, shape[0])
                text_getter += _accessor_1d_last(object_name, attribute_name, shape[0])

                text_setter += _accessor_1d_element(object_name, attribute_name, True)
                text_setter += _accessor_1d_all(
                    object_name, attribute_name, shape[0], True
                )
                text_setter += _accessor_1d_last(
                    object_name, attribute_name, shape[0], True
                )

            elif len(shape) == 2:
                text_getter += _accessor_2d_vector(
                    object_name, attribute_name, shape[1]
                )
                text_getter += _accessor_2d_element(
                    object_name, attribute_name, shape[1]
                )

                text_setter += _accessor_2d_vector(
                    object_name, attribute_name, shape[1], True
                )
                text_setter += _accessor_2d_element(
                    object_name, attribute_name, shape[1], True
                )

            elif len(shape) == 3:
                text_getter += _accessor_3d_element(
                    object_name, attribute_name, shape[1], shape[2]
                )

                text_setter += _accessor_3d_element(
                    object_name, attribute_name, shape[1], shape[2], True
                )

            text_getter += _accessor_chunk(object_name, attribute_name)
            text_setter += _accessor_chunk(object_name, attribute_name, True)

        file_getter.write(text_getter[:-2])
        file_setter.write(text_setter[:-2])

        file_getter.close()
        file_setter.close()

    for key in ["get", "set"]:
        with open(f"{Path(mcdc.__file__).parent}/mcdc_{key}/__init__.py", "w") as f:
            text = "# The following is automatically generated by code_factory.py\n\n"
            for i, object_name in enumerate(targets.keys()):
                text += f"import mcdc.mcdc_{key}.{object_name} as {object_name}\n"
                if i < len(targets.keys()) - 1:
                    text += "\n"
            f.write(text)


def _accessor_1d_element(object_name, attribute_name, setter=False):
    text = f"@njit\n"
    if setter:
        text += f"def {attribute_name}(index, {object_name}, data, value):\n"
    else:
        text += f"def {attribute_name}(index, {object_name}, data):\n"
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    if setter:
        text += f"    data[offset + index] = value\n\n\n"
    else:
        text += f"    return data[offset + index]\n\n\n"
    return text


def _accessor_1d_all(object_name, attribute_name, size, setter=False):
    text = f"@njit\n"
    if setter:
        text += f"def {attribute_name}_all({object_name}, data, value):\n"
    else:
        text += f"def {attribute_name}_all({object_name}, data):\n"
    text += f'    start = {object_name}["{attribute_name}_offset"]\n'
    if type(size) == str:
        text += f'    size = {object_name}["{size}"]\n'
    else:
        text += f"    size = {size}\n"
    text += f"    end = start + size\n"
    if setter:
        text += f"    data[start:end] = value\n\n\n"
    else:
        text += f"    return data[start:end]\n\n\n"
    return text


def _accessor_1d_last(object_name, attribute_name, size, setter=False):
    text = f"@njit\n"
    if setter:
        text += f"def {attribute_name}_last({object_name}, data, value):\n"
    else:
        text += f"def {attribute_name}_last({object_name}, data):\n"
    text += f'    start = {object_name}["{attribute_name}_offset"]\n'
    if type(size) == str:
        text += f'    size = {object_name}["{size}"]\n'
    else:
        text += f"    size = {size}\n"
    text += f"    end = start + size\n"
    if setter:
        text += f"    data[end - 1] = value\n\n\n"
    else:
        text += f"    return data[end - 1]\n\n\n"
    return text


def _accessor_chunk(object_name, attribute_name, setter=False):
    text = f"@njit\n"
    if setter:
        text += (
            f"def {attribute_name}_chunk(start, length, {object_name}, data, value):\n"
        )
    else:
        text += f"def {attribute_name}_chunk(start, length, {object_name}, data):\n"
    text += f'    start += {object_name}["{attribute_name}_offset"]\n'
    text += f"    end = start + length\n"
    if setter:
        text += f"    data[start:end] = value\n\n\n"
    else:
        text += f"    return data[start:end]\n\n\n"
    return text


def _accessor_2d_element(object_name, attribute_name, stride, setter=False):
    text = f"@njit\n"
    if setter:
        text += f"def {attribute_name}(index_1, index_2, {object_name}, data, value):\n"
    else:
        text += f"def {attribute_name}(index_1, index_2, {object_name}, data):\n"
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    if isinstance(stride, str):
        text += f'    stride = {object_name}["{stride}"]\n'
    else:
        text += f"    stride = {stride}\n"
    if setter:
        text += f"    data[offset + index_1 * stride + index_2] = value\n\n\n"
    else:
        text += f"    return data[offset + index_1 * stride + index_2]\n\n\n"
    return text


def _accessor_2d_vector(object_name, attribute_name, stride, setter=False):
    text = f"@njit\n"
    if setter:
        text += f"def {attribute_name}_vector(index_1, {object_name}, data, value):\n"
    else:
        text += f"def {attribute_name}_vector(index_1, {object_name}, data):\n"
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    if isinstance(stride, str):
        text += f'    stride = {object_name}["{stride}"]\n'
    else:
        text += f"    stride = {stride}\n"
    text += f"    start = offset + index_1 * stride\n"
    text += f"    end = start + stride\n"
    if setter:
        text += f"    data[start:end] - value\n\n\n"
    else:
        text += f"    return data[start:end]\n\n\n"
    return text


def _accessor_3d_element(object_name, attribute_name, stride_2, stride_3, setter=False):
    text = f"@njit\n"
    if setter:
        text += f"def {attribute_name}(index_1, index_2, index_3, {object_name}, data, value):\n"
    else:
        text += (
            f"def {attribute_name}(index_1, index_2, index_3, {object_name}, data):\n"
        )
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    text += f'    stride_2 = {object_name}["{stride_2}"]\n'
    text += f'    stride_3 = {object_name}["{stride_3}"]\n'
    if setter:
        text += f"    data[offset + index_1 * stride_2 * stride_3 + index_2 * stride_3 + index_3] = value\n\n\n"
    else:
        text += f"    return data[offset + index_1 * stride_2 * stride_3 + index_2 * stride_3 + index_3]\n\n\n"
    return text


# ======================================================================================
# Misc.
# ======================================================================================


def plural_to_singular(word: str) -> str:
    """
    Convert a plural English noun (possibly underscore-separated) to singular.
    Applies only to the last word and handles common irregulars.
    """
    irregulars = {
        "universes": "universe",
        "children": "child",
        "men": "man",
        "women": "woman",
        "people": "person",
        "mice": "mouse",
        "geese": "goose",
        "teeth": "tooth",
        "feet": "foot",
        "indices": "index",
        "matrices": "matrix",
        "criteria": "criterion",
        "data": "data",  # invariant
        "spectra": "spectrum",
    }

    parts = word.lower().split("_")
    w = parts[-1]

    if w in irregulars:
        parts[-1] = irregulars[w]
    elif w.endswith("ies") and len(w) > 3:
        parts[-1] = w[:-3] + "y"
    elif w.endswith("ves") and len(w) > 3:
        parts[-1] = w[:-3] + "f"
    elif w.endswith("oes"):
        parts[-1] = w[:-2]
    elif any(w.endswith(suffix) for suffix in ("ses", "xes", "zes", "ches", "shes")):
        parts[-1] = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        parts[-1] = w[:-1]

    return "_".join(parts)


def singular_to_plural(word: str) -> str:
    """
    Convert a singular English noun (possibly underscore-separated) to plural.
    Applies only to the last word and handles common irregulars.
    """
    irregulars = {
        "universe": "universes",
        "child": "children",
        "man": "men",
        "woman": "women",
        "person": "people",
        "mouse": "mice",
        "goose": "geese",
        "tooth": "teeth",
        "foot": "feet",
        "index": "indices",
        "matrix": "matrices",
        "criterion": "criteria",
        "data": "data",  # invariant
        "spectrum": "spectra",
    }

    parts = word.lower().split("_")
    w = parts[-1]

    if w in irregulars:
        parts[-1] = irregulars[w]
    elif w.endswith("y") and w[-2:] not in ("ay", "ey", "iy", "oy", "uy"):
        parts[-1] = w[:-1] + "ies"
    elif w.endswith("f"):
        parts[-1] = w[:-1] + "ves"
    elif w.endswith("fe"):
        parts[-1] = w[:-2] + "ves"
    elif w.endswith(("s", "x", "z", "ch", "sh")):
        parts[-1] = w + "es"
    else:
        parts[-1] = w + "s"

    return "_".join(parts)


# ==============================================================================
# MC/DC Member Array Sizes
# ==============================================================================


def literalize(value):
    jit_str = f"@njit\ndef impl():\n    return {value}\n"
    exec(jit_str, globals(), locals())
    return eval("impl")


def rpn_buffer_size():
    pass


def make_size_rpn(cells):
    global rpn_buffer_size
    size = max([np.sum(np.array(x.region_RPN_tokens) >= 0.0) for x in cells])
    rpn_buffer_size = literalize(size)


# ======================================================================================
# Make literals
# ======================================================================================


def make_literals(simulation):
    # Sizes
    rpn_evaluation_buffer_size = int(
        max([np.sum(np.array(x.region_RPN_tokens) >= 0.0) for x in simulation.cells])
    )

    path = f"{Path(mcdc.__file__).parent}"
    with open(f"{path}/transport/literals.py", "w") as f:
        text = "# The following is automatically generated by code_factory.py\n\n"

        text += f"rpn_evaluation_buffer_size = {rpn_evaluation_buffer_size}\n"

        f.write(text)
