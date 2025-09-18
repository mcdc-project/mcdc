import numpy as np

####

import mcdc.objects as objects

from mcdc.data_container import DataContainer, DataMaxwellian, DataMultiPDF, DataPolynomial, DataTable
from mcdc.objects import (
    ObjectNonSingleton,
    ObjectPolymorphic,
    ObjectSingleton,
)
from mcdc.reaction import ReactionNeutronFission

# ======================================================================================
# Python object to Numba structured data converter
# ======================================================================================

type_map = {
    np.float64: "f8",
    float: "f8",
    str: "U32",
    np.int64: "f8",
    int: "i8",
    bool: "?",
}


def numbafy_object(object_, structures, records, data):
    # Skip if already numbafied
    if object_.numbafied:
        return

    structure = []
    record = ()

    # Loop over data attributes of the object
    attribute_names = [
        x
        for x in dir(object_)
        if (
            x[:2] != "__"
            and not callable(getattr(object_, x))
            and x
            not in [
                "label",
                "numbafied",
                "ID",
                "type",
                "nuclide_composition",
                "ID_numba",
            ]
        )
    ]
    for attribute_name in attribute_names:
        attribute = getattr(object_, attribute_name)

        # Scalar
        if type(attribute) in type_map.keys():
            structure.append((attribute_name, type_map[type(attribute)]))
            record += (attribute,)

        # Numpy array
        elif type(attribute) == np.ndarray:
            structure.append((f"{attribute_name}_offset", "i8"))
            structure.append((f"{attribute_name}_length", "i8"))

            offset = len(data)
            length = len(attribute.flatten())
            record += (offset, length)

            data.extend(attribute.flatten())

        # Data
        elif isinstance(attribute, DataContainer):
            numbafy_object(attribute, structures, records, data)
            structure.append((f"{attribute_name}_type", "i8"))
            structure.append((f"{attribute_name}_index", "i8"))
            record += (attribute.type, attribute.ID_numba)

        # List of objects
        elif type(attribute) == list:
            if not isinstance(attribute[0], ObjectNonSingleton):
                print(
                    f"[ERROR] get a list of non-object for {attribute_name}: {attribute}"
                )
                exit()

            # List of non-polymorphic objects
            if not isinstance(attribute[0], ObjectPolymorphic):
                structure.append((f"N_{attribute_name[:-1]}", "i8"))
                structure.append((f"{attribute_name[:-1]}_index_offset", "i8"))

                length = len(attribute)
                offset = len(data)
                record += (length, offset)

                data.extend([-1] * length)
                for i, subobject in enumerate(attribute):
                    # Generate the numba object
                    if not subobject.numbafied:
                        numbafy_object(subobject, structures, records, data)
                    data[offset + i] = subobject.ID_numba

            # List of polymorphic objects
            else:
                structure.append((f"N_{attribute_name[:-1]}", "i8"))
                structure.append((f"{attribute_name[:-1]}_type_offset", "i8"))
                structure.append((f"{attribute_name[:-1]}_index_offset", "i8"))

                length = len(attribute)
                offset_type = len(data)
                offset_id = offset_type + length
                record += (length, offset_type, offset_id)

                data.extend([-1] * length * 2)
                for i, subobject in enumerate(attribute):
                    # Generate the numba object
                    if not subobject.numbafied:
                        numbafy_object(subobject, structures, records, data)
                    data[offset_type + i] = subobject.type
                    data[offset_id + i] = subobject.ID_numba

        # Dictionary
        else:
            print(f"[ERROR] Unsupported attribute: {attribute_name}: {attribute}")
            exit()

    # Register the numbafied object
    object_.numbafied = True
    structures[object_.label] = np.dtype(structure)
    if isinstance(object_, ObjectSingleton):
        records[object_.label] = record
    elif isinstance(object_, ObjectNonSingleton):
        object_.ID_numba = len(records[object_.label])
        records[object_.label].append(record)


def generate_numba_objects():
    object_list = (
        objects.materials
        + objects.nuclides
        + objects.reactions
        + [objects.settings]
        + objects.data_containers
    )

    # Create necessary dummies
    if not objects.settings.multigroup_mode:
        vector = np.zeros(1)
        if not any([isinstance(x, DataPolynomial) for x in object_list]):
            polynomial = DataPolynomial(vector)
            object_list += [polynomial]
            data_1d = polynomial
        if not any([isinstance(x, DataTable) for x in object_list]):
            table = DataTable(vector, vector)
            object_list += [table]
            data_1d = table
        if not any([isinstance(x, DataMultiPDF) for x in object_list]):
            multipdf = DataMultiPDF(vector, vector, vector, vector)
            object_list += [multipdf]
            distribution = multipdf
        if not any([isinstance(x, DataMaxwellian) for x in object_list]):
            maxwellian = DataMaxwellian(0.0, vector, vector)
            object_list += [maxwellian]
            distribution = maxwellian
        if not any([isinstance(x, ReactionNeutronFission) for x in object_list]):
            fission = ReactionNeutronFission(vector, data_1d, distribution, [data_1d], [distribution], vector)
            object_list += [fission]

    # Containers for structures and records for all object types
    structures = {}
    records = {}
    for object_ in object_list:
        structures[object_.label] = []
        if isinstance(object_, ObjectNonSingleton):
            records[object_.label] = []
    data = []

    # Loop over all objects
    for object_ in object_list:
        numbafy_object(object_, structures, records, data)

    data = np.array(data, dtype=np.float64)

    return data, structures, records
