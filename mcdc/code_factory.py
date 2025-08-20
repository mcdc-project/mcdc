import numpy as np

####

from mcdc.settings import Settings


# ======================================================================================
# Generate object's Numba-compaible data and structure
# ======================================================================================

type_map = {
    np.float64: "f8",
    float: "f8",
    str: "U32",
    int: "i8",
    bool: "?",
}


def numbafy_object(object_, structures, records, data):
    structure = []
    record = ()

    # Loop over data attributes of the object
    attribute_names = [
        x
        for x in dir(object_)
        if (
            x[:2] != "__"
            and x not in ["type", "derived_class", "numba_ID", "type_enum"]
        )
    ]
    for attribute_name in attribute_names:
        attribute = getattr(object_, attribute_name)

        # Simple attribute
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

        # List
        elif type(attribute) == list:
            # List of non-derived class
            if not attribute[0].derived_class:
                structure.append((f"N_{attribute_name[:-1]}", "i8"))
                structure.append((f"{attribute_name[:-1]}_idx_offset", "i8"))

                length = len(attribute)
                offset = len(data)
                record += (length, offset)

                data.extend([-1] * length)
                for i, subobject in enumerate(attribute):
                    # Generate the numba object
                    if subobject.numba_ID == -1:
                        numbafy_object(subobject, structures, records, data)
                    data[offset + i] = subobject.numba_ID

            # List of derived class
            if attribute[0].derived_class:
                structure.append((f"N_{attribute_name[:-1]}", "i8"))
                structure.append((f"{attribute_name[:-1]}_type_offset", "i8"))
                structure.append((f"{attribute_name[:-1]}_idx_offset", "i8"))

                length = len(attribute)
                offset_type = len(data)
                offset_id = offset_type + length
                record += (length, offset_type, offset_id)

                data.extend([-1] * length * 2)
                for i, subobject in enumerate(attribute):
                    # Generate the numba object
                    if subobject.numba_ID == -1:
                        numbafy_object(subobject, structures, records, data)
                    data[offset_type + i] = subobject.type_enum
                    data[offset_id + i] = subobject.numba_ID

        # Others
        else:
            pass

    object_.numba_ID = len(records[object_.type])

    # Register the structure and record
    structures[object_.type] = np.dtype(structure)
    records[object_.type].append(record)


def generate_numba_objects(object_list):
    # Containers for structures and records for all object types
    structures = {}
    records = {}
    for object_ in object_list:
        structures[object_.type] = []
        records[object_.type] = []
    data = []

    # Loop over all objects
    for object_ in object_list:
        # Skip if already generated
        if object_.numba_ID > -1:
            continue

        # Numbafy!
        numbafy_object(object_, structures, records, data)

    # Set the global structure
    global_structure = []
    for key in structures.keys():
        if key in ["settings"]:
            global_structure += [(f"{key}", structures[key])]
        else:
            global_structure += [(f"{key}s", structures[key], (len(records[key]),))]

    # Initialize the global structure
    mcdc = np.zeros((), dtype=global_structure)
    for key in structures.keys():
        if key in ["settings"]:
            mcdc[f"{key}"] = np.array(records[key], dtype=structures[key])
        else:
            mcdc[f"{key}s"] = np.array(records[key], dtype=structures[key])

    data = np.array(data, dtype=np.float64)

    return data, mcdc
