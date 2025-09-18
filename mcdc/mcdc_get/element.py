from numba import njit


@njit
def from_material(index, material, mcdc, data):
    offset = material["element_index_offset"]
    element_ID = int(data[offset + index])
    return mcdc["elements"][element_ID]


@njit
def xs_energy_grid_length(element):
    return int(element["xs_energy_grid_length"])


@njit
def xs_energy_grid_all(element, data):
    start = element["xs_energy_grid_offset"]
    end = start + element["xs_energy_grid_length"]
    return data[start:end]


@njit
def xs_energy_grid(index, element, data):
    offset = element["xs_energy_grid_offset"]
    return data[offset + index]


@njit
def xs_energy_grid_chunk(start, size, element, data):
    start += element["xs_energy_grid_offset"]
    end = start + size
    return data[start:end]


@njit
def total_xs_length(element):
    return int(element["total_xs_length"])


@njit
def total_xs_all(element, data):
    start = element["total_xs_offset"]
    end = start + element["total_xs_length"]
    return data[start:end]


@njit
def total_xs(index, element, data):
    offset = element["total_xs_offset"]
    return data[offset + index]


@njit
def total_xs_chunk(start, size, element, data):
    start += element["total_xs_offset"]
    end = start + size
    return data[start:end]


@njit
def reaction_type_length(element):
    return int(element["reaction_type_length"])


@njit
def reaction_type_all(element, data):
    start = element["reaction_type_offset"]
    end = start + element["reaction_type_length"]
    return data[start:end]


@njit
def reaction_type(index, element, data):
    offset = element["reaction_type_offset"]
    return data[offset + index]


@njit
def reaction_type_chunk(start, size, element, data):
    start += element["reaction_type_offset"]
    end = start + size
    return data[start:end]


@njit
def reaction_index_length(element):
    return int(element["reaction_index_length"])


@njit
def reaction_index_all(element, data):
    start = element["reaction_index_offset"]
    end = start + element["reaction_index_length"]
    return data[start:end]


@njit
def reaction_index(index, element, data):
    offset = element["reaction_index_offset"]
    return data[offset + index]


@njit
def reaction_index_chunk(start, size, element, data):
    start += element["reaction_index_offset"]
    end = start + size
    return data[start:end]
