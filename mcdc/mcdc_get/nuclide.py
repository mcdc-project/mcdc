from numba import njit


@njit
def from_material(index, material, mcdc, data):
    offset = material["nuclide_index_offset"]
    nuclide_ID = int(data[offset + index])
    return mcdc["nuclides"][nuclide_ID]
