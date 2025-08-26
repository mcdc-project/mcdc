from numba import njit


@njit
def xs_all(reaction, data):
    start = reaction["xs_offset"]
    end = start + reaction["xs_length"]
    return data[start:end]


@njit
def xs(index, reaction, data):
    offset = reaction["xs_offset"]
    return data[offset + index]


@njit
def xs_chunk(start, size, reaction, data):
    start += reaction["xs_offset"]
    end = start + size
    return data[start:end]
