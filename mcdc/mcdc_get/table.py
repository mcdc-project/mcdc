from numba import njit


@njit
def x_length(table):
    return int(table["x_length"])


@njit
def x_all(table, data):
    start = table["x_offset"]
    end = start + table["x_length"]
    return data[start:end]


@njit
def x(index, table, data):
    offset = table["x_offset"]
    return data[offset + index]


@njit
def x_chunk(start, size, table, data):
    start += table["x_offset"]
    end = start + size
    return data[start:end]


@njit
def y_length(table):
    return int(table["y_length"])


@njit
def y_all(table, data):
    start = table["y_offset"]
    end = start + table["y_length"]
    return data[start:end]


@njit
def y(index, table, data):
    offset = table["y_offset"]
    return data[offset + index]


@njit
def y_chunk(start, size, table, data):
    start += table["y_offset"]
    end = start + size
    return data[start:end]
