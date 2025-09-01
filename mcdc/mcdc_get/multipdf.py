from numba import njit


@njit
def grid_length(multipdf):
    return int(multipdf["grid_length"])


@njit
def grid_all(multipdf, data):
    start = multipdf["grid_offset"]
    end = start + multipdf["grid_length"]
    return data[start:end]


@njit
def grid(index, multipdf, data):
    offset = multipdf["grid_offset"]
    return data[offset + index]


@njit
def grid_chunk(start, size, multipdf, data):
    start += multipdf["grid_offset"]
    end = start + size
    return data[start:end]


@njit
def offset_length(multipdf):
    return int(multipdf["offset_length"])


@njit
def offset_all(multipdf, data):
    start = multipdf["offset_offset"]
    end = start + multipdf["offset_length"]
    return data[start:end]


@njit
def offset(index, multipdf, data):
    offset = multipdf["offset_offset"]
    return data[offset + index]


@njit
def offset_chunk(start, size, multipdf, data):
    start += multipdf["offset_offset"]
    end = start + size
    return data[start:end]


@njit
def value_length(multipdf):
    return int(multipdf["value_length"])


@njit
def value_all(multipdf, data):
    start = multipdf["value_offset"]
    end = start + multipdf["value_length"]
    return data[start:end]


@njit
def value(index, multipdf, data):
    offset = multipdf["value_offset"]
    return data[offset + index]


@njit
def value_chunk(start, size, multipdf, data):
    start += multipdf["value_offset"]
    end = start + size
    return data[start:end]


@njit
def pdf_length(multipdf):
    return int(multipdf["pdf_length"])


@njit
def pdf_all(multipdf, data):
    start = multipdf["pdf_offset"]
    end = start + multipdf["pdf_length"]
    return data[start:end]


@njit
def pdf(index, multipdf, data):
    offset = multipdf["pdf_offset"]
    return data[offset + index]


@njit
def pdf_chunk(start, size, multipdf, data):
    start += multipdf["pdf_offset"]
    end = start + size
    return data[start:end]


@njit
def cdf_length(multipdf):
    return int(multipdf["cdf_length"])


@njit
def cdf_all(multipdf, data):
    start = multipdf["cdf_offset"]
    end = start + multipdf["cdf_length"]
    return data[start:end]


@njit
def cdf(index, multipdf, data):
    offset = multipdf["cdf_offset"]
    return data[offset + index]


@njit
def cdf_chunk(start, size, multipdf, data):
    start += multipdf["cdf_offset"]
    end = start + size
    return data[start:end]
