from numba import njit


@njit
def census_time_all(settings, data):
    start = settings["census_time_offset"]
    end = start + settings["census_time_length"]
    return data[start:end]


@njit
def census_time(index, settings, data):
    offset = settings["census_time_offset"]
    return data[offset + index]


@njit
def census_time_chunk(start, size, settings, data):
    start += settings["census_time_offset"]
    end = start + size
    return data[start:end]
