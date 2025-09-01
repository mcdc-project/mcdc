from numba import njit


@njit
def delayed_yield_type_length(fission_reaction):
    return int(fission_reaction["delayed_yield_type_length"])


@njit
def delayed_yield_type_all(fission_reaction, data):
    start = fission_reaction["delayed_yield_type_offset"]
    end = start + fission_reaction["delayed_yield_type_length"]
    return data[start:end]


@njit
def delayed_yield_type(index, fission_reaction, data):
    offset = fission_reaction["delayed_yield_type_offset"]
    return data[offset + index]


@njit
def delayed_yield_type_chunk(start, size, fission_reaction, data):
    start += fission_reaction["delayed_yield_type_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_yield_index_length(fission_reaction):
    return int(fission_reaction["delayed_yield_index_length"])


@njit
def delayed_yield_index_all(fission_reaction, data):
    start = fission_reaction["delayed_yield_index_offset"]
    end = start + fission_reaction["delayed_yield_index_length"]
    return data[start:end]


@njit
def delayed_yield_index(index, fission_reaction, data):
    offset = fission_reaction["delayed_yield_index_offset"]
    return data[offset + index]


@njit
def delayed_yield_index_chunk(start, size, fission_reaction, data):
    start += fission_reaction["delayed_yield_index_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_spectrum_type_length(fission_reaction):
    return int(fission_reaction["delayed_spectrum_type_length"])


@njit
def delayed_spectrum_type_all(fission_reaction, data):
    start = fission_reaction["delayed_spectrum_type_offset"]
    end = start + fission_reaction["delayed_spectrum_type_length"]
    return data[start:end]


@njit
def delayed_spectrum_type(index, fission_reaction, data):
    offset = fission_reaction["delayed_spectrum_type_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_type_chunk(start, size, fission_reaction, data):
    start += fission_reaction["delayed_spectrum_type_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_spectrum_index_length(fission_reaction):
    return int(fission_reaction["delayed_spectrum_index_length"])


@njit
def delayed_spectrum_index_all(fission_reaction, data):
    start = fission_reaction["delayed_spectrum_index_offset"]
    end = start + fission_reaction["delayed_spectrum_index_length"]
    return data[start:end]


@njit
def delayed_spectrum_index(index, fission_reaction, data):
    offset = fission_reaction["delayed_spectrum_index_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_index_chunk(start, size, fission_reaction, data):
    start += fission_reaction["delayed_spectrum_index_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_decay_rate_length(fission_reaction):
    return int(fission_reaction["delayed_decay_rate_length"])


@njit
def delayed_decay_rate_all(fission_reaction, data):
    start = fission_reaction["delayed_decay_rate_offset"]
    end = start + fission_reaction["delayed_decay_rate_length"]
    return data[start:end]


@njit
def delayed_decay_rate(index, fission_reaction, data):
    offset = fission_reaction["delayed_decay_rate_offset"]
    return data[offset + index]


@njit
def delayed_decay_rate_chunk(start, size, fission_reaction, data):
    start += fission_reaction["delayed_decay_rate_offset"]
    end = start + size
    return data[start:end]
