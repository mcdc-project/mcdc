from numba import njit


@njit
def atomic_densities_length(material):
    return int(material["atomic_densities_length"])


@njit
def atomic_densities_all(material, data):
    start = material["atomic_densities_offset"]
    end = start + material["atomic_densities_length"]
    return data[start:end]


@njit
def atomic_densities(index, material, data):
    offset = material["atomic_densities_offset"]
    return data[offset + index]


@njit
def atomic_densities_chunk(start, size, material, data):
    start += material["atomic_densities_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_speed_length(material):
    return int(material["mgxs_speed_length"])


@njit
def mgxs_speed_all(material, data):
    start = material["mgxs_speed_offset"]
    end = start + material["mgxs_speed_length"]
    return data[start:end]


@njit
def mgxs_speed(index, material, data):
    offset = material["mgxs_speed_offset"]
    return data[offset + index]


@njit
def mgxs_speed_chunk(start, size, material, data):
    start += material["mgxs_speed_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_decay_rate_length(material):
    return int(material["mgxs_decay_rate_length"])


@njit
def mgxs_decay_rate_all(material, data):
    start = material["mgxs_decay_rate_offset"]
    end = start + material["mgxs_decay_rate_length"]
    return data[start:end]


@njit
def mgxs_decay_rate(index, material, data):
    offset = material["mgxs_decay_rate_offset"]
    return data[offset + index]


@njit
def mgxs_decay_rate_chunk(start, size, material, data):
    start += material["mgxs_decay_rate_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_capture_length(material):
    return int(material["mgxs_capture_length"])


@njit
def mgxs_capture_all(material, data):
    start = material["mgxs_capture_offset"]
    end = start + material["mgxs_capture_length"]
    return data[start:end]


@njit
def mgxs_capture(index, material, data):
    offset = material["mgxs_capture_offset"]
    return data[offset + index]


@njit
def mgxs_capture_chunk(start, size, material, data):
    start += material["mgxs_capture_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_scatter_length(material):
    return int(material["mgxs_scatter_length"])


@njit
def mgxs_scatter_all(material, data):
    start = material["mgxs_scatter_offset"]
    end = start + material["mgxs_scatter_length"]
    return data[start:end]


@njit
def mgxs_scatter(index, material, data):
    offset = material["mgxs_scatter_offset"]
    return data[offset + index]


@njit
def mgxs_scatter_chunk(start, size, material, data):
    start += material["mgxs_scatter_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_fission_length(material):
    return int(material["mgxs_fission_length"])


@njit
def mgxs_fission_all(material, data):
    start = material["mgxs_fission_offset"]
    end = start + material["mgxs_fission_length"]
    return data[start:end]


@njit
def mgxs_fission(index, material, data):
    offset = material["mgxs_fission_offset"]
    return data[offset + index]


@njit
def mgxs_fission_chunk(start, size, material, data):
    start += material["mgxs_fission_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_total_length(material):
    return int(material["mgxs_total_length"])


@njit
def mgxs_total_all(material, data):
    start = material["mgxs_total_offset"]
    end = start + material["mgxs_total_length"]
    return data[start:end]


@njit
def mgxs_total(index, material, data):
    offset = material["mgxs_total_offset"]
    return data[offset + index]


@njit
def mgxs_total_chunk(start, size, material, data):
    start += material["mgxs_total_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_nu_s_length(material):
    return int(material["mgxs_nu_s_length"])


@njit
def mgxs_nu_s_all(material, data):
    start = material["mgxs_nu_s_offset"]
    end = start + material["mgxs_nu_s_length"]
    return data[start:end]


@njit
def mgxs_nu_s(index, material, data):
    offset = material["mgxs_nu_s_offset"]
    return data[offset + index]


@njit
def mgxs_nu_s_chunk(start, size, material, data):
    start += material["mgxs_nu_s_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_nu_p_length(material):
    return int(material["mgxs_nu_p_length"])


@njit
def mgxs_nu_p_all(material, data):
    start = material["mgxs_nu_p_offset"]
    end = start + material["mgxs_nu_p_length"]
    return data[start:end]


@njit
def mgxs_nu_p(index, material, data):
    offset = material["mgxs_nu_p_offset"]
    return data[offset + index]


@njit
def mgxs_nu_p_chunk(start, size, material, data):
    start += material["mgxs_nu_p_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_nu_d_vector(index_1, material, data):
    offset = material["mgxs_nu_d_offset"]
    stride = material["J"]
    start = offset + index_1 * stride
    end = start + stride
    return data[start:end]


@njit
def mgxs_nu_d(index_1, index_2, material, data):
    offset = material["mgxs_nu_d_offset"]
    stride = material["J"]
    return data[offset + index_1 * stride + index_2]


@njit
def mgxs_nu_d_chunk(start, size, material, data):
    start += material["mgxs_nu_d_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_nu_d_total_length(material):
    return int(material["mgxs_nu_d_total_length"])


@njit
def mgxs_nu_d_total_all(material, data):
    start = material["mgxs_nu_d_total_offset"]
    end = start + material["mgxs_nu_d_total_length"]
    return data[start:end]


@njit
def mgxs_nu_d_total(index, material, data):
    offset = material["mgxs_nu_d_total_offset"]
    return data[offset + index]


@njit
def mgxs_nu_d_total_chunk(start, size, material, data):
    start += material["mgxs_nu_d_total_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_nu_f_length(material):
    return int(material["mgxs_nu_f_length"])


@njit
def mgxs_nu_f_all(material, data):
    start = material["mgxs_nu_f_offset"]
    end = start + material["mgxs_nu_f_length"]
    return data[start:end]


@njit
def mgxs_nu_f(index, material, data):
    offset = material["mgxs_nu_f_offset"]
    return data[offset + index]


@njit
def mgxs_nu_f_chunk(start, size, material, data):
    start += material["mgxs_nu_f_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_chi_s_vector(index_1, material, data):
    offset = material["mgxs_chi_s_offset"]
    stride = material["G"]
    start = offset + index_1 * stride
    end = start + stride
    return data[start:end]


@njit
def mgxs_chi_s(index_1, index_2, material, data):
    offset = material["mgxs_chi_s_offset"]
    stride = material["G"]
    return data[offset + index_1 * stride + index_2]


@njit
def mgxs_chi_s_chunk(start, size, material, data):
    start += material["mgxs_chi_s_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_chi_p_vector(index_1, material, data):
    offset = material["mgxs_chi_p_offset"]
    stride = material["G"]
    start = offset + index_1 * stride
    end = start + stride
    return data[start:end]


@njit
def mgxs_chi_p(index_1, index_2, material, data):
    offset = material["mgxs_chi_p_offset"]
    stride = material["G"]
    return data[offset + index_1 * stride + index_2]


@njit
def mgxs_chi_p_chunk(start, size, material, data):
    start += material["mgxs_chi_p_offset"]
    end = start + size
    return data[start:end]


@njit
def mgxs_chi_d_vector(index_1, material, data):
    offset = material["mgxs_chi_d_offset"]
    stride = material["G"]
    start = offset + index_1 * stride
    end = start + stride
    return data[start:end]


@njit
def mgxs_chi_d(index_1, index_2, material, data):
    offset = material["mgxs_chi_d_offset"]
    stride = material["G"]
    return data[offset + index_1 * stride + index_2]


@njit
def mgxs_chi_d_chunk(start, size, material, data):
    start += material["mgxs_chi_d_offset"]
    end = start + size
    return data[start:end]
