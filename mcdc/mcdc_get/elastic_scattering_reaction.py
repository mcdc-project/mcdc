from numba import njit


@njit
def mu_energy_grid_all(elastic_scattering_reaction, data):
    start = elastic_scattering_reaction["mu_energy_grid_offset"]
    end = start + elastic_scattering_reaction["mu_energy_grid_length"]
    return data[start:end]


@njit
def mu_energy_grid(index, elastic_scattering_reaction, data):
    offset = elastic_scattering_reaction["mu_energy_grid_offset"]
    return data[offset + index]


@njit
def mu_energy_grid_chunk(start, size, elastic_scattering_reaction, data):
    start += elastic_scattering_reaction["mu_energy_grid_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_energy_offset_all(elastic_scattering_reaction, data):
    start = elastic_scattering_reaction["mu_energy_offset_offset"]
    end = start + elastic_scattering_reaction["mu_energy_offset_length"]
    return data[start:end]


@njit
def mu_energy_offset(index, elastic_scattering_reaction, data):
    offset = elastic_scattering_reaction["mu_energy_offset_offset"]
    return data[offset + index]


@njit
def mu_energy_offset_chunk(start, size, elastic_scattering_reaction, data):
    start += elastic_scattering_reaction["mu_energy_offset_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_all(elastic_scattering_reaction, data):
    start = elastic_scattering_reaction["mu_offset"]
    end = start + elastic_scattering_reaction["mu_length"]
    return data[start:end]


@njit
def mu(index, elastic_scattering_reaction, data):
    offset = elastic_scattering_reaction["mu_offset"]
    return data[offset + index]


@njit
def mu_chunk(start, size, elastic_scattering_reaction, data):
    start += elastic_scattering_reaction["mu_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_PDF_all(elastic_scattering_reaction, data):
    start = elastic_scattering_reaction["mu_PDF_offset"]
    end = start + elastic_scattering_reaction["mu_PDF_length"]
    return data[start:end]


@njit
def mu_PDF(index, elastic_scattering_reaction, data):
    offset = elastic_scattering_reaction["mu_PDF_offset"]
    return data[offset + index]


@njit
def mu_PDF_chunk(start, size, elastic_scattering_reaction, data):
    start += elastic_scattering_reaction["mu_PDF_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_CDF_all(elastic_scattering_reaction, data):
    start = elastic_scattering_reaction["mu_CDF_offset"]
    end = start + elastic_scattering_reaction["mu_CDF_length"]
    return data[start:end]


@njit
def mu_CDF(index, elastic_scattering_reaction, data):
    offset = elastic_scattering_reaction["mu_CDF_offset"]
    return data[offset + index]


@njit
def mu_CDF_chunk(start, size, elastic_scattering_reaction, data):
    start += elastic_scattering_reaction["mu_CDF_offset"]
    end = start + size
    return data[start:end]
