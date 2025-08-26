from numba import njit


@njit
def mu_energy_grid_all(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["mu_energy_grid_offset"]
    end = start + neutron_elastic_scattering_reaction["mu_energy_grid_length"]
    return data[start:end]


@njit
def mu_energy_grid(index, neutron_elastic_scattering_reaction, data):
    offset = neutron_elastic_scattering_reaction["mu_energy_grid_offset"]
    return data[offset + index]


@njit
def mu_energy_grid_chunk(start, size, neutron_elastic_scattering_reaction, data):
    start += neutron_elastic_scattering_reaction["mu_energy_grid_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_energy_offset_all(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["mu_energy_offset_offset"]
    end = start + neutron_elastic_scattering_reaction["mu_energy_offset_length"]
    return data[start:end]


@njit
def mu_energy_offset(index, neutron_elastic_scattering_reaction, data):
    offset = neutron_elastic_scattering_reaction["mu_energy_offset_offset"]
    return data[offset + index]


@njit
def mu_energy_offset_chunk(start, size, neutron_elastic_scattering_reaction, data):
    start += neutron_elastic_scattering_reaction["mu_energy_offset_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_all(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["mu_offset"]
    end = start + neutron_elastic_scattering_reaction["mu_length"]
    return data[start:end]


@njit
def mu(index, neutron_elastic_scattering_reaction, data):
    offset = neutron_elastic_scattering_reaction["mu_offset"]
    return data[offset + index]


@njit
def mu_chunk(start, size, neutron_elastic_scattering_reaction, data):
    start += neutron_elastic_scattering_reaction["mu_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_PDF_all(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["mu_PDF_offset"]
    end = start + neutron_elastic_scattering_reaction["mu_PDF_length"]
    return data[start:end]


@njit
def mu_PDF(index, neutron_elastic_scattering_reaction, data):
    offset = neutron_elastic_scattering_reaction["mu_PDF_offset"]
    return data[offset + index]


@njit
def mu_PDF_chunk(start, size, neutron_elastic_scattering_reaction, data):
    start += neutron_elastic_scattering_reaction["mu_PDF_offset"]
    end = start + size
    return data[start:end]


@njit
def mu_CDF_all(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["mu_CDF_offset"]
    end = start + neutron_elastic_scattering_reaction["mu_CDF_length"]
    return data[start:end]


@njit
def mu_CDF(index, neutron_elastic_scattering_reaction, data):
    offset = neutron_elastic_scattering_reaction["mu_CDF_offset"]
    return data[offset + index]


@njit
def mu_CDF_chunk(start, size, neutron_elastic_scattering_reaction, data):
    start += neutron_elastic_scattering_reaction["mu_CDF_offset"]
    end = start + size
    return data[start:end]
