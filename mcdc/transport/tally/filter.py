import math

from numba import literal_unroll, njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.mcdc_set as mcdc_set

from mcdc.constant import (
    COINCIDENCE_TOLERANCE_DIRECTION,
    COINCIDENCE_TOLERANCE_ENERGY,
    COINCIDENCE_TOLERANCE_TIME,
)
from mcdc.transport.util import find_bin


@njit
def get_filter_indices(particle_container, tally, data, MG_mode):
    i_mu, i_azi, i_energy, i_time = 0, 0, 0, 0

    if tally["filter_direction"]:
        i_mu, i_azi = get_direction_index(particle_container, tally, data)

    if tally["filter_energy"]:
        i_energy = get_energy_index(particle_container, tally, data, MG_mode)

    if tally["filter_time"]:
        i_time = get_time_index(particle_container, tally, data)

    return i_mu, i_azi, i_energy, i_time


@njit
def get_direction_index(particle_container, tally, data):
    particle = particle_container[0]

    # Particle properties
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Polar reference
    nx = tally["polar_reference"][0]
    ny = tally["polar_reference"][1]
    nz = tally["polar_reference"][2]

    # TODO: Rotate direction based on the polar reference
    if nz != 1.0:
        pass

    mu = uz
    azi = math.acos(ux / math.sqrt(ux * ux + uy * uy))
    if uy < 0.0:
        azi *= -1

    tolerance = COINCIDENCE_TOLERANCE_DIRECTION
    i_mu = find_bin(mu, mcdc_get.tally.mu_all(tally, data), tolerance)
    i_azi = find_bin(azi, mcdc_get.tally.azi_all(tally, data), tolerance)
    return i_mu, i_azi


@njit
def get_energy_index(particle_container, tally, data, multigroup_mode):
    particle = particle_container[0]

    if multigroup_mode:
        E = particle["g"]
    else:
        E = particle["E"]

    tolerance = COINCIDENCE_TOLERANCE_ENERGY
    return find_bin(E, mcdc_get.tally.energy_all(tally, data), tolerance)


@njit
def get_time_index(particle_container, tally, data):
    particle = particle_container[0]

    # Particle properties
    time = particle["t"]

    tolerance = COINCIDENCE_TOLERANCE_TIME
    return find_bin(
        time, mcdc_get.tally.time_all(tally, data), tolerance, go_lower=False
    )


@njit
def set_census_based_time_grid(mcdc, data):
    settings = mcdc["settings"]
    tally_frequency = settings["census_tally_frequency"]
    idx_census = mcdc["idx_census"]

    # Starting time
    if idx_census == 0:
        t_start = 0.0
    else:
        t_start = mcdc_get.settings.census_time(idx_census - 1, settings, data)

    # Ending time
    t_end = mcdc_get.settings.census_time(idx_census, settings, data)

    # Time grid width
    dt = (t_end - t_start) / tally_frequency

    # Set the time grid to all tallies
    for tally in mcdc["tallies"]:
        mcdc_set.tally.time(0, tally, data, t_start)
        for j in range(tally_frequency):
            t_next = mcdc_get.tally.time(j, tally, data) + dt
            mcdc_set.tally.time(j + 1, tally, data, t_next)
