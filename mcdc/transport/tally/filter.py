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
from mcdc.transport.util import find_bin_with_tolerance, find_bin_with_rules


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

    grid_mu = data[tally["mu_offset"] : (tally["mu_offset"] + tally["mu_length"])]
    # Above is equivalent to: grid_mu = mcdc_get.tally.mu_all(tally, data)
    grid_azi = data[tally["azi_offset"] : (tally["azi_offset"] + tally["azi_length"])]
    # Above is equivalent to: grid_azi = mcdc_get.tally.azi_all(tally, data)

    i_mu = find_bin_with_tolerance(mu, grid_mu, tolerance)
    i_azi = find_bin_with_tolerance(azi, grid_azi, tolerance)
    return i_mu, i_azi


@njit
def get_energy_index(particle_container, tally, data, neutron_multigroup_mode):
    particle = particle_container[0]

    if neutron_multigroup_mode:
        E = particle["g"]
    else:
        E = particle["E"]

    tolerance = COINCIDENCE_TOLERANCE_ENERGY
    grid_energy = data[
        tally["energy_offset"] : (tally["energy_offset"] + tally["energy_length"])
    ]
    # Above is equivalent to: grid_energy = mcdc_get.tally.energy_all(tally, data)

    return find_bin_with_tolerance(E, grid_energy, tolerance)


@njit
def get_time_index(particle_container, tally, data):
    particle = particle_container[0]

    # Particle properties
    time = particle["t"]

    grid_time = data[
        tally["time_offset"] : (tally["time_offset"] + tally["time_length"])
    ]
    # Above is equivalent to: grid_time = mcdc_get.tally.time_all(tally, data)

    tolerance = COINCIDENCE_TOLERANCE_TIME
    go_lower = False
    return find_bin_with_rules(time, grid_time, tolerance, go_lower)


@njit
def set_census_based_time_grid(simulation, data):
    settings = simulation["settings"]
    tally_frequency = settings["census_tally_frequency"]
    idx_census = simulation["idx_census"]

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
    for tally in simulation["tallies"]:
        mcdc_set.tally.time(0, tally, data, t_start)
        for j in range(tally_frequency):
            t_next = mcdc_get.tally.time(j, tally, data) + dt
            mcdc_set.tally.time(j + 1, tally, data, t_next)
