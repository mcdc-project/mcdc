import numpy as np
import math

from numba import njit

####

import mcdc.mcdc_get.weight_windows as ww_get
import mcdc.mcdc_get.tally as tally_get
import mcdc.mcdc_set.weight_windows as ww_set
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    COINCIDENCE_TOLERANCE_TIME,
    EVENT_TIME_CENSUS,
    PARTICLE_NEUTRON,
    PARTICLE_ELECTRON,
    PARTICLE_PROTON,
    PARTICLE_TYPE_NAME_PAIRS
)
from mcdc.transport.mesh import get_indices as get_mesh_indices

# ======================================================================================
# Weight Roulette
# ======================================================================================


@njit
def weight_roulette(particle_container, w_threshold, w_target):
    """
    Russian roulette particle if weight is below threshold.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    w_threshold : float
        Lower weight bound triggering roulette.
    w_target : float
        Target weight assigned upon survival.
    """
    particle = particle_container[0]
    if particle["w"] < w_threshold:
        survival_probability = particle["w"] / w_target
        # sample random number to determine survival
        if rng.lcg(particle_container) < survival_probability:
            particle["w"] = w_target
        else:
            particle["alive"] = False


# ======================================================================================
# Global weight Roulette
# ======================================================================================


@njit
def global_weight_roulette(particle_container, simulation):
    """
    Russian roulette particle with the global weight roulette parameters.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    simulation : object
        Simulation state containing global weight roulette parameters.
    """
    w_threshold = simulation["technique"]["global_weight_roulette"]["weight_threshold"]
    w_target = simulation["technique"]["global_weight_roulette"]["weight_target"]
    weight_roulette(particle_container, w_threshold, w_target)


# ======================================================================================
# Weight Windows
# ======================================================================================


@njit
def get_weight_window_object(ptype, program):
    """
    Get the appropriate weight window object that applies to the provided
    particle.

    Parameters
    ----------
    ptype : int
        Type of particle to get.
    program : object
        Program object containing simulation state with weight window objects.

    Returns
    -------
    ww_obj : object
        The weight window object that corresponds to the provided particle.
    """
    simulation = util.access_simulation(program)
    technique = simulation["technique"]
    if ptype == PARTICLE_NEUTRON:
        ww_obj = technique["neutron_weight_windows"]
    elif ptype == PARTICLE_ELECTRON:
        ww_obj = technique["electron_weight_windows"]
    elif ptype == PARTICLE_PROTON:
        ww_obj = technique["proton_weight_windows"]

    return ww_obj


@njit
def weight_windows(particle_container, program, data):
    """
    Apply weight window splitting and rouletting to a particle.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    program : object
        Program object containing simulation state with weight window and mesh data.
    data : object
        Simulation data for array access.
    """
    [lower, target, upper] = query_weight_window(particle_container, program, data)
    # split
    split_from_weight_window(particle_container, upper, target, lower, program)
    # roulette original particle
    weight_roulette(particle_container, lower, target)


@njit
def query_weight_window(particle_container, program, data):
    """
    Query weight window bounds for the particle.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    program : object
        Program object containing simulation state with weight window and mesh data.
    data : object
        Simulation data for array access.

    Returns
    -------
    lower : float
        Lower weight bound.
    target : float
        Target weight.
    upper : float
        Upper weight bound.
    """
    # grab objects
    ww_obj = get_weight_window_object(particle_container[0]["particle_type"], program)
    indices = get_ww_indices(particle_container, ww_obj, program, data)
    # grab the actual ww parameters
    lower = ww_get.weights(*indices, 0, ww_obj, data)
    target = ww_get.weights(*indices, 1, ww_obj, data)
    upper = ww_get.weights(*indices, 2, ww_obj, data)
    return lower, target, upper


@njit
def get_ww_indices(particle_container, ww_obj, program, data):
    """
    Get the particle's bin index in each weight-window dimension.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    ww_obj : object
        The weight window object containing index information.
    program : object
        Program object containing simulation state with weight window and mesh data.
    data : object
        Simulation data for array access.

    Returns
    -------
    indices : tuple of int
        Seven bin indices in (time, energy, mu, azimuthal, x, y, z) order.
    """
    simulation = util.access_simulation(program)
    particle = particle_container[0]

    # get time index
    time_bounds = ww_get.time_bounds_all(ww_obj, data)
    it = util.find_bin_with_rules(
        particle["t"], time_bounds, COINCIDENCE_TOLERANCE_TIME, False
    )

    # get energy index
    energy_bounds = ww_get.energy_bounds_all(ww_obj, data)
    energy = particle["E"]
    ie = util.find_bin(energy, energy_bounds)

    # get angular indices
    mu, azimuthal = util.calculate_angles(particle_container, ww_obj["polar_reference"])

    # mu
    mu_bounds = ww_get.mu_bounds_all(ww_obj, data)
    imu = util.find_bin(mu, mu_bounds)

    # azimuthal
    azi_bounds = ww_get.azi_bounds_all(ww_obj, data)
    ia = util.find_bin(azimuthal, azi_bounds)

    # get spatial index
    mesh = simulation["meshes"][ww_obj["mesh_ID"]]
    idx, idy, idz = get_mesh_indices(particle_container, mesh, simulation, data)

    return (it, ie, imu, ia, idx, idy, idz)


@njit
def split_from_weight_window(particle_container, w_upper, w_target, w_lower, program):
    """
    Split a particle if its weight exceeds the threshold.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    w_upper : float
        Upper weight bound triggering splitting.
    w_target : float
        Target weight to assign to split particles.
    w_lower : float
        Lower weight bound triggering roulette on residual particle.
    program : object
        Program object containing simulation state with access to active bank.
    """
    particle = particle_container[0]
    weight = particle["w"]
    if weight > w_upper:
        # determine how many to split into
        num_split_to_target = math.floor(weight / w_target)

        # bank target particles
        particle["w"] = w_target
        for _ in range(num_split_to_target - 1):
            container_copy = util.local_array(1, type_.particle)
            particle_module.copy_as_child(container_copy, particle_container)
            if particle["event"] & EVENT_TIME_CENSUS:
                particle_bank_module.bank_census_particle(container_copy, program)
            else:
                particle_bank_module.bank_active_particle(container_copy, program)

        # bank residual particle
        residual_weight = weight - num_split_to_target * w_target
        if residual_weight > 0.0:
            residual_copy = util.local_array(1, type_.particle)
            particle_module.copy_as_child(residual_copy, particle_container)
            residual_copy[0]["w"] = residual_weight
            residual_copy[0]["alive"] = True
            weight_roulette(residual_copy, w_lower, w_target)
            if residual_copy[0]["alive"]:
                if particle["event"] & EVENT_TIME_CENSUS:
                    particle_bank_module.bank_census_particle(residual_copy, program)
                else:
                    particle_bank_module.bank_active_particle(residual_copy, program)


# ======================================================================================
# Population Control
# ======================================================================================


@njit
def weight_window_generator(program, data):
    simulation = util.access_simulation(program)
    technique = simulation["technique"]

    for ptype, pname in PARTICLE_TYPE_NAME_PAIRS:
        wwg = get_weight_window_generator(ptype, technique)
        if not wwg["active"]:
            continue

        ww = get_weight_window_object(ptype, program)
        flux_tally = simulation["tallies"][wwg["flux_tally_ID"]]
        if wwg["magic_active"]:
            MAGIC_update(flux_tally, ww, wwg, data)

@njit
def get_weight_window_generator(ptype, technique):
    if ptype == PARTICLE_NEUTRON:
        wwg = technique["neutron_weight_window_generator"]
    elif ptype == PARTICLE_ELECTRON:
        wwg = technique["electron_weight_window_generator"]
    elif ptype == PARTICLE_PROTON:
        wwg = technique["proton_weight_window_generator"]

    return wwg


@njit 
def MAGIC_update(flux_tally, weight_window_object, weight_window_generator, data):
    max_flux = max(tally_get.bin_mean_all(flux_tally, data))
    lower_scale = 1 / (2 * max_flux)
    target_scale = weight_window_generator["target_scale"]
    upper_scale = weight_window_generator["upper_scale"]

    for it in range(weight_window_object["Nt"]):
        for ie in range(weight_window_object["Ne"]):
            for imu in range(weight_window_object["Nmu"]):
                for ia in range(weight_window_object["Na"]):
                    for ix in range(weight_window_object["Nx"]):
                        for iy in range(weight_window_object["Ny"]):
                            for iz in range(weight_window_object["Nz"]):
                                flat_index = int(ww_get.weights_flat_index(it, ie, imu, ia, ix, iy, iz, 0, weight_window_object) / 3)
                                flux = tally_get.bin_mean(flat_index, flux_tally, data)
                                ww_value = flux * lower_scale
                                if ww_value <= 0.0:
                                    continue
                                ww_set.weights(it, ie, imu, ia, ix, iy, iz, 0, weight_window_object, data, ww_value)
                                ww_set.weights(it, ie, imu, ia, ix, iy, iz, 1, weight_window_object, data, ww_value * target_scale)
                                ww_set.weights(it, ie, imu, ia, ix, iy, iz, 2, weight_window_object, data, ww_value * upper_scale)


# ======================================================================================
# Population Control
# ======================================================================================


@njit
def population_control(simulation):
    """Uniform Splitting-Roulette technique"""

    bank_census = simulation["bank_census"]
    M = simulation["settings"]["N_particle"]
    bank_source = simulation["bank_source"]

    # Scan the bank
    idx_start, N_local, N = particle_bank_module.bank_scanning(bank_census, simulation)
    idx_end = idx_start + N_local

    # Abort if census bank is empty
    if N == 0:
        return

    # Weight scaling
    ws = float(N) / float(M)

    # Splitting Number
    sn = 1.0 / ws

    P_rec_arr = util.local_array(1, type_.particle_data)
    P_rec = P_rec_arr[0]

    # Perform split-roulette to all particles in local bank
    particle_bank_module.set_bank_size(bank_source, 0)
    for idx in range(N_local):
        # Weight of the surviving particles
        w = bank_census["particle_data"][idx]["w"]
        w_survive = w * ws

        # Determine number of guaranteed splits
        N_split = math.floor(sn)

        # Survive the russian roulette?
        xi = rng.lcg(bank_census["particle_data"][idx : idx + 1])
        if xi < sn - N_split:
            N_split += 1

        # Split the particle
        for i in range(N_split):
            particle_module.copy_as_child(
                P_rec_arr, bank_census["particle_data"][idx : idx + 1]
            )
            # Set weight
            P_rec["w"] = w_survive
            particle_bank_module.bank_source_particle(P_rec_arr, simulation)
