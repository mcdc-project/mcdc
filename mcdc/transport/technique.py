import numpy as np
import math

from numba import njit

####

import mcdc.mcdc_get.weight_windows as ww_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.transport.mesh import get_indices as get_mesh_indices

# ======================================================================================
# Weight Roulette
# ======================================================================================


@njit
def weight_roulette(particle_container, simulation):
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
    w_threshold = simulation["global_weight_roulette"]["weight_threshold"]
    w_target = simulation["global_weight_roulette"]["weight_target"]
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
    w_threshold = simulation["global_weight_roulette"]["weight_threshold"]
    w_target = simulation["global_weight_roulette"]["weight_target"]
    weight_roulette(particle_container, w_threshold, w_target)


# ======================================================================================
# Weight Windows
# ======================================================================================


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
    simulation = util.access_simulation(program)
    [lower, target, upper] = query_weight_window(particle_container, simulation, data)
    # split
    split_from_weight_window(particle_container, upper, target, lower, program)
    # roulette original particle
    weight_roulette(particle_container, lower, target)


@njit
def query_weight_window(particle_container, simulation, data):
    """
    Query weight window bounds for the particle.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    simulation : object
        Simulation state containing weight window and mesh data.
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
    ww_obj = simulation["weight_windows"]
    indices = get_ww_indices(particle_container, ww_obj, simulation, data)
    # grab the actual ww parameters
    lower = ww_get.lower_weights(*indices, ww_obj, data)
    target = ww_get.target_weights(*indices, ww_obj, data)
    upper = ww_get.upper_weights(*indices, ww_obj, data)
    return lower, target, upper


@njit
def get_ww_indices(particle_container, ww_obj, simulation, data):
    """
    Get flattened weight window index from particle information

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    weight_window_object : object
        The weight window object containing index information
    simulation : object
        Simulation state containing weight window and mesh data.
    data : object
        Simulation data for array access.

    Returns
    -------
    indices: Tuple[int]
        the flattened index in the weight window array
    """
    particle = particle_container[0]

    # get energy index
    energy_bounds = ww_get.energy_bounds_all(ww_obj, data)
    if simulation["settings"]["neutron_multigroup_mode"]:
        energy = particle["g"]
    else:
        energy = particle["E"]
    ie = util.find_bin(energy, energy_bounds)

    # get spatial index
    mesh = simulation["meshes"][ww_obj["mesh_ID"]]
    idx, idy, idz = get_mesh_indices(particle_container, mesh, simulation, data)

    return (ie, idx, idy, idz)


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
                particle_bank_module.bank_active_particle(residual_copy, program)


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
