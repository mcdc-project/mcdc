import numpy as np
import math

from numba import njit

####
import mcdc.mcdc_get.weight_windows as ww_get
import mcdc.numba_types as type_
from mcdc.transport.mesh import get_indices as get_mesh_indices
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

# ======================================================================================
# Weight Roulette
# ======================================================================================


@njit
def weight_roulette(particle_container, simulation):
    particle = particle_container[0]
    if particle["w"] < simulation["weight_roulette"]["weight_threshold"]:
        w_target = simulation["weight_roulette"]["weight_target"]
        survival_probability = particle["w"] / w_target
        if rng.lcg(particle_container) < survival_probability:
            particle["w"] = w_target
        else:
            particle["alive"] = False


# ======================================================================================
# Weight Windows
# ======================================================================================


@njit
def weight_windows(particle_container, mcdc, data):
    [lower, target, upper] = query_weight_window(particle_container, mcdc, data)
    split_from_weight_window(particle_container, upper, mcdc)
    roulette_from_weight_window(particle_container, lower, target)


@njit
def query_weight_window(particle_container, mcdc, data):
    ww_obj = mcdc["weight_windows"]
    mesh = mcdc["meshes"][ww_obj["mesh_ID"]]
    idx, idy, idz = get_mesh_indices(particle_container, mesh, mcdc, data)
    index = ((idx * ww_obj["Ny"]) + idy) * ww_obj["Nz"] + idz
    lower = ww_get.lower_weights(index, ww_obj, data)
    target = ww_get.target_weights(index, ww_obj, data)
    upper = ww_get.upper_weights(index, ww_obj, data)
    return lower, target, upper


@njit
def split_from_weight_window(particle_container, threshold_weight, mcdc):
    particle = particle_container[0]
    weight = particle["w"]
    if weight > threshold_weight:
        num_split = math.ceil(weight / threshold_weight)
        particle["w"] = weight / num_split
        for _ in range(num_split - 1):
            particle_bank_module.bank_active_particle(particle_container, mcdc)


@njit
def roulette_from_weight_window(particle_container, w_threshold, w_target):
    particle = particle_container[0]
    if particle["w"] < w_threshold:
        survival_probability = particle["w"] / w_target
        if rng.lcg(particle_container) < survival_probability:
            particle["w"] = w_target
        else:
            particle["alive"] = False


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
