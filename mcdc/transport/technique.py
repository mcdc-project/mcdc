import numpy as np
import math

from numba import njit

####

import mcdc.code_factory.adapt as adapt
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng

# ======================================================================================
# Weight Roulette
# ======================================================================================


@njit
def weight_roulette(particle_container, mcdc):
    particle = particle_container[0]
    if particle["w"] < mcdc["weight_roulette"]["weight_threshold"]:
        w_target = mcdc["weight_roulette"]["weight_target"]
        survival_probability = particle["w"] / w_target
        if rng.lcg(particle_container) < survival_probability:
            particle["w"] = w_target
        else:
            particle["alive"] = False


# ======================================================================================
# Population Control
# ======================================================================================


@njit
def population_control(mcdc):
    """Uniform Splitting-Roulette technique"""

    bank_census = mcdc["bank_census"]
    M = mcdc["settings"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank
    idx_start, N_local, N = particle_bank_module.bank_scanning(bank_census, mcdc)
    idx_end = idx_start + N_local

    # Abort if census bank is empty
    if N == 0:
        return

    # Weight scaling
    ws = float(N) / float(M)

    # Splitting Number
    sn = 1.0 / ws

    P_rec_arr = np.zeros(1, type_.particle_data)
    P_rec = P_rec_arr[0]

    # Perform split-roulette to all particles in local bank
    particle_bank_module.set_bank_size(bank_source, 0)
    for idx in range(N_local):
        # Weight of the surviving particles
        w = bank_census["particles"][idx]["w"]
        w_survive = w * ws

        # Determine number of guaranteed splits
        N_split = math.floor(sn)

        # Survive the russian roulette?
        xi = rng.lcg(bank_census["particles"][idx : idx + 1])
        if xi < sn - N_split:
            N_split += 1

        # Split the particle
        for i in range(N_split):
            particle_module.copy_as_child(
                P_rec_arr, bank_census["particles"][idx : idx + 1]
            )
            # Set weight
            P_rec["w"] = w_survive
            particle_bank_module.add_source(P_rec_arr, mcdc)
