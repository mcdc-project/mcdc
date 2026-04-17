import numpy as np
import math

from numba import njit

####

import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
from mcdc.transport.physics import interface as physics
import mcdc.transport.util as util
import mcdc.mcdc_get as mcdc_get

# ======================================================================================
# Forced Collisions
# ======================================================================================


@njit
def forced_collisions(particle_container, surface_distance, program, data):
    simulation = util.access_simulation(program)
    fc_object = simulation["forced_collisions"]
    SigmaT = physics.total_xs(particle_container, simulation, data)

    # create collided and transmitted particles
    collided_container = particle_container
    collided = collided_container[0]

    transmitted_container = util.local_array(1, type_.particle_data)
    transmitted = transmitted_container[0]
    particle_module.copy_as_child(transmitted_container, collided_container)
    
    # update transmitted particle
    weight_multiplier = math.exp(-surface_distance * SigmaT)
    transmitted["w"] *= weight_multiplier
    particle_module.move(transmitted_container, surface_distance, simulation, data)
    particle_bank_module.bank_active_particle(transmitted_container, program)

    # update collided particle
    collided["w"] *= (1 - weight_multiplier)
    # return distance to forced collision, let simulation handle the rest (tallies)
    return physics.forced_collision_distance(collided_container, surface_distance, simulation, data) 


@njit
def in_forced_collision_cell(particle_container, simulation, data):
    fc_object = simulation["forced_collisions"]
    # not active, dont need to query cells
    if not fc_object["active"]:
        return False
    # active, need to check if in active cell
    cell_ids = mcdc_get.forced_collisions.cell_IDs_all(fc_object, data)
    if particle_container[0]["cell_ID"] not in cell_ids:
        return True
    # not in active cell
    return False


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
