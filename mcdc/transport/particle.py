from numba import njit

####

import mcdc.transport.physics as physics
import mcdc.transport.rng as rng


@njit
def move(particle_container, distance, mcdc, data):
    particle = particle_container[0]
    ut = 1.0 / physics.particle_speed(particle_container, mcdc, data)

    particle["x"] += particle["ux"] * distance
    particle["y"] += particle["uy"] * distance
    particle["z"] += particle["uz"] * distance
    particle["t"] += ut * distance


@njit
def copy(target_particle_container, source_particle_container):
    target_particle = target_particle_container[0]
    source_particle = source_particle_container[0]

    target_particle["x"] = source_particle["x"]
    target_particle["y"] = source_particle["y"]
    target_particle["z"] = source_particle["z"]
    target_particle["t"] = source_particle["t"]
    target_particle["ux"] = source_particle["ux"]
    target_particle["uy"] = source_particle["uy"]
    target_particle["uz"] = source_particle["uz"]
    target_particle["g"] = source_particle["g"]
    target_particle["E"] = source_particle["E"]
    target_particle["w"] = source_particle["w"]
    target_particle["particle_type"] = source_particle["particle_type"]
    target_particle["rng_seed"] = source_particle["rng_seed"]


@njit
def copy_as_child(child_particle_container, parent_particle_container):
    parent_particle = parent_particle_container[0]
    child_particle = child_particle_container[0]

    copy(child_particle_container, parent_particle_container)

    # Set child RNG seed based of the parent
    parent_seed = parent_particle["rng_seed"]
    child_particle["rng_seed"] = rng.split_seed(parent_seed, rng.SEED_SPLIT_PARTICLE)

    # Evolve parent seed
    rng.lcg(parent_particle_container)
