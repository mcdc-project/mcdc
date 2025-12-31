from numba import njit

####

import mcdc.transport.physics as physics
import mcdc.transport.rng as rng

import mcdc.code_factory.adapt as adapt



@adapt.toggle("sensitivity")
def _copy_resp_cum(target_particle_container, source_particle_container):
    """Copy per-particle response accumulator (sensitivity mode only)."""
    target = target_particle_container[0]
    source = source_particle_container[0]
    target["resp_cum"][:] = source["resp_cum"][:]

@adapt.toggle("sensitivity")
def _reset_resp_cum(particle_container):
    """Reset per-particle response accumulator for a new child history."""
    particle_container[0]["resp_cum"][:] = 0.0

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

    _copy_resp_cum(target_particle_container, source_particle_container)


@njit
def copy_as_child(child_particle_container, parent_particle_container):
    parent_particle = parent_particle_container[0]
    child_particle = child_particle_container[0]

    copy(child_particle_container, parent_particle_container)

    _reset_resp_cum(child_particle_container)

    # Set child RNG seed based of the parent
    parent_seed = parent_particle["rng_seed"]
    child_particle["rng_seed"] = rng.split_seed(parent_seed, rng.SEED_SPLIT_PARTICLE)

    # Evolve parent seed
    rng.lcg(parent_particle_container)
