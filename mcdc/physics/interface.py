import math

from numba import njit

###

import mcdc.kernel as kernel
import mcdc.mcdc_get as mcdc_get

import mcdc.physics.neutron.interface as neutron

from mcdc.constant import *
from mcdc.util import binary_search, linear_interpolation


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, material, data):
    particle = particle_container[0]
    if particle['type'] == PARTICLE_NEUTRON:
        return neutron.particle_speed(particle_container, material, data)
    return -1.0


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    if particle['type'] == PARTICLE_NEUTRON:
        return neutron.macro_xs(reaction_type, material, particle_container, mcdc, data)
    return -1.0


@njit
def neutron_production_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    if particle['type'] == PARTICLE_NEUTRON:
        return neutron.neutron_production_xs(
            reaction_type, material, particle_container, mcdc, data
        )
    return -1.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision_distance(particle_container, material, mcdc, data):
    # Get total cross-section
    SigmaT = macro_xs(REACTION_TOTAL, material, particle_container, mcdc, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = kernel.rng(particle_container)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def collision(particle_container, prog, data):
    particle = particle_container[0]
    if particle['type'] == PARTICLE_NEUTRON:
        neutron.collision(particle_container, prog, data)
