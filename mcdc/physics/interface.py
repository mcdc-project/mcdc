import math

from numba import njit

###

import mcdc.kernel as kernel
import mcdc.physics.native as native
import mcdc.mcdc_get as mcdc_get

from mcdc.constant import *
from mcdc.util import binary_search, linear_interpolation


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, material, data):
    return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    return native.macro_xs(reaction_type, material, particle_container, mcdc, data)


@njit
def neutron_production_xs(reaction_type, material, particle_container, mcdc, data):
    return native.neutron_production_xs(
        reaction_type, material, particle_container, mcdc, data
    )


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
    native.collision(particle_container, prog, data)
