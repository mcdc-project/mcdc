import math

from numba import njit

####

import mcdc.kernel as kernel
import mcdc.physics.electron.native as native

from mcdc.constant import REACTION_TOTAL, INF


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


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, prog, data):
    native.collision(particle_container, prog, data)
