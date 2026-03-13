import math

from numba import njit

###

import mcdc.transport.rng as rng
import mcdc.transport.physics.neutron as neutron

from mcdc.constant import *

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.particle_speed(particle_container, simulation, data)
    return -1.0


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.macro_xs(reaction_type, particle_container, simulation, data)
    return -1.0


@njit
def neutron_production_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.neutron_production_xs(
            reaction_type, particle_container, simulation, data
        )
    return -1.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision_distance(particle_container, simulation, data):
    # Get total cross-section
    SigmaT = macro_xs(NEUTRON_REACTION_TOTAL, particle_container, simulation, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng.lcg(particle_container)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def collision(particle_container, program, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        neutron.collision(particle_container, program, data)
