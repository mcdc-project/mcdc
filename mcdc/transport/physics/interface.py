import math

from numba import njit

###

import mcdc.transport.rng as rng
import mcdc.transport.physics.electron as electron
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
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        return electron.particle_speed(particle_container, simulation, data)
    return -1.0


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def total_xs(particle_container, simulation, data):
    """
    Convenience helper for getting specifically the total cross section. 

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    simulation : object
        Simulation object.
    data : object
        Simulation data for array access.

    Returns
    -------
    float
        Total macroscopic cross section.
    """
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        module = neutron
        type_total = NEUTRON_REACTION_TOTAL
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        module = electron
        type_total = ELECTRON_REACTION_TOTAL
    else:
        return 0.0
    return module.macro_xs(type_total, particle_container, simulation, data)


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.macro_xs(reaction_type, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        return electron.macro_xs(reaction_type, particle_container, simulation, data)
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
    SigmaT = total_xs(particle_container, simulation, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng.lcg(particle_container)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def forced_collision_distance(particle_container, surface_distance, simulation, data):
    """
    Method for finding the distance for a forced collision particle to travel.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    surface_distance:
        The distance to the next surface along the particles direction.
    simulation : object
        Simulation object.
    data : object
        Simulation data for array access.
    
    Returns
    -------
    distance : float
        Distance for particle to travel.
    """
    # Get total cross-section
    SigmaT = total_xs(particle_container, simulation, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng.lcg(particle_container)
    
    distance = - math.log(1 - xi*(1-math.exp(-surface_distance * SigmaT))) / SigmaT
    return distance


@njit
def collision(particle_container, collision_data_container, program, data):
    particle = particle_container[0]

    if particle["particle_type"] == PARTICLE_NEUTRON:
        neutron.collision(particle_container, collision_data_container, program, data)
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        electron.collision(particle_container, collision_data_container, program, data)
