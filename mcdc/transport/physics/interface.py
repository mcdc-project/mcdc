import math

from numba import njit

###

import mcdc.transport.physics.electron as electron
import mcdc.transport.physics.neutron as neutron
import mcdc.transport.physics.photon as photon
import mcdc.transport.physics.proton as proton
import mcdc.transport.rng as rng

from mcdc.constant import *

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def is_electron_like(particle_type):
    """Positrons are transported with the electron physics."""
    return particle_type == PARTICLE_ELECTRON or particle_type == PARTICLE_POSITRON


@njit
def particle_speed(particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.particle_speed(particle_container, simulation, data)
    elif is_electron_like(particle["particle_type"]):
        return electron.particle_speed(particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        return proton.particle_speed(particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PHOTON:
        return photon.particle_speed(particle_container, simulation, data)
    return -1.0


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.macro_xs(reaction_type, particle_container, simulation, data)
    elif is_electron_like(particle["particle_type"]):
        return electron.macro_xs(reaction_type, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        return proton.macro_xs(reaction_type, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PHOTON:
        return photon.macro_xs(reaction_type, particle_container, simulation, data)
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
    particle = particle_container[0]

    # Get total cross-section
    SigmaT = 0.0
    if particle["particle_type"] == PARTICLE_NEUTRON:
        SigmaT = macro_xs(NEUTRON_REACTION_TOTAL, particle_container, simulation, data)
    elif is_electron_like(particle["particle_type"]):
        SigmaT = macro_xs(ELECTRON_REACTION_TOTAL, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        SigmaT = macro_xs(PROTON_REACTION_TOTAL, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PHOTON:
        SigmaT = macro_xs(PHOTON_REACTION_TOTAL, particle_container, simulation, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng.lcg(particle_container)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def collision(particle_container, interaction_data_container, program, data):
    particle = particle_container[0]

    if particle["particle_type"] == PARTICLE_NEUTRON:
        neutron.collision(particle_container, interaction_data_container, program, data)
    elif is_electron_like(particle["particle_type"]):
        electron.collision(
            particle_container, interaction_data_container, program, data
        )
    elif particle["particle_type"] == PARTICLE_PROTON:
        proton.collision(particle_container, interaction_data_container, program, data)
    elif particle["particle_type"] == PARTICLE_PHOTON:
        photon.collision(particle_container, interaction_data_container, program, data)
