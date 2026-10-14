from numba import njit

####

import mcdc.transport.physics.photon.constant_xs as constant_xs
import mcdc.transport.physics.photon.native as native
import mcdc.transport.util as util

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    if constant_xs.applicable(particle_container, simulation, data):
        return constant_xs.particle_speed(particle_container, simulation, data)
    else:
        return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    if constant_xs.applicable(particle_container, simulation, data):
        return constant_xs.macro_xs(reaction_type, particle_container, simulation, data)
    else:
        return native.macro_xs(reaction_type, particle_container, simulation, data)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, interaction_data_container, program, data):
    simulation = util.access_simulation(program)

    if constant_xs.applicable(particle_container, simulation, data):
        constant_xs.collision(
            particle_container, interaction_data_container, program, data
        )
    else:
        native.collision(particle_container, interaction_data_container, program, data)
