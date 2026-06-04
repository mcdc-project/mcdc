from numba import njit

####

import mcdc.transport.physics.proton.multigroup as multigroup
import mcdc.transport.physics.proton.native as native
import mcdc.transport.util as util

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    return native.macro_xs(reaction_type, particle_container, simulation, data)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)
    native.collision(particle_container, collision_data_container, program, data)


# ======================================================================================
# Continuous Slowing Down Approximation
# ======================================================================================


@njit
def csda_edep(particle_container, collision_data_container, distance, simulation, data):
    native.csda_edep(
        particle_container, collision_data_container, distance, simulation, data
    )
