from numba import njit

####

import mcdc.transport.physics.neutron.multigroup as multigroup
import mcdc.transport.physics.neutron.native as native
import mcdc.transport.util as util
import mcdc.code_factory.gpu.substitution as sub

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    if multigroup.applicable(particle_container, simulation, data):
        return multigroup.particle_speed(particle_container, simulation, data)
    else:
        return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    if multigroup.applicable(particle_container, simulation, data):
        return multigroup.macro_xs(reaction_type, particle_container, simulation, data)
    else:
        return native.macro_xs(reaction_type, particle_container, simulation, data)


@njit
def neutron_production_xs(reaction_type, particle_container, simulation, data):
    if multigroup.applicable(particle_container, simulation, data):
        return multigroup.neutron_production_xs(
            reaction_type, particle_container, simulation, data
        )
    else:
        return native.neutron_production_xs(
            reaction_type, particle_container, simulation, data
        )


# ======================================================================================
# Collision
# ======================================================================================


@sub.target(tag="async")
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)

    if multigroup.applicable(particle_container, simulation, data):
        multigroup.collision(
            particle_container, collision_data_container, program, data
        )
    else:
        native.collision(particle_container, collision_data_container, program, data)
