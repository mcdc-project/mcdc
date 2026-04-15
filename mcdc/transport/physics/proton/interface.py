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
    if simulation["settings"]["proton_multigroup_mode"]:
        return multigroup.particle_speed(particle_container, simulation, data)
    else:
        return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    if simulation["settings"]["proton_multigroup_mode"]:
        return multigroup.macro_xs(reaction_type, particle_container, simulation, data)
    else:
        return native.macro_xs(reaction_type, particle_container, simulation, data)


# @njit
# def proton_production_xs(reaction_type, particle_container, simulation, data):
#     if simulation["settings"]["proton_multigroup_mode"]:
#         return multigroup.proton_production_xs(
#             reaction_type, particle_container, simulation, data
#         )
#     else:
#         return native.proton_production_xs(
#             reaction_type, particle_container, simulation, data
#         )


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)

    if simulation["settings"]["proton_multigroup_mode"]:
        multigroup.collision(
            particle_container, collision_data_container, program, data
        )
    else:
        native.collision(particle_container, collision_data_container, program, data)
