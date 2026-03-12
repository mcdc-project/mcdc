from numba import njit

####

import mcdc.transport.physics.neutron.multigroup as multigroup
import mcdc.transport.physics.neutron.native as native

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    if simulation["settings"]["neutron_multigroup_mode"]:
        return multigroup.particle_speed(particle_container)
    else:
        return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    if simulation["settings"]["neutron_multigroup_mode"]:
        return multigroup.macro_xs(reaction_type, particle_container, simulation, data)
    else:
        return native.macro_xs(reaction_type, particle_container, simulation, data)


@njit
def neutron_production_xs(reaction_type, particle_container, simulation, data):
    if simulation["settings"]["neutron_multigroup_mode"]:
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


@njit
def collision(particle_container, simulation, data):
    if simulation["settings"]["neutron_multigroup_mode"]:
        multigroup.collision(particle_container, simulation, data)
    else:
        native.collision(particle_container, simulation, data)
