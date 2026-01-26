from numba import njit

####

import mcdc.transport.physics.neutron.native as native

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, mcdc, data):
    return native.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, mcdc, data):
    return native.macro_xs(reaction_type, particle_container, mcdc, data)


@njit
def neutron_production_xs(reaction_type, particle_container, mcdc, data):
    return native.neutron_production_xs(reaction_type, particle_container, mcdc, data)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, prog, data):
    native.collision(particle_container, prog, data)
