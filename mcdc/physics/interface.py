import math

from numba import njit

###

import mcdc.kernel as kernel
import mcdc.physics.native as native
import mcdc.mcdc_get as mcdc_get

from mcdc.constant import *
from mcdc.util import binary_search, linear_interpolation


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


@njit
def neutron_production_xs(reaction_type, material, particle_container, mcdc, data):
    return native.neutron_production_xs(
        reaction_type, material, particle_container, mcdc, data
    )


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision_distance(particle_container, material, mcdc, data):
    # Get total cross-section
    SigmaT = macro_xs(REACTION_TOTAL, material, particle_container, mcdc, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = kernel.rng(particle_container)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def collision(particle_container, prog, data):
    native.collision(particle_container, prog, data)


# ======================================================================================
# Reaction common attributes
# ======================================================================================


@njit
def reaction_production_xs(E, reaction_type, nuclide, mcdc, data):
    # Total reaction
    if reaction_type == REACTION_TOTAL:
        # TODO
        return 0.0

    # Search if the reaction exists
    for i in range(nuclide["N_reaction"]):
        the_type = int(mcdc_get.nuclide.reaction_type(i, nuclide, data))

        if the_type == reaction_type:
            # Reaction exists!
            reaction_idx = mcdc_get.nuclide.reaction_index(i, nuclide, data)
            idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)

            if reaction_type == REACTION_NEUTRON_CAPTURE:
                return 0.0

            elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
                reaction = mcdc["neutron_elastic_scattering_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            elif reaction_type == REACTION_NEUTRON_FISSION:
                reaction = mcdc["fission_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                xs = linear_interpolation(E, E0, E1, xs0, xs1)
                # TODO: nu
                return 0.0

    return 0.0


# ======================================================================================
# helper functions
# ======================================================================================


@njit
def evaluate_xs_energy_grid(e, nuclide, data):
    energy_grid = mcdc_get.nuclide.xs_energy_grid_all(nuclide, data)
    idx = binary_search(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1
