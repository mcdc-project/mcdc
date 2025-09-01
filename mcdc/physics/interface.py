import math

from numba import njit

###

import mcdc.kernel as kernel
import mcdc.physics.analog as analog
import mcdc.mcdc_get as mcdc_get

from mcdc.constant import *
from mcdc.util import binary_search, linear_interpolation


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, material, data):
    return analog.particle_speed(particle_container)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    return analog.macro_xs(reaction_type, material, particle_container, mcdc, data)


@njit
def neutron_production_xs(reaction_type, material, particle_container, mcdc, data):
    return analog.neutron_production_xs(
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
    analog.collision(particle_container, prog, data)


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
# Sample nuclide speed
# ======================================================================================


@njit
def sample_nucleus_speed(A, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["materials"][particle["material_ID"]]

    # Particle speed
    P_speed = particle_speed(particle_container, material, data)

    # Maxwellian parameter
    beta = math.sqrt(2.0659834e-11 * A)
    # The constant above is
    #   (1.674927471e-27 kg) / (1.38064852e-19 cm^2 kg s^-2 K^-1) / (293.6 K)/2

    # Sample nuclide speed candidate V_tilda and
    #   nuclide-neutron polar cosine candidate mu_tilda via
    #   rejection sampling
    y = beta * P_speed
    while True:
        if kernel.rng(particle_container) < 2.0 / (2.0 + PI_SQRT * y):
            x = math.sqrt(
                -math.log(
                    kernel.rng(particle_container) * kernel.rng(particle_container)
                )
            )
        else:
            cos_val = math.cos(PI_HALF * kernel.rng(particle_container))
            x = math.sqrt(
                -math.log(kernel.rng(particle_container))
                - math.log(kernel.rng(particle_container)) * cos_val * cos_val
            )
        V_tilda = x / beta
        mu_tilda = 2.0 * kernel.rng(particle_container) - 1.0

        # Accept candidate V_tilda and mu_tilda?
        if kernel.rng(particle_container) > math.sqrt(
            P_speed * P_speed + V_tilda * V_tilda - 2.0 * P_speed * V_tilda * mu_tilda
        ) / (P_speed + V_tilda):
            break

    # Set nuclide velocity - LAB
    azi = 2.0 * PI * kernel.rng(particle_container)
    ux, uy, uz = kernel.scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu_tilda, azi
    )
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz


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
