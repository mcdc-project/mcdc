import math

from numba import njit

###

import mcdc.kernel as kernel
import mcdc.mcdc_get as mcdc_get

from mcdc.constant import *
from mcdc.src.algorithm import binary_search, linear_interpolation


# ======================================================================================
# Particle speed
# ======================================================================================


@njit
def particle_speed(particle_container, material, data):
    return particle_speed_(particle_container)


@njit
def particle_speed_(particle_container):
    particle = particle_container[0]
    return math.sqrt(particle["E"]) * SQRT_E_TO_SPEED


@njit
def particle_speed_mg(particle_container, material, data):
    particle = particle_container[0]
    return mcdc_get.material.mgxs_speed(particle["g"], material, data)


# ======================================================================================
# Material macroscopic cross-section
# ======================================================================================


@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    E = particle['E']

    # Sum over all nuclides
    total = 0.0
    for i in range(material['N_nuclide']):
        nuclide = mcdc_get.nuclide.from_material(i, material, mcdc, data)
        atomic_density = mcdc_get.material.atomic_densities(i, material, data)
        xs = reaction_xs(E, reaction_type, nuclide, mcdc, data)
        total += atomic_density * xs
    return total


@njit
def macro_xs_mg(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    g = particle["g"]
    if reaction_type == REACTION_TOTAL:
        return mcdc_get.material.mgxs_total(g, material, data)
    elif reaction_type == REACTION_CAPTURE:
        return mcdc_get.material.mgxs_capture(g, material, data)
    elif reaction_type == REACTION_ELASTIC_SCATTERING:
        return mcdc_get.material.mgxs_scatter(g, material, data)
    elif reaction_type == REACTION_FISSION:
        return mcdc_get.material.mgxs_fission(g, material, data)


# ======================================================================================
# Material production cross-section
# ======================================================================================


@njit
def production_xs(reaction_type, material, particle_container, mcdc, data):
    return production_xs_(reaction_type, material, particle_container, mcdc, data)


@njit
def production_xs_(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    E = particle['E']

    # Sum over all nuclides
    total = 0.0
    for i in range(material['N_nuclide']):
        nuclide = mcdc_get.nuclide.from_material(i, material, mcdc, data)
        atomic_density = mcdc_get.material.atomic_densities(i, material, data)
        nu_xs = reaction_production_xs(E, reaction_type, nuclide, mcdc, data)
        total += atomic_density * nu_xs
    return total


@njit
def production_xs_mg(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    g = particle["g"]
    if reaction_type == REACTION_TOTAL:
        total = 0.0
        total += production_xs_mg(REACTION_ELASTIC_SCATTERING, material, particle_container, mcdc, data)
        total += production_xs_mg(REACTION_FISSION, material, particle_container, mcdc, data)
        return total
    elif reaction_type == REACTION_ELASTIC_SCATTERING:
        nu = mcdc_get.material.mgxs_nu_s(g, material, data)
        xs = mcdc_get.material.mgxs_scatter(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_FISSION:
        nu = mcdc_get.material.mgxs_nu_f(g, material, data)
        xs = mcdc_get.material.mgxs_fission(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_FISSION_PROMPT:
        nu = mcdc_get.material.mgxs_nu_p(g, material, data)
        xs = mcdc_get.material.mgxs_fission(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_FISSION_DELAYED:
        nu = mcdc_get.material.mgxs_nu_d_total(g, material, data)
        xs = mcdc_get.material.mgxs_fission(g, material, data)
        return nu * xs


# ======================================================================================
# Reaction common attributes
# ======================================================================================


@njit
def reaction_xs(E, reaction_type, nuclide, mcdc, data):
    # Total reaction
    if reaction_type == REACTION_TOTAL:
        idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)
        xs0 = mcdc_get.nuclide.total_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.total_xs(idx + 1, nuclide, data)
        return linear_interpolation(E, E0, E1, xs0, xs1)

    # Search if the reaction exists
    for i in range(nuclide['N_reaction']):
        the_type = int(mcdc_get.nuclide.reaction_type(i, nuclide, data))

        if the_type == reaction_type:
            # Reaction exists!
            reaction_idx = int(mcdc_get.nuclide.reaction_index(i, nuclide, data))
            idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)
            
            if reaction_type == REACTION_CAPTURE:
                reaction = mcdc['capture_reactions'][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)
                
            elif reaction_type == REACTION_ELASTIC_SCATTERING:
                reaction = mcdc['elastic_scattering_reactions'][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            elif reaction_type == REACTION_FISSION:
                reaction = mcdc['fission_reactions'][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)
            
    return 0.0


@njit
def reaction_production_xs(E, reaction_type, nuclide, mcdc, data):
    # Total reaction
    if reaction_type == REACTION_TOTAL:
        # TODO
        return 0.0

    # Search if the reaction exists
    for i in range(nuclide['N_reaction']):
        the_type = int(mcdc_get.nuclide.reaction_type(i, nuclide, data))

        if the_type == reaction_type:
            # Reaction exists!
            reaction_idx = (mcdc_get.nuclide.reaction_index(i, nuclide, data))
            idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)

            if reaction_type == REACTION_CAPTURE:
                return 0.0
                
            elif reaction_type == REACTION_ELASTIC_SCATTERING:
                reaction = mcdc['elastic_scattering_reactions'][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            elif reaction_type == REACTION_FISSION:
                reaction = mcdc['fission_reactions'][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                xs = linear_interpolation(E, E0, E1, xs0, xs1)
                # TODO: nu
                return 0.0
            
    return 0.0



# ======================================================================================
# Collision distance
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


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, mcdc, data):
    P = particle_container[0]

    # Get the reaction cross-sections
    material = mcdc['materials'][P["material_ID"]]
    SigmaT = macro_xs(REACTION_TOTAL, material, particle_container, mcdc, data)
    SigmaS = macro_xs(
        REACTION_ELASTIC_SCATTERING, material, particle_container, mcdc, data
    )
    SigmaC = macro_xs(REACTION_CAPTURE, material, particle_container, mcdc, data)
    SigmaF = macro_xs(REACTION_FISSION, material, particle_container, mcdc, data)

    # Implicit capture
    if mcdc["technique"]["implicit_capture"]:
        P["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    # Sample collision type
    xi = kernel.rng(particle_container) * SigmaT
    tot = SigmaS
    if tot > xi:
        P["event"] += EVENT_SCATTERING
    else:
        tot += SigmaF
        if tot > xi:
            P["event"] += EVENT_FISSION
        else:
            P["event"] += EVENT_CAPTURE


# ======================================================================================
# Sample nuclide speed
# ======================================================================================


@njit
def sample_nucleus_speed(A, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc['materials'][particle["material_ID"]]

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
            x = math.sqrt(-math.log(kernel.rng(particle_container) * kernel.rng(particle_container)))
        else:
            cos_val = math.cos(PI_HALF * kernel.rng(particle_container))
            x = math.sqrt(
                -math.log(kernel.rng(particle_container)) - math.log(kernel.rng(particle_container)) * cos_val * cos_val
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
    ux, uy, uz = kernel.scatter_direction(particle["ux"], particle["uy"], particle["uz"], mu_tilda, azi)
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz



# ======================================================================================
# Helper functions
# ======================================================================================


@njit
def evaluate_xs_energy_grid(E, nuclide, data):
    energy_grid = mcdc_get.nuclide.xs_energy_grid_all(nuclide, data)
    idx = binary_search(E, energy_grid)
    E0 = energy_grid[idx]
    E1 = energy_grid[idx + 1]
    return idx, E0, E1
