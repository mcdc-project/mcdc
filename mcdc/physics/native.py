import math
from numba import njit

####

import mcdc.adapt as adapt
import mcdc.kernel as kernel
import mcdc.mcdc_get as mcdc_get
import mcdc.physics.neutron_reaction as neutron_reaction

from mcdc.constant import (
    REACTION_TOTAL,
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    SQRT_E_TO_SPEED,
)
from mcdc.physics.util import evaluate_xs_energy_grid
from mcdc.util import linear_interpolation


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    particle = particle_container[0]
    return math.sqrt(particle["E"]) * SQRT_E_TO_SPEED


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    E = particle["E"]

    # Sum over all nuclides
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide = mcdc_get.nuclide.from_material(i, material, mcdc, data)
        atomic_density = mcdc_get.material.atomic_densities(i, material, data)
        xs = micro_xs(E, reaction_type, nuclide, mcdc, data)
        total += atomic_density * xs
    return total


@njit
def micro_xs(E, reaction_type, nuclide, mcdc, data):
    # Total reaction
    if reaction_type == REACTION_TOTAL:
        idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)
        xs0 = mcdc_get.nuclide.total_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.total_xs(idx + 1, nuclide, data)
        return linear_interpolation(E, E0, E1, xs0, xs1)

    # Search if the reaction exists
    for i in range(nuclide["N_reaction"]):
        the_type = int(mcdc_get.nuclide.reaction_type(i, nuclide, data))

        if the_type == reaction_type:
            # Reaction exists!
            reaction_idx = int(mcdc_get.nuclide.reaction_index(i, nuclide, data))
            idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)

            if reaction_type == REACTION_NEUTRON_CAPTURE:
                reaction = mcdc["neutron_capture_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
                reaction = mcdc["neutron_elastic_scattering_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            elif reaction_type == REACTION_NEUTRON_FISSION:
                reaction = mcdc["fission_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

    return 0.0


@njit
def neutron_production_xs_(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    E = particle["E"]

    # Sum over all nuclides
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide = mcdc_get.nuclide.from_material(i, material, mcdc, data)
        atomic_density = mcdc_get.material.atomic_densities(i, material, data)
        nu_xs = reaction_production_xs(E, reaction_type, nuclide, mcdc, data)
        total += atomic_density * nu_xs
    return total


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)
    material = mcdc["materials"][particle["material_ID"]]

    # TODO implicit capture

    # ==================================================================================
    # Sample colliding nuclide
    # ==================================================================================

    SigmaT = macro_xs(REACTION_TOTAL, material, particle_container, mcdc, data)
    xi = kernel.rng(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide = mcdc_get.nuclide.from_material(i, material, mcdc, data)
        atomic_density = mcdc_get.material.atomic_densities(i, material, data)
        sigmaT = micro_xs(particle["E"], REACTION_TOTAL, nuclide, mcdc, data)
        SigmaT_nuclide = atomic_density * sigmaT
        total += SigmaT_nuclide
        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    xi = kernel.rng(particle_container) * sigmaT
    total = 0.0
    for i in range(nuclide["N_reaction"]):
        reaction_type = int(mcdc_get.nuclide.reaction_type(i, nuclide, data))
        reaction_xs = micro_xs(particle["E"], reaction_type, nuclide, mcdc, data)
        total += reaction_xs
        if total < xi:
            continue

        # Reaction is sampled
        reaction_idx = int(mcdc_get.nuclide.reaction_index(i, nuclide, data))
        if reaction_type == REACTION_NEUTRON_CAPTURE:
            particle["alive"] = False
        elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
            reaction = mcdc["neutron_elastic_scattering_reactions"][reaction_idx]
            neutron_reaction.elastic_scattering(particle_container, nuclide, reaction, prog, data)
        elif reaction_type == REACTION_NEUTRON_FISSION:
            reaction = mcdc["neutron_fission_reactions"][reaction_idx]
            neutron_reaction.fission(particle_container, nuclide, reaction, prog, data)
