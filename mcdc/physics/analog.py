import math
from numba import njit

####

import mcdc.mcdc_get as mcdc_get

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
        sigmaT = reaction_xs(particle["E"], REACTION_TOTAL, nuclide, mcdc, data)
        SigmaT_nuclide = atomic_density * sigmaT
        total += SigmaT_nuclide
        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    xi = kernel.rng(particle_container) * SigmaT_nuclide
    total = 0.0
    for i in range(nuclide["N_reaction"]):
        reaction_type = int(mcdc_get.nuclide.reaction_type(i, nuclide, data))
        if reaction_type == REACTION_NEUTRON_CAPTURE:
            particle["alive"] = False
        elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
            neutron_reaction.elastic_scattering(particle_container, prog, data)
        elif reaction_type == REACTION_NEUTRON_FISSION:
            neutron_reaction.fission(particle_container, prog, data)


# ======================================================================================
# Reactions
# ======================================================================================


@njit
def scattering(P_arr, prog, data):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)
    # Kill the current particle
    P["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Get number of secondaries
    material = mcdc["materials"][P["material_ID"]]
    g = P["g"]
    if mcdc["settings"]["multigroup_mode"]:
        nu_s = mcdc_get.material.mgxs_nu_s(g, material, data)
        N = int(math.floor(weight_eff * nu_s + rng(P_arr)))
    else:
        N = 1

    P_new_arr = adapt.local_array(1, type_.particle_record)
    P_new = P_new_arr[0]

    for n in range(N):
        # Create new particle
        split_as_record(P_new_arr, P_arr)

        # Set weight
        P_new["w"] = weight_new

        # Sample scattering phase space
        sample_phasespace_scattering(P_arr, material, P_new_arr, mcdc, data)

        # Bank, but keep it if it is the last particle
        if n == N - 1:
            P["alive"] = True
            P["ux"] = P_new["ux"]
            P["uy"] = P_new["uy"]
            P["uz"] = P_new["uz"]
            P["g"] = P_new["g"]
            P["E"] = P_new["E"]
            P["w"] = P_new["w"]
        else:
            adapt.add_active(P_new_arr, prog)


@njit
def fission(P_arr, prog, data):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Kill the current particle
    P["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Sample nuclide if CE
    material = mcdc["materials"][P["material_ID"]]

    # Get number of secondaries
    if mcdc["settings"]["multigroup_mode"]:
        g = P["g"]
        nu = mcdc_get.material.mgxs_nu_f(g, material, data)
    else:
        nuclide = sample_nuclide(material, P_arr, XS_FISSION, mcdc)
        E = P["E"]
        nu = get_nu(NU_FISSION, nuclide, E)
    N = int(math.floor(weight_eff * nu / mcdc["k_eff"] + rng(P_arr)))

    P_new_arr = adapt.local_array(1, type_.particle_record)
    P_new = P_new_arr[0]

    for n in range(N):
        # Create new particle
        split_as_record(P_new_arr, P_arr)

        # Set weight
        P_new["w"] = weight_new

        # Sample fission neutron phase space
        if mcdc["settings"]["multigroup_mode"]:
            sample_phasespace_fission(P_arr, material, P_new_arr, mcdc, data)
        else:
            sample_phasespace_fission_nuclide(P_arr, nuclide, P_new_arr, mcdc)

        # Eigenvalue mode: bank right away
        if mcdc["settings"]["eigenvalue_mode"]:
            adapt.add_census(P_new_arr, prog)
            continue
        # Below is only relevant for fixed-source problem

        # Skip if it's beyond time boundary
        if P_new["t"] > mcdc["settings"]["time_boundary"]:
            continue

        # Check if it is beyond current or next census times
        hit_census = False
        hit_next_census = False
        idx_census = mcdc["idx_census"]
        if idx_census < mcdc["settings"]["N_census"] - 1:
            settings = mcdc["settings"]
            if P["t"] > mcdc_get.settings.census_time(idx_census + 1, settings, data):
                hit_census = True
                hit_next_census = True
            elif P_new["t"] > mcdc_get.settings.census_time(idx_census, settings, data):
                hit_census = True

        if not hit_census:
            # Keep it if it is the last particle
            if n == N - 1:
                P["alive"] = True
                P["ux"] = P_new["ux"]
                P["uy"] = P_new["uy"]
                P["uz"] = P_new["uz"]
                P["t"] = P_new["t"]
                P["g"] = P_new["g"]
                P["E"] = P_new["E"]
                P["w"] = P_new["w"]
            else:
                adapt.add_active(P_new_arr, prog)
        elif not hit_next_census:
            # Particle will participate after the current census
            adapt.add_census(P_new_arr, prog)
        else:
            # Particle will participate in the future
            adapt.add_future(P_new_arr, prog)
