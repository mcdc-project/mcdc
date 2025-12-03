import numpy as np
import math

from numba import njit

####

import mcdc.code_factory.adapt as adapt
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng

from mcdc.constant import (
    PI,
    REACTION_TOTAL,
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    REACTION_NEUTRON_FISSION_DELAYED,
    REACTION_NEUTRON_FISSION_PROMPT,
)
from mcdc.transport.physics.util import scatter_direction
from mcdc.transport.distribution import sample_isotropic_direction


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["multigroup_materials"][particle["material_ID"]]
    return mcdc_get.multigroup_material.mgxs_speed(particle["g"], material, data)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["multigroup_materials"][particle["material_ID"]]
    g = particle["g"]

    if reaction_type == REACTION_TOTAL:
        return mcdc_get.multigroup_material.mgxs_total(g, material, data)
    elif reaction_type == REACTION_NEUTRON_CAPTURE:
        return mcdc_get.multigroup_material.mgxs_capture(g, material, data)
    elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
        return mcdc_get.multigroup_material.mgxs_scatter(g, material, data)
    elif reaction_type == REACTION_NEUTRON_FISSION:
        return mcdc_get.multigroup_material.mgxs_fission(g, material, data)
    return 0.0


@njit
def neutron_production_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["multigroup_materials"][particle["material_ID"]]

    g = particle["g"]
    if reaction_type == REACTION_TOTAL:
        total = 0.0
        total += neutron_production_xs(
            REACTION_NEUTRON_ELASTIC_SCATTERING,
            particle_container,
            mcdc,
            data,
        )
        total += neutron_production_xs(
            REACTION_NEUTRON_FISSION, particle_container, mcdc, data
        )
        return total
    elif reaction_type == REACTION_NEUTRON_CAPTURE:
        return 0.0
    elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
        nu = mcdc_get.multigroup_material.mgxs_nu_s(g, material, data)
        xs = mcdc_get.multigroup_material.mgxs_scatter(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_NEUTRON_FISSION:
        nu = mcdc_get.multigroup_material.mgxs_nu_f(g, material, data)
        xs = mcdc_get.multigroup_material.mgxs_fission(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_NEUTRON_FISSION_PROMPT:
        nu = mcdc_get.multigroup_material.mgxs_nu_p(g, material, data)
        xs = mcdc_get.multigroup_material.mgxs_fission(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_NEUTRON_FISSION_DELAYED:
        nu = mcdc_get.multigroup_material.mgxs_nu_d_total(g, material, data)
        xs = mcdc_get.multigroup_material.mgxs_fission(g, material, data)
        return nu * xs


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, prog, data):
    particle = particle_container[0]

    # Get the reaction cross-sections
    SigmaT = macro_xs(REACTION_TOTAL, particle_container, mcdc, data)
    SigmaS = macro_xs(
        REACTION_NEUTRON_ELASTIC_SCATTERING, particle_container, mcdc, data
    )
    SigmaC = macro_xs(REACTION_NEUTRON_CAPTURE, particle_container, mcdc, data)
    SigmaF = macro_xs(REACTION_NEUTRON_FISSION, particle_container, mcdc, data)

    # Implicit capture
    if mcdc["implicit_capture"]["active"]:
        particle["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    # Sample reaction type and perform the reaction
    xi = rng.lcg(particle_container) * SigmaT
    total = SigmaS
    if total > xi:
        scattering(particle_container, prog, data)
    else:
        total += SigmaF
        if total > xi:
            fission(particle_container, prog, data)
        else:
            particle["alive"] = False


# ======================================================================================
# Reactions
# ======================================================================================


@njit
def scattering(particle_container, prog, data):

    # Particle attributes
    particle = particle_container[0]
    g = particle["g"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Material attributes
    material = mcdc["multigroup_materials"][particle["material_ID"]]
    G = material["G"]

    # Kill the current particle
    particle["alive"] = False

    # Adjust production and product weights if weighted emission
    weight_production = 1.0
    weight_product = particle["w"]
    if mcdc["weighted_emission"]["active"]:
        weight_target = mcdc["weighted_emission"]["weight_target"]
        weight_production = particle["w"] / weight_target
        weight_product = weight_target

    # Get number of secondaries
    nu_s = mcdc_get.multigroup_material.mgxs_nu_s(g, material, data)
    N = int(math.floor(weight_production * nu_s + rng.lcg(particle_container)))

    # Set up secondary partice container
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_product

        # Sample scattering angle
        mu0 = 2.0 * rng.lcg(particle_container_new) - 1.0

        # Scatter direction
        azi = 2.0 * PI * rng.lcg(particle_container_new)
        ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)
        particle_new["ux"] = ux_new
        particle_new["uy"] = uy_new
        particle_new["uz"] = uz_new

        # Get outgoing spectrum
        chi_s = mcdc_get.multigroup_material.mgxs_chi_s_vector(g, material, data)

        # Sample outgoing energy
        xi = rng.lcg(particle_container_new)
        total = 0.0
        for g_out in range(G):
            total += chi_s[g_out]
            if total > xi:
                break
        particle_new["g"] = g_out

        # Bank, but keep it if it is the last particle
        if n == N - 1:
            particle["alive"] = True
            particle["ux"] = particle_new["ux"]
            particle["uy"] = particle_new["uy"]
            particle["uz"] = particle_new["uz"]
            particle["g"] = particle_new["g"]
            particle["E"] = particle_new["E"]
            particle["w"] = particle_new["w"]
        else:
            particle_bank_module.bank_active_particle(particle_container_new, prog)


@njit
def fission(particle_container, prog, data):
    settings = mcdc["settings"]

    # Particle properties
    particle = particle_container[0]
    g = particle["g"]

    # Material properties
    material = mcdc["multigroup_materials"][particle["material_ID"]]
    G = material["G"]
    J = material["J"]

    # Kill the current particle
    particle["alive"] = False

    # Adjust production and product weights if weighted emission
    weight_production = 1.0
    weight_product = particle["w"]
    if mcdc["weighted_emission"]["active"]:
        weight_target = mcdc["weighted_emission"]["weight_target"]
        weight_production = particle["w"] / weight_target
        weight_product = weight_target

    # Fission yields
    nu = mcdc_get.multigroup_material.mgxs_nu_f(g, material, data)
    nu_p = mcdc_get.multigroup_material.mgxs_nu_p(g, material, data)
    if J > 0:
        nu_d = mcdc_get.multigroup_material.mgxs_nu_d_vector(g, material, data)

    # Get number of secondaries
    N = int(
        math.floor(weight_production * nu / mcdc["k_eff"] + rng.lcg(particle_container))
    )

    # Set up secondary partice container
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_product

        # Sample isotropic direction
        ux_new, uy_new, uz_new = sample_isotropic_direction(particle_container_new)
        particle_new["ux"] = ux_new
        particle_new["uy"] = uy_new
        particle_new["uz"] = uz_new

        # Prompt or delayed?
        xi = rng.lcg(particle_container_new) * nu
        total = nu_p
        if xi < total:
            prompt = True
            spectrum = mcdc_get.multigroup_material.mgxs_chi_p_vector(g, material, data)
        else:
            prompt = False

            # Determine delayed group, decay constant, and spectrum
            for j in range(J):
                total += nu_d[j]
                if xi < total:
                    spectrum = mcdc_get.multigroup_material.mgxs_chi_d_vector(
                        j, material, data
                    )
                    decay = mcdc_get.multigroup_material.mgxs_decay_rate(
                        j, material, data
                    )
                    break

        # Sample outgoing energy
        xi = rng.lcg(particle_container_new)
        tot = 0.0
        for g_out in range(G):
            tot += spectrum[g_out]
            if tot > xi:
                break
        particle_new["g"] = g_out

        # Sample emission time
        if not prompt:
            xi = rng.lcg(particle_container_new)
            particle_new["t"] -= math.log(xi) / decay

        # Eigenvalue mode: bank right away
        if settings["eigenvalue_mode"]:
            particle_bank_module.bank_census_particle(particle_container_new, prog)
            continue
        # Below is only relevant for fixed-source problem

        # Skip if it's beyond time boundary
        if particle_new["t"] > settings["time_boundary"]:
            continue

        # Check if it hits current or next census times
        hit_current_census = False
        hit_future_census = False
        idx_census = mcdc["idx_census"]
        if settings["N_census"] > 1:
            if particle_new["t"] > mcdc_get.settings.census_time(
                idx_census, settings, data
            ):
                hit_current_census = True
                if particle_new["t"] > mcdc_get.settings.census_time(
                    idx_census + 1, settings, data
                ):
                    hit_future_census = True

        # Not hitting census --> add to active bank
        if not hit_current_census:
            # Keep it if it is the last particle
            if n == N - 1:
                particle["alive"] = True
                particle["ux"] = particle_new["ux"]
                particle["uy"] = particle_new["uy"]
                particle["uz"] = particle_new["uz"]
                particle["t"] = particle_new["t"]
                particle["g"] = particle_new["g"]
                particle["E"] = particle_new["E"]
                particle["w"] = particle_new["w"]
            else:
                particle_bank_module.bank_active_particle(particle_container_new, prog)

        # Hit future census --> add to future bank
        elif hit_future_census:
            # Particle will participate in the future
            particle_bank_module.bank_future_particle(particle_container_new, prog)

        # Hit current census --> add to census bank
        else:
            # Particle will participate after the current census is completed
            particle_bank_module.bank_census_particle(particle_container_new, prog)
