import math

from numba import njit

####

import mcdc.adapt as adapt
import mcdc.kernel as kernel
import mcdc.mcdc_get as mcdc_get
import mcdc.type_ as type_

from mcdc.constant import (
    PI,
    REACTION_TOTAL,
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    REACTION_NEUTRON_FISSION_DELAYED,
    REACTION_NEUTRON_FISSION_PROMPT,
)
from mcdc.physics.util import scatter_direction, sample_isotropic_direction


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, material, data):
    particle = particle_container[0]
    return mcdc_get.material.mgxs_speed(particle["g"], material, data)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    g = particle["g"]
    if reaction_type == REACTION_TOTAL:
        return mcdc_get.material.mgxs_total(g, material, data)
    elif reaction_type == REACTION_NEUTRON_CAPTURE:
        return mcdc_get.material.mgxs_capture(g, material, data)
    elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
        return mcdc_get.material.mgxs_scatter(g, material, data)
    elif reaction_type == REACTION_NEUTRON_FISSION:
        return mcdc_get.material.mgxs_fission(g, material, data)
    return 0.0


@njit
def neutron_production_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    g = particle["g"]
    if reaction_type == REACTION_TOTAL:
        total = 0.0
        total += neutron_production_xs(
            REACTION_NEUTRON_ELASTIC_SCATTERING,
            material,
            particle_container,
            mcdc,
            data,
        )
        total += neutron_production_xs(
            REACTION_NEUTRON_FISSION, material, particle_container, mcdc, data
        )
        return total
    elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
        nu = mcdc_get.material.mgxs_nu_s(g, material, data)
        xs = mcdc_get.material.mgxs_scatter(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_NEUTRON_FISSION:
        nu = mcdc_get.material.mgxs_nu_f(g, material, data)
        xs = mcdc_get.material.mgxs_fission(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_NEUTRON_FISSION_PROMPT:
        nu = mcdc_get.material.mgxs_nu_p(g, material, data)
        xs = mcdc_get.material.mgxs_fission(g, material, data)
        return nu * xs
    elif reaction_type == REACTION_NEUTRON_FISSION_DELAYED:
        nu = mcdc_get.material.mgxs_nu_d_total(g, material, data)
        xs = mcdc_get.material.mgxs_fission(g, material, data)
        return nu * xs


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)

    # Get the reaction cross-sections
    material = mcdc["materials"][particle["material_ID"]]
    SigmaT = macro_xs(REACTION_TOTAL, material, particle_container, mcdc, data)
    SigmaS = macro_xs(
        REACTION_NEUTRON_ELASTIC_SCATTERING, material, particle_container, mcdc, data
    )
    SigmaC = macro_xs(
        REACTION_NEUTRON_CAPTURE, material, particle_container, mcdc, data
    )
    SigmaF = macro_xs(
        REACTION_NEUTRON_FISSION, material, particle_container, mcdc, data
    )

    # Implicit capture
    if mcdc["technique"]["implicit_capture"]:
        particle["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    # Sample reaction type and perform the reaction
    xi = kernel.rng(particle_container) * SigmaT
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
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    g = particle["g"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Material attributes
    material = mcdc["materials"][particle["material_ID"]]
    G = material["G"]

    # Kill the current particle
    particle["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = particle["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = particle["w"]

    # Get number of secondaries
    nu_s = mcdc_get.material.mgxs_nu_s(g, material, data)
    N = int(math.floor(weight_eff * nu_s + kernel.rng(particle_container)))

    # Set up secondary partice container
    particle_container_new = adapt.local_array(1, type_.particle_record)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        kernel.split_as_record(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_new

        # Sample scattering angle
        mu0 = 2.0 * kernel.rng(particle_container_new) - 1.0

        # Scatter direction
        azi = 2.0 * PI * kernel.rng(particle_container_new)
        ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)
        particle_new["ux"] = ux_new
        particle_new["uy"] = uy_new
        particle_new["uz"] = uz_new

        # Get outgoing spectrum
        chi_s = mcdc_get.material.mgxs_chi_s_vector(g, material, data)

        # Sample outgoing energy
        xi = kernel.rng(particle_container_new)
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
            adapt.add_active(particle_container_new, prog)


@njit
def fission(particle_container, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    g = particle["g"]

    # Material attributes
    material = mcdc["materials"][particle["material_ID"]]
    G = material["G"]
    J = material["J"]
    nu = mcdc_get.material.mgxs_nu_f(g, material, data)
    nu_p = mcdc_get.material.mgxs_nu_p(g, material, data)
    if J > 0:
        nu_d = mcdc_get.material.mgxs_nu_d_vector(g, material, data)

    # Kill the current particle
    particle["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = particle["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = particle["w"]

    # Get number of secondaries
    N = int(
        math.floor(weight_eff * nu / mcdc["k_eff"] + kernel.rng(particle_container))
    )

    # Set up secondary partice container
    particle_container_new = adapt.local_array(1, type_.particle_record)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        kernel.split_as_record(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_new

        # Sample isotropic direction
        ux_new, uy_new, uz_new = sample_isotropic_direction(particle_container_new)
        particle_new["ux"] = ux_new
        particle_new["uy"] = uy_new
        particle_new["uz"] = uz_new

        # Prompt or delayed?
        xi = kernel.rng(particle_container_new) * nu
        total = nu_p
        if xi < total:
            prompt = True
            spectrum = mcdc_get.material.mgxs_chi_p_vector(g, material, data)
        else:
            prompt = False

            # Determine delayed group, decay constant, and spectrum
            for j in range(J):
                total += nu_d[j]
                if xi < total:
                    spectrum = mcdc_get.material.mgxs_chi_d_vector(j, material, data)
                    decay = mcdc_get.material.mgxs_decay_rate(j, material, data)
                    break

        # Sample outgoing energy
        xi = kernel.rng(particle_container_new)
        tot = 0.0
        for g_out in range(G):
            tot += spectrum[g_out]
            if tot > xi:
                break
        particle_new["g"] = g_out

        # Sample emission time
        if not prompt:
            xi = kernel.rng(particle_container_new)
            particle_new["t"] -= math.log(xi) / decay

        # Eigenvalue mode: bank right away
        if mcdc["settings"]["eigenvalue_mode"]:
            adapt.add_census(particle_container_new, prog)
            continue
        # Below is only relevant for fixed-source problem

        # Skip if it's beyond time boundary
        if particle_new["t"] > mcdc["settings"]["time_boundary"]:
            continue

        # Check if it is beyond current or next census times
        hit_census = False
        hit_next_census = False
        idx_census = mcdc["idx_census"]
        if idx_census < mcdc["settings"]["N_census"] - 1:
            settings = mcdc["settings"]
            if particle["t"] > mcdc_get.settings.census_time(
                idx_census + 1, settings, data
            ):
                hit_census = True
                hit_next_census = True
            elif particle_new["t"] > mcdc_get.settings.census_time(
                idx_census, settings, data
            ):
                hit_census = True

        if not hit_census:
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
                adapt.add_active(particle_container_new, prog)
        elif not hit_next_census:
            # Particle will participate after the current census
            adapt.add_census(particle_container_new, prog)
        else:
            # Particle will participate in the future
            adapt.add_future(particle_container_new, prog)
