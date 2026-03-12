import math
import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng

from mcdc.constant import (
    ANGLE_DISTRIBUTED,
    ANGLE_ENERGY_CORRELATED,
    ANGLE_ISOTROPIC,
    BOLTZMANN_K,
    THERMAL_THRESHOLD_FACTOR,
    LIGHT_SPEED,
    NEUTRON_MASS,
    PI,
    PI_HALF,
    PI_SQRT,
    NEUTRON_REACTION_INELASTIC_SCATTERING,
    NEUTRON_REACTION_TOTAL,
    NEUTRON_REACTION_CAPTURE,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    REFERENCE_FRAME_COM,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_correlated_distribution,
    sample_distribution,
    sample_isotropic_cosine,
    sample_isotropic_direction,
    sample_multi_table,
)
from mcdc.transport.physics.util import (
    evaluate_neutron_xs_energy_grid,
    scatter_direction,
)
from mcdc.transport.util import find_bin, linear_interpolation

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    particle = particle_container[0]
    E = particle["E"]
    mass = NEUTRON_MASS
    return LIGHT_SPEED * math.sqrt(E * (E + 2.0 * mass)) / (E + mass)


@njit
def particle_energy_from_speed(speed):
    beta = speed / LIGHT_SPEED
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    mass = NEUTRON_MASS
    return mass * (gamma - 1.0)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["native_materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0

    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = mcdc["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, nuclide, data)

        total += nuclide_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, nuclide, data):
    idx, E0, E1 = evaluate_neutron_xs_energy_grid(E, nuclide, data)
    if reaction_type == NEUTRON_REACTION_TOTAL:
        xs0 = mcdc_get.nuclide.neutron_total_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.neutron_total_xs(idx + 1, nuclide, data)
    elif reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
        xs0 = mcdc_get.nuclide.neutron_elastic_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.neutron_elastic_xs(idx + 1, nuclide, data)
    elif reaction_type == NEUTRON_REACTION_CAPTURE:
        xs0 = mcdc_get.nuclide.neutron_capture_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.neutron_capture_xs(idx + 1, nuclide, data)
    elif reaction_type == NEUTRON_REACTION_INELASTIC_SCATTERING:
        xs0 = mcdc_get.nuclide.neutron_inelastic_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.neutron_inelastic_xs(idx + 1, nuclide, data)
    elif reaction_type == NEUTRON_REACTION_FISSION:
        xs0 = mcdc_get.nuclide.neutron_fission_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.neutron_fission_xs(idx + 1, nuclide, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction_base, nuclide, data):
    idx, E0, E1 = evaluate_neutron_xs_energy_grid(E, nuclide, data)

    # Apply offset
    offset = reaction_base["xs_offset_"]
    if idx < offset:
        return 0.0
    else:
        idx -= offset

    xs0 = mcdc_get.neutron_reaction.xs(idx, reaction_base, data)
    xs1 = mcdc_get.neutron_reaction.xs(idx + 1, reaction_base, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def neutron_production_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material_base = mcdc["materials"][particle["material_ID"]]
    material = mcdc["native_materials"][material_base["child_ID"]]

    if reaction_type == NEUTRON_REACTION_TOTAL:
        elastic_type = NEUTRON_REACTION_ELASTIC_SCATTERING
        inelastic_type = NEUTRON_REACTION_INELASTIC_SCATTERING
        fission_type = NEUTRON_REACTION_FISSION
        elastic_xs = neutron_production_xs(elastic_type, particle_container, mcdc, data)
        inelastic_xs = neutron_production_xs(
            inelastic_type, particle_container, mcdc, data
        )
        fission_xs = neutron_production_xs(fission_type, particle_container, mcdc, data)
        return elastic_xs + inelastic_xs + fission_xs

    elif reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
        return macro_xs(reaction_type, particle_container, mcdc, data)

    elif reaction_type == NEUTRON_REACTION_CAPTURE:
        return 0.0

    elif reaction_type == NEUTRON_REACTION_INELASTIC_SCATTERING:
        total = 0.0
        for i in range(material["N_nuclide"]):
            nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
            nuclide = mcdc["nuclides"][nuclide_ID]

            E = particle["E"]
            nuclide_density = mcdc_get.native_material.nuclide_densities(
                i, material, data
            )

            for j in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
                reaction_ID = int(
                    mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
                        j, nuclide, data
                    )
                )
                reaction_base = mcdc["neutron_reactions"][reaction_ID]
                reaction = mcdc["neutron_inelastic_scattering_reactions"][
                    reaction_base["child_ID"]
                ]

                xs = reaction_micro_xs(E, reaction_base, nuclide, data)
                nu = reaction["multiplicity"]
                total += nuclide_density * nu * xs

        return total

    elif reaction_type == NEUTRON_REACTION_FISSION:
        if not material_base["fissionable"]:
            return 0.0

        total = 0.0
        for i in range(material["N_nuclide"]):
            nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
            nuclide = mcdc["nuclides"][nuclide_ID]
            if not nuclide["fissionable"]:
                continue

            E = particle["E"]
            nuclide_density = mcdc_get.native_material.nuclide_densities(
                i, material, data
            )

            for j in range(nuclide["N_neutron_fission_reaction"]):
                reaction_ID = int(
                    mcdc_get.nuclide.neutron_fission_reaction_IDs(j, nuclide, data)
                )
                reaction_base = mcdc["neutron_reactions"][reaction_ID]
                reaction = mcdc["neutron_fission_reactions"][reaction_base["child_ID"]]

                xs = reaction_micro_xs(E, reaction_base, nuclide, data)
                nu_p = neutron_fission_prompt_multiplicity(E, nuclide, mcdc, data)
                nu_d = neutron_fission_delayed_multiplicity(E, nuclide, mcdc, data)
                nu = nu_d + nu_p
                total += nuclide_density * nu * xs

        return total

    else:
        return -1.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["native_materials"][particle["material_ID"]]

    # Particle properties
    E_in = particle["E"]
    w_in = particle["w"]

    # ==================================================================================
    # Sample colliding nuclide
    # ==================================================================================

    SigmaT = macro_xs(NEUTRON_REACTION_TOTAL, particle_container, mcdc, data)

    # Implicit capture
    if mcdc["implicit_capture"]["active"]:
        SigmaC = macro_xs(NEUTRON_REACTION_CAPTURE, particle_container, mcdc, data)
        particle["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = mcdc["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        sigmaT = total_micro_xs(NEUTRON_REACTION_TOTAL, E, nuclide, data)

        if mcdc["implicit_capture"]["active"]:
            sigmaC = total_micro_xs(NEUTRON_REACTION_CAPTURE, E, nuclide, data)
            particle["w"] *= (sigmaT - sigmaC) / sigmaT
            sigmaT -= sigmaC

        SigmaT_nuclide = nuclide_density * sigmaT
        total += SigmaT_nuclide

        if total > xi:
            break

    # Transported weight after implicit-capture adjustment
    w_transport = particle["w"]

    # Energy deposited by implicit capture portion
    edep_weighted = (w_in - w_transport) * E_in

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    sigma_elastic = total_micro_xs(
        NEUTRON_REACTION_ELASTIC_SCATTERING, E, nuclide, data
    )
    sigma_inelastic = total_micro_xs(
        NEUTRON_REACTION_INELASTIC_SCATTERING, E, nuclide, data
    )
    sigma_fission = total_micro_xs(NEUTRON_REACTION_FISSION, E, nuclide, data)

    xi = rng.lcg(particle_container) * sigmaT

    # Elastic scattering
    total = sigma_elastic
    if xi < total:
        total -= sigma_elastic
        for i in range(nuclide["N_neutron_elastic_scattering_reaction"]):
            reaction_ID = int(
                mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs(
                    i, nuclide, data
                )
            )
            reaction = mcdc["neutron_elastic_scattering_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["neutron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, nuclide, data)
            if xi < total:
                E_out_weighted = elastic_scattering(
                    reaction, particle_container, nuclide, mcdc, data
                )
                dep = E_in * w_transport - E_out_weighted
                if dep > 0.0:
                    edep_weighted += dep
                if edep_weighted < 0.0:
                    edep_weighted = 0.0
                return edep_weighted

    # Capture
    if not mcdc["implicit_capture"]["active"]:
        sigma_capture = total_micro_xs(NEUTRON_REACTION_CAPTURE, E, nuclide, data)
        total += sigma_capture
        if xi < total:
            particle["alive"] = False
            edep_weighted += E_in * w_transport
            if edep_weighted < 0.0:
                edep_weighted = 0.0
            return edep_weighted

    # Inelastic scattering
    total += sigma_inelastic
    if xi < total:
        total -= sigma_inelastic

        for i in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
            reaction_ID = int(
                mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
                    i, nuclide, data
                )
            )
            reaction = mcdc["neutron_inelastic_scattering_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["neutron_reactions"][reaction_base_ID]
            xs = reaction_micro_xs(E, reaction_base, nuclide, data)
            total += xs
            if xi < total:
                E_out_weighted = inelastic_scattering(
                    reaction, particle_container, nuclide, mcdc, data
                )
                dep = E_in * w_transport - E_out_weighted
                if dep > 0.0:
                    edep_weighted += dep
                if edep_weighted < 0.0:
                    edep_weighted = 0.0
                return edep_weighted

    # Fission (arive here only if nuclide is fissionable)
    total += sigma_fission
    if xi < total:
        total -= sigma_fission
        for i in range(nuclide["N_neutron_fission_reaction"]):
            reaction_ID = int(
                mcdc_get.nuclide.neutron_fission_reaction_IDs(i, nuclide, data)
            )
            reaction = mcdc["neutron_fission_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["neutron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, nuclide, data)
            if xi < total:
                E_out_weighted = fission(
                    reaction, particle_container, nuclide, mcdc, data
                )
                dep = E_in * w_transport - E_out_weighted
                if dep > 0.0:
                    edep_weighted += dep
                if edep_weighted < 0.0:
                    edep_weighted = 0.0
                return edep_weighted

    if edep_weighted < 0.0:
        edep_weighted = 0.0
    return edep_weighted


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def elastic_scattering(reaction, particle_container, nuclide, mcdc, data):
    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Sample nucleus thermal velocity
    A = nuclide["atomic_weight_ratio"]
    temperature = nuclide["temperature"]
    if E > THERMAL_THRESHOLD_FACTOR * BOLTZMANN_K * temperature:
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
    else:
        Vx, Vy, Vz = sample_nucleus_velocity(A, particle_container)

    # =========================================================================
    # COM kinematics
    # =========================================================================

    # Particle speed
    speed = particle_speed(particle_container)

    # Neutron velocity - LAB
    vx = speed * ux
    vy = speed * uy
    vz = speed * uz

    # COM velocity
    COM_x = (vx + A * Vx) / (1.0 + A)
    COM_y = (vy + A * Vy) / (1.0 + A)
    COM_z = (vz + A * Vz) / (1.0 + A)

    # Neutron velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Neutron speed - COM
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Neutron initial direction - COM
    ux = vx / speed
    uy = vy / speed
    uz = vz / speed

    # Sample the scattering cosine from the multi-PDF distribution
    multi_table = mcdc["multi_table_distributions"][reaction["mu_table_ID"]]
    mu0 = sample_multi_table(E, particle_container, multi_table, data)

    # Scatter the direction in COM
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Neutron final velocity - COM
    vx = speed * ux_new
    vy = speed * uy_new
    vz = speed * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx = vx + COM_x
    vy = vy + COM_y
    vz = vz + COM_z

    # Final energy - LAB
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    particle["E"] = particle_energy_from_speed(speed)

    # Final direction - LAB
    particle["ux"] = vx / speed
    particle["uy"] = vy / speed
    particle["uz"] = vz / speed

    return particle["E"] * particle["w"]


@njit
def sample_nucleus_velocity(A, particle_container):
    particle = particle_container[0]

    # Particle speed
    speed = particle_speed(particle_container)

    # Maxwellian parameter
    beta = math.sqrt(2.0659834e-11 * A)
    # The constant above is
    #   (1.674927471e-27 kg) / (1.38064852e-19 cm^2 kg s^-2 K^-1) / (293.6 K)/2

    # Sample nuclide speed candidate V_tilda and
    #   nuclide-neutron polar cosine candidate mu_tilda via
    #   rejection sampling
    y = beta * speed
    while True:
        if rng.lcg(particle_container) < 2.0 / (2.0 + PI_SQRT * y):
            x = math.sqrt(
                -math.log(rng.lcg(particle_container) * rng.lcg(particle_container))
            )
        else:
            cos_val = math.cos(PI_HALF * rng.lcg(particle_container))
            x = math.sqrt(
                -math.log(rng.lcg(particle_container))
                - math.log(rng.lcg(particle_container)) * cos_val * cos_val
            )
        V_tilda = x / beta
        mu_tilda = 2.0 * rng.lcg(particle_container) - 1.0

        # Accept candidate V_tilda and mu_tilda?
        if rng.lcg(particle_container) > math.sqrt(
            speed * speed + V_tilda * V_tilda - 2.0 * speed * V_tilda * mu_tilda
        ) / (speed + V_tilda):
            break

    # Set nuclide velocity - LAB
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux, uy, uz = scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu_tilda, azi
    )
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz


# ======================================================================================
# Inelastic scattering
# ======================================================================================


@njit
def inelastic_scattering(reaction, particle_container, nuclide, mcdc, data):
    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Kill the current particle
    particle["alive"] = False

    # Number of secondaries and spectra
    N = reaction["multiplicity"]
    N_spectrum = reaction["N_spectrum"]
    use_all_spectrum = N == N_spectrum

    # Set up secondary partice container
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]
    E_out_weighted = 0.0

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # ==============================================================================
        # Sample angle (if not energy-correlated)
        # ==============================================================================

        angle_type = reaction["angle_type"]
        if angle_type == ANGLE_ENERGY_CORRELATED:
            pass
        elif angle_type == ANGLE_ISOTROPIC:
            mu = sample_isotropic_cosine(particle_container_new)
        elif angle_type == ANGLE_DISTRIBUTED:
            distribution_base = mcdc["distributions"][reaction["mu_ID"]]
            multi_table = mcdc["multi_table_distributions"][
                distribution_base["child_ID"]
            ]
            mu = sample_multi_table(E, particle_container_new, multi_table, data)

        # ==============================================================================
        # Sample energy (also angle if correlated)
        # ==============================================================================

        # Get energy spectrum
        if use_all_spectrum:
            ID = int(
                mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
                    n, reaction, data
                )
            )
            spectrum_base = mcdc["distributions"][ID]
        else:
            probability_grid = mcdc_get.neutron_inelastic_scattering_reaction.spectrum_probability_grid_all(
                reaction, data
            )
            probability_idx = find_bin(E, probability_grid)
            xi = rng.lcg(particle_container_new)
            total = 0.0
            for j in range(N_spectrum):
                probability = (
                    mcdc_get.neutron_inelastic_scattering_reaction.spectrum_probability(
                        probability_idx, j, reaction, data
                    )
                )
                total += probability
                if xi < total:
                    ID = int(
                        mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
                            j, reaction, data
                        )
                    )
                    spectrum_base = mcdc["distributions"][ID]
                    break

        # Sample energy
        if not angle_type == ANGLE_ENERGY_CORRELATED:
            E_new = sample_distribution(
                E, spectrum_base, particle_container_new, mcdc, data, scale=True
            )
        else:
            E_new, mu = sample_correlated_distribution(
                E, spectrum_base, particle_container_new, mcdc, data, scale=True
            )

        # ==============================================================================
        # Frame transformation
        # ==============================================================================

        reaction_base = mcdc["neutron_reactions"][int(reaction["parent_ID"])]
        reference_frame = reaction_base["reference_frame"]
        if reference_frame == REFERENCE_FRAME_COM:
            A = nuclide["atomic_weight_ratio"]
            mu_COM = mu
            E_COM = E_new

            E_new = (
                E_COM + (E + 2 * mu_COM * (A + 1) * math.sqrt(E * E_COM)) / (A + 1) ** 2
            )
            mu = mu_COM * math.sqrt(E_COM / E_new) + math.sqrt(E / E_new) / (A + 1)

        azi = 2.0 * PI * rng.lcg(particle_container_new)
        ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu, azi)

        # Now the secondary angle and energy are finalized
        particle_new["ux"] = ux_new
        particle_new["uy"] = uy_new
        particle_new["uz"] = uz_new
        particle_new["E"] = E_new

        E_out_weighted += particle_new["E"] * particle_new["w"]

        # ==============================================================================
        # Bank the new particle
        # ==============================================================================

        # Keep it if it is the last particle
        if n == N - 1:
            particle["alive"] = True
            particle["ux"] = particle_new["ux"]
            particle["uy"] = particle_new["uy"]
            particle["uz"] = particle_new["uz"]
            particle["E"] = particle_new["E"]
        else:
            particle_bank_module.bank_active_particle(particle_container_new, mcdc)

    return E_out_weighted


# ======================================================================================
# Fission
# ======================================================================================


@njit
def fission(reaction, particle_container, nuclide, mcdc, data):
    settings = mcdc["settings"]

    # Particle properties
    particle = particle_container[0]
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

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
    N_delayed = nuclide["N_neutron_fission_delayed_precursor"]
    nu_p = neutron_fission_prompt_multiplicity(E, nuclide, mcdc, data)
    nu_d = neutron_fission_delayed_multiplicity(E, nuclide, mcdc, data)
    nu = nu_p + nu_d

    # Get number of secondaries
    N = int(
        math.floor(weight_production * nu / mcdc["k_eff"] + rng.lcg(particle_container))
    )

    # Set up secondary partice container
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]
    E_out_weighted = 0.0

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_product

        # Prompt or delayed?
        prompt = True
        delayed_group = -1
        xi = rng.lcg(particle_container_new)
        total = nu_p
        if xi > total:
            prompt = False
            # Determine delayed group
            for j in range(N_delayed):
                fraction = mcdc_get.nuclide.neutron_fission_delayed_fractions(
                    j, nuclide, data
                )
                total += fraction
                if xi < total:
                    delayed_group = j
                    break

        # ==============================================================================
        # Sample prompt neutron
        # ==============================================================================

        if prompt:
            # Sample angle (if not energy-correlated)
            angle_type = reaction["angle_type"]
            if angle_type == ANGLE_ENERGY_CORRELATED:
                pass
            elif angle_type == ANGLE_ISOTROPIC:
                mu = sample_isotropic_cosine(particle_container_new)
            elif angle_type == ANGLE_DISTRIBUTED:
                distribution_base = mcdc["distributions"][reaction["mu_ID"]]
                multi_table = mcdc["multi_table_distributions"][
                    distribution_base["child_ID"]
                ]
                mu = sample_multi_table(E, particle_container_new, multi_table, data)

            # Sample energy (also angle if correlated)
            spectrum_base = mcdc["distributions"][reaction["spectrum_ID"]]
            if not angle_type == ANGLE_ENERGY_CORRELATED:
                E_new = sample_distribution(
                    E, spectrum_base, particle_container_new, mcdc, data, scale=True
                )
            else:
                E_new, mu = sample_correlated_distribution(
                    E, spectrum_base, particle_container_new, mcdc, data, scale=True
                )

            # Frame transformation
            reaction_base = mcdc["neutron_reactions"][int(reaction["parent_ID"])]
            reference_frame = reaction_base["reference_frame"]
            if reference_frame == REFERENCE_FRAME_COM:
                A = nuclide["atomic_weight_ratio"]
                mu_COM = mu
                E_COM = E_new

                E_new = (
                    E_COM
                    + (E + 2 * mu_COM * (A + 1) * math.sqrt(E * E_COM)) / (A + 1) ** 2
                )
                mu = mu_COM * math.sqrt(E_COM / E_new) + math.sqrt(E / E_new) / (A + 1)

            azi = 2.0 * PI * rng.lcg(particle_container_new)
            ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu, azi)

            # Now the secondary angle and energy are finalized
            particle_new["ux"] = ux_new
            particle_new["uy"] = uy_new
            particle_new["uz"] = uz_new
            particle_new["E"] = E_new

        # ==============================================================================
        # Sample delayed fission neutron
        # ==============================================================================

        else:
            # Sample isotropic angle
            ux_new, uy_new, uz_new = sample_isotropic_direction(particle_container_new)

            # Sample emission time
            decay_rate = mcdc_get.nuclide.neutron_fission_delayed_fractions(
                delayed_group, nuclide, data
            )
            if not prompt:
                xi = rng.lcg(particle_container_new)
                particle_new["t"] -= math.log(xi) / decay_rate

        E_out_weighted += particle_new["E"] * particle_new["w"]

        # Eigenvalue mode: bank right away
        if settings["eigenvalue_mode"]:
            particle_bank_module.bank_census_particle(particle_container_new, mcdc)
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
                particle_bank_module.bank_active_particle(particle_container_new, mcdc)

        # Hit future census --> add to future bank
        elif hit_future_census:
            # Particle will participate in the future
            particle_bank_module.bank_future_particle(particle_container_new, mcdc)

        # Hit current census --> add to census bank
        else:
            # Particle will participate after the current census is completed
            particle_bank_module.bank_census_particle(particle_container_new, mcdc)

    return E_out_weighted


@njit
def neutron_fission_prompt_multiplicity(E, nuclide, mcdc, data):
    data_base = mcdc["data"][nuclide["neutron_fission_prompt_multiplicity_ID"]]
    return evaluate_data(E, data_base, mcdc, data)


@njit
def neutron_fission_delayed_multiplicity(E, nuclide, mcdc, data):
    data_base = mcdc["data"][nuclide["neutron_fission_delayed_multiplicity_ID"]]
    return evaluate_data(E, data_base, mcdc, data)
