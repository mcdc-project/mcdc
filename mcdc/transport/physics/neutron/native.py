import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    COINCIDENCE_TOLERANCE_TIME,
    EVENT_TIME_CENSUS,
    DISTRIBUTION_NONE,
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
from mcdc.transport.data import evaluate_data_clamped
from mcdc.transport.distribution import (
    sample_correlated_distribution_with_scale,
    sample_distribution,
    sample_distribution_with_scale,
    sample_isotropic_direction,
)
from mcdc.transport.physics.util import (
    evaluate_neutron_xs_energy_grid,
    evaluate_reaction_xs,
    interpolate_xs,
    relativistic_energy_from_speed,
    scatter_direction,
)
from mcdc.transport.util import find_bin

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
    return relativistic_energy_from_speed(speed, NEUTRON_MASS)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0

    for i in range(material["N_nuclide"]):
        nuclide_ID = mcdc_get.material.nuclide_IDs(i, material, data)
        nuclide = simulation["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, nuclide, simulation, data)

        total += nuclide_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, nuclide, simulation, data):
    if not nuclide["neutron_xs_is_linear"]:
        total = 0.0
        for i in range(nuclide["N_neutron_reaction"]):
            ID = mcdc_get.nuclide.neutron_reaction_IDs(i, nuclide, data)
            reaction = simulation["neutron_reactions"][ID]
            if (
                reaction_type == NEUTRON_REACTION_TOTAL
                or reaction["sub_type"] == reaction_type
            ):
                total += reaction_micro_xs(E, reaction, nuclide, data)
        return total
    grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)
    E = min(max(E, grid[0]), grid[-1])
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
    else:
        # Should be unreachable
        xs0 = 0.0
        xs1 = 0.0
    return interpolate_xs(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction, nuclide, data):
    """Evaluate a reaction on the shared grid with its own interpolation regions."""
    grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)
    return evaluate_reaction_xs(
        E,
        grid,
        reaction["xs_offset_"],
        mcdc_get.neutron_reaction.xs_all(reaction, data),
        mcdc_get.neutron_reaction.xs_interpolations_all(reaction, data),
        mcdc_get.neutron_reaction.xs_interpolation_boundaries_all(reaction, data),
    )


@njit
def neutron_production_xs(reaction_type, particle_container, simulation, data):
    # Total production
    if reaction_type == NEUTRON_REACTION_TOTAL:
        elastic_xs = macro_xs(
            NEUTRON_REACTION_ELASTIC_SCATTERING, particle_container, simulation, data
        )
        inelastic_xs = _neutron_inelastic_scattering_production_xs(
            particle_container, simulation, data
        )
        fission_xs = _neutron_fission_production_xs(
            particle_container, simulation, data
        )
        return elastic_xs + inelastic_xs + fission_xs

    # Elastic scattering production
    elif reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
        return macro_xs(reaction_type, particle_container, simulation, data)

    # Capture production (none)
    elif reaction_type == NEUTRON_REACTION_CAPTURE:
        return 0.0

    # Inelastic scattering production
    elif reaction_type == NEUTRON_REACTION_INELASTIC_SCATTERING:
        return _neutron_inelastic_scattering_production_xs(
            particle_container, simulation, data
        )

    # Fission production
    elif reaction_type == NEUTRON_REACTION_FISSION:
        return _neutron_fission_production_xs(particle_container, simulation, data)

    # Unsupported default
    else:
        return 0.0


@njit
def _neutron_inelastic_scattering_production_xs(particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]

    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = mcdc_get.material.nuclide_IDs(i, material, data)
        nuclide = simulation["nuclides"][nuclide_ID]

        E = particle["E"]
        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)

        for j in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
            reaction_ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
                j, nuclide, data
            )
            reaction = simulation["neutron_reactions"][reaction_ID]
            inelastic_scattering = simulation["neutron_inelastic_scattering_reactions"][
                reaction["sub_ID"]
            ]

            xs = reaction_micro_xs(E, reaction, nuclide, data)
            nu = inelastic_scattering["multiplicity"]
            total += nuclide_density * nu * xs

    return total


@njit
def _neutron_fission_production_xs(particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]

    if not material["fissionable"]:
        return 0.0

    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = mcdc_get.material.nuclide_IDs(i, material, data)
        nuclide = simulation["nuclides"][nuclide_ID]
        if not nuclide["fissionable"]:
            continue

        E = particle["E"]
        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)

        for j in range(nuclide["N_neutron_fission_reaction"]):
            reaction_ID = mcdc_get.nuclide.neutron_fission_reaction_IDs(
                j, nuclide, data
            )
            reaction = simulation["neutron_reactions"][reaction_ID]

            xs = reaction_micro_xs(E, reaction, nuclide, data)
            fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
            nu_p = neutron_fission_prompt_multiplicity(E, fission, simulation, data)
            nu_d = neutron_fission_delayed_multiplicity(E, fission, simulation, data)
            nu = nu_d + nu_p
            total += nuclide_density * nu * xs

    return total


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, interaction_data_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    material = simulation["materials"][particle["material_ID"]]

    # Particle properties
    E = particle["E"]

    # ==================================================================================
    # Sample colliding nuclide
    # ==================================================================================

    SigmaT = macro_xs(NEUTRON_REACTION_TOTAL, particle_container, simulation, data)

    # Implicit capture
    if simulation["technique"]["implicit_capture"]["active"]:
        # Calculate capture fraction
        SigmaC = macro_xs(
            NEUTRON_REACTION_CAPTURE, particle_container, simulation, data
        )
        capture_fraction = SigmaC / SigmaT

        # Deposit energy captured
        interaction_data["energy_deposition"] += E * particle["w"] * capture_fraction

        # Q-value: xs-weighted average over all nuclides and capture reactions
        for i in range(material["N_nuclide"]):
            nuclide_ID = mcdc_get.material.nuclide_IDs(i, material, data)
            nuclide = simulation["nuclides"][nuclide_ID]
            nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
            for j in range(nuclide["N_neutron_capture_reaction"]):
                reaction_ID = mcdc_get.nuclide.neutron_capture_reaction_IDs(
                    j, nuclide, data
                )
                reaction = simulation["neutron_reactions"][reaction_ID]
                xs = reaction_micro_xs(E, reaction, nuclide, data)
                Sigma_rx = nuclide_density * xs
                interaction_data["energy_deposition"] += (
                    reaction["q_value"] * particle["w"] * Sigma_rx / SigmaT
                )

        # Capture particle weight
        particle["w"] *= 1.0 - capture_fraction

        # Adjust total XS
        SigmaT -= SigmaC

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = mcdc_get.material.nuclide_IDs(i, material, data)
        nuclide = simulation["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
        sigmaT = total_micro_xs(NEUTRON_REACTION_TOTAL, E, nuclide, simulation, data)

        if simulation["technique"]["implicit_capture"]["active"]:
            sigmaC = total_micro_xs(
                NEUTRON_REACTION_CAPTURE, E, nuclide, simulation, data
            )
            sigmaT -= sigmaC

        SigmaT_nuclide = nuclide_density * sigmaT
        total += SigmaT_nuclide

        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    sigma_elastic = total_micro_xs(
        NEUTRON_REACTION_ELASTIC_SCATTERING, E, nuclide, simulation, data
    )
    sigma_inelastic = total_micro_xs(
        NEUTRON_REACTION_INELASTIC_SCATTERING, E, nuclide, simulation, data
    )
    sigma_fission = total_micro_xs(
        NEUTRON_REACTION_FISSION, E, nuclide, simulation, data
    )

    xi = rng.lcg(particle_container) * sigmaT

    # Elastic scattering
    total = sigma_elastic
    if xi < total:
        # Sample the actual reaction from the group
        total -= sigma_elastic
        for i in range(nuclide["N_neutron_elastic_scattering_reaction"]):
            reaction_ID = mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs(
                i, nuclide, data
            )
            reaction = simulation["neutron_reactions"][reaction_ID]
            total += reaction_micro_xs(E, reaction, nuclide, data)

            # Execute the reaction
            if xi < total:
                sample_elastic_scattering(
                    reaction,
                    particle_container,
                    interaction_data_container,
                    nuclide,
                    simulation,
                    data,
                )
                return

    # Capture
    if not simulation["technique"]["implicit_capture"]["active"]:
        sigma_capture = total_micro_xs(
            NEUTRON_REACTION_CAPTURE, E, nuclide, simulation, data
        )
        total += sigma_capture
        if xi < total:
            # Sample the actual reaction from the group
            total -= sigma_capture
            for i in range(nuclide["N_neutron_capture_reaction"]):
                reaction_ID = mcdc_get.nuclide.neutron_capture_reaction_IDs(
                    i, nuclide, data
                )
                reaction = simulation["neutron_reactions"][reaction_ID]
                xs = reaction_micro_xs(E, reaction, nuclide, data)
                total += xs

                # Execute the reaction
                if xi < total:
                    capture(
                        reaction,
                        particle_container,
                        interaction_data_container,
                        nuclide,
                        simulation,
                        data,
                    )
                    return

    # Inelastic scattering
    total += sigma_inelastic
    if xi < total:
        # Sample the actual reaction from the group
        total -= sigma_inelastic
        for i in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
            reaction_ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
                i, nuclide, data
            )
            reaction = simulation["neutron_reactions"][reaction_ID]
            xs = reaction_micro_xs(E, reaction, nuclide, data)
            total += xs

            # Execute the reaction
            if xi < total:
                sample_inelastic_scattering(
                    reaction,
                    particle_container,
                    interaction_data_container,
                    nuclide,
                    program,
                    data,
                )
                return

    # Fission (arive here only if nuclide is fissionable)
    total += sigma_fission
    if xi < total:
        # Sample the actual reaction from the group
        total -= sigma_fission
        for i in range(nuclide["N_neutron_fission_reaction"]):
            reaction_ID = mcdc_get.nuclide.neutron_fission_reaction_IDs(
                i, nuclide, data
            )
            reaction = simulation["neutron_reactions"][reaction_ID]
            total += reaction_micro_xs(E, reaction, nuclide, data)

            # Execute the reaction
            if xi < total:
                sample_fission(
                    reaction,
                    particle_container,
                    interaction_data_container,
                    nuclide,
                    program,
                    data,
                )
                return


# ======================================================================================
# Capture
# ======================================================================================


@njit
def capture(
    reaction, particle_container, interaction_data_container, nuclide, simulation, data
):
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    # Terminate the particle
    particle["alive"] = False

    # Energy deposition
    E = particle["E"]
    q_value = reaction["q_value"]
    interaction_data["energy_deposition"] += (E + q_value) * particle["w"]


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def sample_elastic_scattering(
    reaction, particle_container, interaction_data_container, nuclide, simulation, data
):
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    sub_ID = reaction["sub_ID"]
    elastic_scattering = simulation["neutron_elastic_scattering_reactions"][sub_ID]

    # Particle attributes
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Energy deposition
    interaction_data["energy_deposition"] += E * particle["w"]
    # Note: Q-value is zero in elastic scattering

    # Sample nucleus thermal velocity
    A = nuclide["atomic_weight_ratio"]
    temperature = nuclide["temperature"]
    if E > THERMAL_THRESHOLD_FACTOR * BOLTZMANN_K * temperature:
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
    else:
        Vx, Vy, Vz = sample_nucleus_velocity(A, temperature, particle_container)

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

    mu_distribution = simulation["distributions"][elastic_scattering["mu_table_ID"]]
    mu0 = sample_distribution(E, mu_distribution, particle_container, simulation, data)

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

    # Subtract outgoing energy from energy deposition
    interaction_data["energy_deposition"] -= particle["E"] * particle["w"]


@njit
def sample_nucleus_velocity(A, temperature, particle_container):
    """Sample a collision-weighted Maxwellian target velocity at the material temperature."""
    particle = particle_container[0]

    # Particle speed
    speed = particle_speed(particle_container)

    # Maxwellian parameter
    beta = math.sqrt(
        A * NEUTRON_MASS / (2.0 * LIGHT_SPEED**2 * BOLTZMANN_K * temperature)
    )

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
        if rng.lcg(particle_container) < math.sqrt(
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
def sample_inelastic_scattering(
    reaction, particle_container, interaction_data_container, nuclide, program, data
):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    sub_ID = reaction["sub_ID"]
    inelastic_scattering = simulation["neutron_inelastic_scattering_reactions"][sub_ID]

    # Particle attributes
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Kill the current particle
    particle["alive"] = False

    # Energy deposition
    q_value = reaction["q_value"]
    interaction_data["energy_deposition"] += (E + q_value) * particle["w"]

    # Number of secondaries
    N = inelastic_scattering["multiplicity"]

    # Set up secondary partice container
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # ==============================================================================
        # Sample angle (if not energy-correlated)
        # ==============================================================================

        mu_distribution = simulation["distributions"][inelastic_scattering["mu_ID"]]
        if mu_distribution["sub_type"] != DISTRIBUTION_NONE:
            mu = sample_distribution(
                E, mu_distribution, particle_container_new, simulation, data
            )

        # ==============================================================================
        # Sample energy (also angle if correlated)
        # ==============================================================================

        spectrum = simulation["distributions"][inelastic_scattering["spectrum_ID"]]

        # Sample energy
        if mu_distribution["sub_type"] != DISTRIBUTION_NONE:
            E_new = sample_distribution_with_scale(
                E, spectrum, particle_container_new, simulation, data
            )
        else:
            E_new, mu = sample_correlated_distribution_with_scale(
                E,
                spectrum,
                particle_container_new,
                simulation,
                data,
                nuclide["atomic_weight_ratio"],
                reaction["q_value"],
            )

        # ==============================================================================
        # Frame transformation
        # ==============================================================================

        reference_frame = reaction["reference_frame"]
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

        # Subtract outgoing energy from energy deposition
        interaction_data["energy_deposition"] -= particle_new["E"] * particle_new["w"]

        # ==============================================================================
        # Bank the new particle
        # ==============================================================================

        # Census takes precedence over retaining the last product.
        if particle["event"] & EVENT_TIME_CENSUS:
            particle_bank_module.bank_census_particle(particle_container_new, program)
        elif n == N - 1:
            particle["alive"] = True
            particle["ux"] = particle_new["ux"]
            particle["uy"] = particle_new["uy"]
            particle["uz"] = particle_new["uz"]
            particle["E"] = particle_new["E"]
        else:
            particle_bank_module.bank_active_particle(particle_container_new, program)


# ======================================================================================
# Fission
# ======================================================================================


@njit
def sample_fission(
    reaction, particle_container, interaction_data_container, nuclide, program, data
):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    sub_ID = reaction["sub_ID"]
    fission = simulation["neutron_fission_reactions"][sub_ID]

    settings = simulation["settings"]

    # Particle properties
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Kill the current particle
    particle["alive"] = False

    # Energy deposition
    #   TODO: Use energy-dependent Q-value
    q_value = reaction["q_value"]
    interaction_data["energy_deposition"] += (E + q_value) * particle["w"]

    # Adjust production and product weights if weighted emission
    weight_production = 1.0
    weight_product = particle["w"]
    if simulation["technique"]["weighted_emission"]["active"]:
        weight_target = simulation["technique"]["weighted_emission"]["weight_target"]
        weight_production = particle["w"] / weight_target
        weight_product = weight_target

    # Fission yields
    N_delayed = fission["N_delayed_precursor"]
    nu_p = neutron_fission_prompt_multiplicity(E, fission, simulation, data)
    nu_d = neutron_fission_delayed_multiplicity(E, fission, simulation, data)
    nu = nu_p + nu_d

    # Get number of secondaries
    N = int(
        math.floor(
            weight_production * nu / simulation["k_eff"] + rng.lcg(particle_container)
        )
    )

    # Set up secondary partice container
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_product

        # Prompt or delayed?
        prompt = True
        delayed_group = N_delayed - 1
        xi = rng.lcg(particle_container_new)
        total = nu_p / nu
        if xi > total:
            prompt = False
            # Determine delayed group
            for j in range(N_delayed):
                fraction = mcdc_get.neutron_fission_reaction.delayed_fractions(
                    j, fission, data
                )
                total += fraction * nu_d / nu
                if xi < total:
                    delayed_group = j
                    break

        # ==============================================================================
        # Sample prompt neutron
        # ==============================================================================

        if prompt:
            # Sample angle (if not energy-correlated)
            mu_distribution = simulation["distributions"][fission["mu_ID"]]
            if mu_distribution["sub_type"] != DISTRIBUTION_NONE:
                mu = sample_distribution(
                    E, mu_distribution, particle_container_new, simulation, data
                )

            # Sample energy (also angle if correlated)
            spectrum = simulation["distributions"][fission["spectrum_ID"]]
            if mu_distribution["sub_type"] != DISTRIBUTION_NONE:
                E_new = sample_distribution_with_scale(
                    E,
                    spectrum,
                    particle_container_new,
                    simulation,
                    data,
                )
            else:
                E_new, mu = sample_correlated_distribution_with_scale(
                    E,
                    spectrum,
                    particle_container_new,
                    simulation,
                    data,
                    nuclide["atomic_weight_ratio"],
                    reaction["q_value"],
                )

            # Frame transformation
            reference_frame = reaction["reference_frame"]
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

            particle_new["ux"] = ux_new
            particle_new["uy"] = uy_new
            particle_new["uz"] = uz_new
            spectrum_ID = mcdc_get.neutron_fission_reaction.delayed_spectrum_IDs(
                delayed_group, fission, data
            )
            spectrum = simulation["distributions"][spectrum_ID]
            particle_new["E"] = sample_distribution_with_scale(
                E, spectrum, particle_container_new, simulation, data
            )

            # Sample emission time
            decay_rate = mcdc_get.neutron_fission_reaction.delayed_decay_rates(
                delayed_group, fission, data
            )
            if not prompt:
                xi = rng.lcg(particle_container_new)
                particle_new["t"] -= math.log(xi) / decay_rate

        # Subtract outgoing energy from energy deposition
        interaction_data["energy_deposition"] -= particle_new["E"] * particle_new["w"]

        # ==============================================================================
        # Bank the new particle
        # ==============================================================================

        # Eigenvalue mode: bank right away
        if settings["neutron_eigenvalue_mode"]:
            particle_bank_module.bank_census_particle(particle_container_new, program)
            continue
        # Below is only relevant for fixed-source problem

        # Prompt-neutron banking
        if prompt:
            # To census bank
            if particle["event"] & EVENT_TIME_CENSUS:
                particle_bank_module.bank_census_particle(
                    particle_container_new, program
                )

            # To active bank
            elif n < N - 1:
                particle_bank_module.bank_active_particle(
                    particle_container_new, program
                )

            # Replace current particle
            else:
                particle["alive"] = True
                particle["ux"] = particle_new["ux"]
                particle["uy"] = particle_new["uy"]
                particle["uz"] = particle_new["uz"]
                particle["t"] = particle_new["t"]
                particle["E"] = particle_new["E"]
                particle["w"] = particle_new["w"]

        # Delayed-neutron banking
        else:
            # Skip if beyond time boundary
            if (
                particle_new["t"]
                > settings["time_boundary"] - COINCIDENCE_TOLERANCE_TIME
            ):
                continue

            # Check if go to census or future bank
            hit_current_census = False
            hit_future_census = False
            idx_census = simulation["idx_census"]
            if settings["N_census"] > 1:
                if particle_new["t"] > (
                    mcdc_get.settings.census_time(idx_census, settings, data)
                    - COINCIDENCE_TOLERANCE_TIME
                ):
                    hit_current_census = True
                    if particle_new["t"] > (
                        mcdc_get.settings.census_time(idx_census + 1, settings, data)
                        - COINCIDENCE_TOLERANCE_TIME
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
                    particle["E"] = particle_new["E"]
                    particle["w"] = particle_new["w"]
                else:
                    particle_bank_module.bank_active_particle(
                        particle_container_new, program
                    )

            # Hit future census --> add to future bank
            elif hit_future_census:
                # Particle will participate in the future
                particle_bank_module.bank_future_particle(
                    particle_container_new, program
                )

            # Hit current census --> add to census bank
            else:
                # Particle will participate after the current census is completed
                particle_bank_module.bank_census_particle(
                    particle_container_new, program
                )


@njit
def neutron_fission_prompt_multiplicity(E, fission, simulation, data):
    reaction_data = simulation["data"][fission["prompt_multiplicity_ID"]]
    return evaluate_data_clamped(E, reaction_data, simulation, data)


@njit
def neutron_fission_delayed_multiplicity(E, fission, simulation, data):
    reaction_data = simulation["data"][fission["delayed_multiplicity_ID"]]
    return evaluate_data_clamped(E, reaction_data, simulation, data)
