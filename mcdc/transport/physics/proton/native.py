import math
import numpy as np
from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    ANGLE_DISTRIBUTED,
    ANGLE_ENERGY_CORRELATED,
    ANGLE_ISOTROPIC,
    BOLTZMANN_K,
    THERMAL_THRESHOLD_FACTOR,
    LIGHT_SPEED,
    PROTON_MASS,
    PI,
    PI_HALF,
    PI_SQRT,
    PROTON_REACTION_TOTAL,
    PROTON_REACTION_ELASTIC_SCATTERING,
    PROTON_REACTION_NONELASTIC,
    REFERENCE_FRAME_COM,
    PARTICLE_ELECTRON,
    PARTICLE_NEUTRON,
    PARTICLE_PROTON,
    PROTON_CUTOFF_ENERGY,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_correlated_distribution_with_scale,
    sample_distribution_with_scale,
    sample_isotropic_cosine,
    sample_isotropic_direction,
    sample_multi_table,
    sample_kalbach_mann,
)
from mcdc.transport.physics.util import (
    evaluate_proton_xs_energy_grid,
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
    mass = PROTON_MASS
    return LIGHT_SPEED * math.sqrt(E * (E + 2.0 * mass)) / (E + mass)


@njit
def particle_energy_from_speed(speed):
    beta = speed / LIGHT_SPEED
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    mass = PROTON_MASS
    return mass * (gamma - 1.0)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["native_materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0

    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = simulation["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, nuclide, data)

        total += nuclide_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, nuclide, data):
    idx, E0, E1 = evaluate_proton_xs_energy_grid(E, nuclide, data)
    if reaction_type == PROTON_REACTION_TOTAL:
        xs0 = mcdc_get.nuclide.proton_total_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.proton_total_xs(idx + 1, nuclide, data)
    elif reaction_type == PROTON_REACTION_ELASTIC_SCATTERING:
        xs0 = mcdc_get.nuclide.proton_elastic_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.proton_elastic_xs(idx + 1, nuclide, data)
    elif reaction_type == PROTON_REACTION_NONELASTIC:
        xs0 = mcdc_get.nuclide.proton_nonelastic_xs(idx, nuclide, data)
        xs1 = mcdc_get.nuclide.proton_nonelastic_xs(idx + 1, nuclide, data)
    else:
        # Should be unreachable
        xs0 = 0.0
        xs1 = 0.0
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction_base, nuclide, data):
    idx, E0, E1 = evaluate_proton_xs_energy_grid(E, nuclide, data)

    # Apply offset
    offset = reaction_base["xs_offset_"]
    if idx < offset:
        return 0.0
    else:
        idx -= offset

    xs0 = mcdc_get.proton_reaction.xs(idx, reaction_base, data)
    xs1 = mcdc_get.proton_reaction.xs(idx + 1, reaction_base, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    collision_data = collision_data_container[0]
    material = simulation["native_materials"][particle["material_ID"]]

    # Particle properties
    E = particle["E"]

    # Check for cutoff energy
    if E <= PROTON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # ==================================================================================
    # Sample colliding nuclide
    # ==================================================================================

    SigmaT = macro_xs(PROTON_REACTION_TOTAL, particle_container, simulation, data)

    # No implicit capture for protons (there's no capture xs)

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = simulation["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        sigmaT = total_micro_xs(PROTON_REACTION_TOTAL, E, nuclide, data)

        SigmaT_nuclide = nuclide_density * sigmaT
        total += SigmaT_nuclide

        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    sigma_elastic = total_micro_xs(PROTON_REACTION_ELASTIC_SCATTERING, E, nuclide, data)
    sigma_nonelastic = total_micro_xs(PROTON_REACTION_NONELASTIC, E, nuclide, data)
    xi = rng.lcg(particle_container) * sigmaT

    # Elastic scattering
    total = sigma_elastic
    if xi < total:
        # Sample the actual reaction from the group
        total -= sigma_elastic
        for i in range(nuclide["N_proton_elastic_scattering_reaction"]):
            reaction_ID = int(
                mcdc_get.nuclide.proton_elastic_scattering_reaction_IDs(
                    i, nuclide, data
                )
            )
            reaction = simulation["proton_elastic_scattering_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = simulation["proton_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, nuclide, data)

            # Execute the reaction
            if xi < total:
                elastic_scattering(
                    reaction,
                    particle_container,
                    collision_data_container,
                    nuclide,
                    simulation,
                    data,
                )
                return

    # Noelastic reaction
    total += sigma_nonelastic
    if xi < total:
        # Sample the actual reaction from the group
        total -= sigma_nonelastic
        for i in range(nuclide["N_proton_nonelastic_reaction"]):
            reaction_ID = int(
                mcdc_get.nuclide.proton_nonelastic_reaction_IDs(i, nuclide, data)
            )
            reaction = simulation["proton_nonelastic_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = simulation["proton_reactions"][reaction_base_ID]
            xs = reaction_micro_xs(E, reaction_base, nuclide, data)
            total += xs

            # Execute the reaction
            if xi < total:
                nonelastic_reaction(
                    reaction,
                    particle_container,
                    collision_data_container,
                    nuclide,
                    program,
                    data,
                )
                return


# ======================================================================================
# Continous Slowing Down Approximation
# ======================================================================================


@njit
def csda_edep(particle_container, collision_data_container, distance, simulation, data):
    particle = particle_container[0]
    collision_data = collision_data_container[0]
    material = simulation["native_materials"][particle["material_ID"]]
    # print(f'material = {material}, type = {type(material)}, {material.dtype.names}')
    E = particle["E"]

    # Check for cutoff energy
    if E <= PROTON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return
 
    total_stopping_power = 0.0
    total_rho_gcm3 = 0.0
    total_Z = 0.0
    total_A = 0.0
    # Find the total stopping power by summing over every nuclide in the material
    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = simulation["nuclides"][nuclide_ID]

        # If no stopping power provided, we calculate it ourselves here
        if not material["stopping_power_provided"]:
            dedx_values = mcdc_get.nuclide.stopping_power_all(nuclide, data)
            dedx_energies = mcdc_get.nuclide.stopping_power_energy_grid_all(nuclide, data)

            # TODO: replace np.interp with a non-numpy function??
            dedx = np.interp(E / 1e6, dedx_energies, dedx_values)
            total_stopping_power += dedx * 1e6

        # Convert atoms/barn-cm to g/cm3:
        atomic_mass = nuclide["atomic_weight_ratio"]  # mass in amu
        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        density_gcm3 = nuclide_density * 1e24 * atomic_mass / (6.022e23)
        total_rho_gcm3 += density_gcm3

        total_Z += nuclide["atomic_number"]
        total_A += nuclide["mass_number"]
    
    Z = total_Z / material["N_nuclide"]
    A = total_A / material["N_nuclide"]

    if material["stopping_power_provided"]:
        dedx_values = mcdc_get.native_material.stopping_power_all(material, data)
        dedx_energies = mcdc_get.native_material.stopping_power_energy_grid_all(material, data)

        dedx = np.interp(E / 1e6, dedx_energies, dedx_values)
        total_stopping_power = dedx * 1e6

    energy_loss = total_stopping_power * total_rho_gcm3 * distance

    # Range straggling - modify energy loss to have some slight variations
    # TODO: Insert different thickness regimes to sample from (e.g. Bohr, Landau, Vavilov)
    gaussian_variance = 0.1569 * total_rho_gcm3 * Z / A * distance
    
    gaussian = np.random.normal(loc=0.0, scale=np.sqrt(gaussian_variance))
    energy_loss += gaussian
    particle["E"] -= energy_loss
    collision_data["energy_deposition"] += energy_loss * particle["w"]

    if energy_loss * particle["w"] <= 0.0:
        print(f'total density = {total_rho_gcm3}')
        print(f'stopping_power = {total_stopping_power}')
        print(f'distance = {distance}')
        print(f'NEGATIVE: energy_loss = {energy_loss * particle["w"]}')
        raise ValueError('negative energy loss')
    return


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def elastic_scattering(
    reaction, particle_container, collision_data_container, nuclide, simulation, data
):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Particle attributes
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Energy deposition
    collision_data["energy_deposition"] += E * particle["w"]
    # print(f'\ndeposited {E * particle["w"]} eV at x={particle["x"]} from elastic scattering')

    # Note: Q-value is zero in elastic scattering

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

    # Proton velocity - LAB
    vx = speed * ux
    vy = speed * uy
    vz = speed * uz

    # COM velocity
    COM_x = (vx + A * Vx) / (1.0 + A)
    COM_y = (vy + A * Vy) / (1.0 + A)
    COM_z = (vz + A * Vz) / (1.0 + A)

    # Proton velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Proton speed - COM
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Proton initial direction - COM
    ux = vx / speed
    uy = vy / speed
    uz = vz / speed

    # Sample the scattering cosine from the multi-PDF distribution
    mu_table_ID = reaction["mu_table_ID"]
    if mu_table_ID >= len(simulation["multi_table_distributions"]):
        mu_table_ID = 0  # Fallback to first distribution
    multi_table = simulation["multi_table_distributions"][mu_table_ID]

    # multi_table = simulation["multi_table_distributions"][reaction["mu_table_ID"]]
    mu0 = sample_multi_table(E, particle_container, multi_table, data)

    # Scatter the direction in COM
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Proton final velocity - COM
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
    collision_data["energy_deposition"] -= particle["E"] * particle["w"]


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
    #   nuclide-proton polar cosine candidate mu_tilda via
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
# Nonelastic scattering
# ======================================================================================


@njit
def nonelastic_reaction(
    reaction, particle_container, collision_data_container, nuclide, program, data
):
    """
    Proton nonelastic scattering with secondary particle production.

    Samples:
    1. Outgoing proton from proton_reactions/inelastic/MT-005
    2. Secondary particles from secondary_particles/ZAP_x/MT-005
    """
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    reaction_base_ID = reaction["parent_ID"]
    reaction_base = simulation["proton_reactions"][reaction_base_ID]

    # Particle attributes
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    w = particle["w"]

    # Kill the incident proton
    particle["alive"] = False

    # Q-value energy available
    q_value = reaction_base["q_value"] * 1e6
    total_energy = E + q_value

    # ===========================================================================
    # 1. Sample outgoing PROTON
    # ===========================================================================

    # Number of outgoing protons and spectra
    N_proton = reaction["multiplicity"]
    N_spectrum = reaction["N_spectrum"]
    use_all_spectrum = N_proton == N_spectrum

    # Set up secondary particle container
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_new = particle_container_new[0]

    # Energy deposition (will be adjusted as we create secondaries)
    collision_data["energy_deposition"] += total_energy * w
    # print(f'\ndeposited {total_energy * particle["w"]} eV at x={particle["x"]} from nonelastic_rxn')

    # Create outgoing protons
    for n in range(N_proton):
        # Set default attributes (copy incident proton)
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
            distribution_base = simulation["distributions"][reaction["mu_ID"]]
            multi_table = simulation["multi_table_distributions"][
                distribution_base["child_ID"]
            ]
            mu = sample_multi_table(E, particle_container_new, multi_table, data)

        # ==============================================================================
        # Sample energy (also angle if correlated)
        # ==============================================================================

        # Get energy spectrum
        if use_all_spectrum:
            ID = int(
                mcdc_get.proton_nonelastic_reaction.energy_spectrum_IDs(
                    n, reaction, data
                )
            )
            spectrum_base = simulation["distributions"][ID]
        else:
            offset = reaction["spectrum_probability_grid_offset"]
            length = reaction["spectrum_probability_grid_length"]
            probability_grid = data[offset : offset + length]
            probability_idx = find_bin(E, probability_grid)
            xi = rng.lcg(particle_container_new)
            total = 0.0
            for j in range(N_spectrum):
                probability = mcdc_get.proton_nonelastic_reaction.spectrum_probability(
                    probability_idx, j, reaction, data
                )
                total += probability
                if xi < total:
                    ID = int(
                        mcdc_get.proton_nonelastic_reaction.energy_spectrum_IDs(
                            j, reaction, data
                        )
                    )
                    spectrum_base = simulation["distributions"][ID]
                    break

        # Sample energy
        if not angle_type == ANGLE_ENERGY_CORRELATED:
            E_new = sample_distribution_with_scale(
                E, spectrum_base, particle_container_new, simulation, data
            )
        else:
            E_new, mu = sample_correlated_distribution_with_scale(
                E, spectrum_base, particle_container_new, simulation, data
            )

        # ==============================================================================
        # Frame transformation
        # ==============================================================================

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
        particle_new["particle_type"] = PARTICLE_PROTON

        # Subtract outgoing energy from energy deposition
        collision_data["energy_deposition"] -= particle_new["E"] * particle_new["w"]

        # ==============================================================================
        # Bank the new particle
        # ==============================================================================

        # Keep it if it is the last particle
        if n == N_proton - 1:
            particle["alive"] = True
            particle["ux"] = particle_new["ux"]
            particle["uy"] = particle_new["uy"]
            particle["uz"] = particle_new["uz"]
            particle["E"] = particle_new["E"]
            particle["particle_type"] = PARTICLE_PROTON
        else:
            particle_bank_module.bank_active_particle(particle_container_new, program)

    # ===========================================================================
    # 2. Sample SECONDARY PARTICLES from secondary_particles groups
    # ===========================================================================

    # Get secondary channels for this MT (if any)
    # MT = int(reaction_base["MT"])
    # nuclide_ID = particle["nuclide_ID"]

    # Check if nuclide has secondary particle data
    # (This requires access to nuclide secondary_channels dict, which needs to be added)
    # For now, we'll skip this part and it can be added when the data structure supports it
    # TODO: Add secondary particle sampling when nuclide.proton_secondary_channels is accessible


# No fission for protons
