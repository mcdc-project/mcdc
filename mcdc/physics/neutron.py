import math
import numpy as np

from numba import njit

####

import mcdc.adapt as adapt
import mcdc.data_processor as data_processor
import mcdc.kernel as kernel
import mcdc.mcdc_get as mcdc_get
import mcdc.physics.native as physics
import mcdc.type_ as type_

from mcdc.constant import DATA_MAXWELLIAN, DATA_MULTIPDF, DATA_POLYNOMIAL, DATA_TABLE, E_THERMAL_THRESHOLD, PI, PI_HALF, PI_SQRT, SQRD_SPEED_TO_E
from mcdc.physics.util import scatter_direction, sample_isotropic_direction


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def elastic_scattering(particle_container, nuclide, reaction, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Sample nucleus thermal velocity
    A = nuclide["atomic_weight_ratio"]
    if E > E_THERMAL_THRESHOLD:
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
    else:
        Vx, Vy, Vz = sample_nucleus_velocity(A, particle_container, mcdc, data)

    # =========================================================================
    # COM kinematics
    # =========================================================================

    # Particle speed
    particle_speed = physics.particle_speed(particle_container)

    # Neutron velocity - LAB
    vx = particle_speed * ux
    vy = particle_speed * uy
    vz = particle_speed * uz

    # COM velocity
    COM_x = (vx + A * Vx) / (1.0 + A)
    COM_y = (vy + A * Vy) / (1.0 + A)
    COM_z = (vz + A * Vz) / (1.0 + A)

    # Neutron velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Neutron speed - COM
    particle_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Neutron initial direction - COM
    ux = vx / particle_speed
    uy = vy / particle_speed
    uz = vz / particle_speed

    # Sample the scattering cosine from the multi-PDF distribution
    index = reaction["mu_index"]
    mu0 = data_processor.sample_distribution(E, DATA_MULTIPDF, index, particle_container, mcdc, data)

    # Scatter the direction in COM
    azi = 2.0 * PI * kernel.rng(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Neutron final velocity - COM
    vx = particle_speed * ux_new
    vy = particle_speed * uy_new
    vz = particle_speed * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx = vx + COM_x
    vy = vy + COM_y
    vz = vz + COM_z

    # Final energy - LAB
    particle_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    particle["E"] = SQRD_SPEED_TO_E * particle_speed * particle_speed

    # Final direction - LAB
    particle["ux"] = vx / particle_speed
    particle["uy"] = vy / particle_speed
    particle["uz"] = vz / particle_speed


@njit
def sample_nucleus_velocity(A, particle_container, mcdc, data):
    particle = particle_container[0]

    # Particle speed
    particle_speed = physics.particle_speed(particle_container)

    # Maxwellian parameter
    beta = math.sqrt(2.0659834e-11 * A)
    # The constant above is
    #   (1.674927471e-27 kg) / (1.38064852e-19 cm^2 kg s^-2 K^-1) / (293.6 K)/2

    # Sample nuclide speed candidate V_tilda and
    #   nuclide-neutron polar cosine candidate mu_tilda via
    #   rejection sampling
    y = beta * particle_speed
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
            particle_speed * particle_speed
            + V_tilda * V_tilda
            - 2.0 * particle_speed * V_tilda * mu_tilda
        ) / (particle_speed + V_tilda):
            break

    # Set nuclide velocity - LAB
    azi = 2.0 * PI * kernel.rng(particle_container)
    ux, uy, uz = scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu_tilda, azi
    )
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz


# ======================================================================================
# Fission
# ======================================================================================


@njit
def fission(particle_container, nuclide, reaction, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    E = particle['E']

    # Nuclide properties
    N_delayed = reaction['N_delayed']

    # Kill the current particle
    particle["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = particle["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = particle["w"]

    # Fission yields
    nu_p = fission_yield_prompt(E, reaction, mcdc, data)
    nu_d = np.zeros(N_delayed)
    nu_d_total = 0.0
    for j in range(N_delayed):
        nu_d[j] = fission_yield_delayed(E, j, reaction, mcdc, data)
        nu_d_total += nu_d[j]
    nu = nu_p + nu_d_total

    # Number of fission neutrons
    N = int(math.floor(weight_eff * nu / mcdc["k_eff"] + kernel.rng(particle_container)))

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
        prompt = True
        delayed_group = -1
        xi = kernel.rng(particle_container_new) * nu
        total = nu_p
        if xi > total:
            prompt = False
            # Determine delayed group
            for j in range(N_delayed):
                total += nu_d[j]
                if xi < total:
                    delayed_group = j
                    break

        # Sample outgoing energy
        if prompt:
            particle_new['E'] = sample_fission_spectrum_prompt(E, reaction, particle_container_new, mcdc, data)
        else:
            particle_new['E'] = sample_fission_spectrum_delayed(E, delayed_group, reaction, particle_container_new, mcdc, data)
        
        # Sample emission time
        decay = mcdc_get.neutron_fission.delayed_decay_rates(delayed_group, reaction, data)
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
   

@njit 
def fission_yield_prompt(E, reaction, mcdc, data):
    data_type = reaction["prompt_yield_type"]
    index = reaction["prompt_yield_index"]
    return data_processor.evaluate_data(E, data_type, index, mcdc, data)


@njit 
def fission_yield_delayed(E, group, reaction, mcdc, data):
    data_type = int(mcdc_get.neutron_fission.delayed_yield_type(group, reaction, data))
    index = int(mcdc_get.neutron_fission.delayed_yield_index(group, reaction, data))
    return data_processor.evaluate_data(E, data_type, index, mcdc, data)


@njit 
def sample_fission_spectrum_prompt(E, reaction, rng_state, mcdc, data):
    data_type = int(reaction["prompt_spectrum_type"])
    index = int(reaction["prompt_spectrum_index"])
    return data_processor.sample_distribution(E, data_type, index, rng_state, mcdc, data, True)


@njit 
def sample_fission_spectrum_delayed(E, group, reaction, rng_state, mcdc, data):
    data_type = int(mcdc_get.neutron_fission.delayed_spectrum_type(group, reaction, data))
    index = int(mcdc_get.neutron_fission.delayed_spectrum_index(group, reaction, data))
    return data_processor.sample_distribution(E, data_type, index, rng_state, mcdc, data, True)
