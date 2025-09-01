import math

from numba import njit

####

import mcdc.adapt as adapt
import mcdc.data_sampler as data_sampler
import mcdc.kernel as kernel
import mcdc.physics.native as physics

from mcdc.constant import E_THERMAL_THRESHOLD, PI, PI_HALF, PI_SQRT, SQRD_SPEED_TO_E
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

    # ==================================================================================
    # Sample scattering cosine
    # ==================================================================================

    # Sample the scattering cosine from the multi-PDF distribution
    data_idx = reaction["mu_index"]
    multipdf = mcdc["data_multipdfs"][data_idx]
    xi1 = kernel.rng(particle_container)
    xi2 = kernel.rng(particle_container)
    mu0 = data_sampler.sample_multipdf(E, xi1, xi2, multipdf, data)

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
    particle['E'] = SQRD_SPEED_TO_E * particle_speed * particle_speed

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
            particle_speed * particle_speed + V_tilda * V_tilda - 2.0 * particle_speed * V_tilda * mu_tilda
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
def fission(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)

    # Kill the current particle
    particle["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = particle["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = particle["w"]

    # Sample nuclide if CE
    material = mcdc["materials"][particle["material_ID"]]

    # Get number of secondaries
    if mcdc["settings"]["multigroup_mode"]:
        g = particle["g"]
        nu = mcdc_get.material.mgxs_nu_f(g, material, data)
    else:
        nuclide = sample_nuclide(material, particle_container, XS_FISSION, mcdc)
        E = particle["E"]
        nu = get_nu(NU_FISSION, nuclide, E)
    N = int(math.floor(weight_eff * nu / mcdc["k_eff"] + rng(particle_container)))

    particle_container_new = adapt.local_array(1, type_.particle_record)
    particle_new = particle_container_new[0]

    for n in range(N):
        # Create new particle
        split_as_record(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_new

        # Sample fission neutron phase space
        if mcdc["settings"]["multigroup_mode"]:
            sample_phasespace_fission(particle_container, material, particle_container_new, mcdc, data)
        else:
            sample_phasespace_fission_nuclide(particle_container, nuclide, particle_container_new, mcdc)

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
            if particle["t"] > mcdc_get.settings.census_time(idx_census + 1, settings, data):
                hit_census = True
                hit_next_census = True
            elif particle_new["t"] > mcdc_get.settings.census_time(idx_census, settings, data):
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
