import numpy as np
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
    PI,
    PROTON_REACTION_TOTAL,
    PROTON_REACTION_ELASTIC_SCATTERING,
    PROTON_REACTION_INELASTIC_SCATTERING,
    PROTON_REACTION_CAPTURE,
)
from mcdc.transport.physics.util import scatter_direction
from mcdc.transport.distribution import sample_isotropic_direction

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["multigroup_materials"][particle["material_ID"]]
    return mcdc_get.multigroup_material.mgxs_speed(particle["g"], material, data)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["multigroup_materials"][particle["material_ID"]]
    g = particle["g"]

    if reaction_type == PROTON_REACTION_TOTAL:
        return mcdc_get.multigroup_material.mgxs_total(g, material, data)
    elif reaction_type == PROTON_REACTION_ELASTIC_SCATTERING:
        return mcdc_get.multigroup_material.mgxs_scatter(g, material, data)
    return 0.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]

    # Get the reaction cross-sections
    SigmaT = macro_xs(PROTON_REACTION_TOTAL, particle_container, simulation, data)
    SigmaS = macro_xs(
        PROTON_REACTION_ELASTIC_SCATTERING, particle_container, simulation, data
    )
    SigmaC = macro_xs(PROTON_REACTION_CAPTURE, particle_container, simulation, data)

    # Implicit capture
    if simulation["implicit_capture"]["active"]:
        particle["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    # Sample reaction type and perform the reaction
    xi = rng.lcg(particle_container) * SigmaT
    total = SigmaS
    if total > xi:
        scattering(particle_container, program, data)
    else:
        particle["alive"] = False


# ======================================================================================
# Reactions
# ======================================================================================


@njit
def scattering(particle_container, program, data):
    simulation = util.access_simulation(program)

    # Particle attributes
    particle = particle_container[0]
    g = particle["g"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Material attributes
    material = simulation["multigroup_materials"][particle["material_ID"]]
    G = material["G"]

    # Kill the current particle
    particle["alive"] = False

    # Adjust production and product weights if weighted emission
    weight_production = 1.0
    weight_product = particle["w"]
    if simulation["weighted_emission"]["active"]:
        weight_target = simulation["weighted_emission"]["weight_target"]
        weight_production = particle["w"] / weight_target
        weight_product = weight_target


    # TODO: make this better for protons, add secondary particle generation to non-MG materials
    # Get number of secondaries
    nu_s = mcdc_get.multigroup_material.mgxs_nu_s(g, material, data)
    N = int(math.floor(weight_production * nu_s + rng.lcg(particle_container)))

    # Set up secondary partice container
    particle_container_new = util.local_array(1, type_.particle_data)
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
        stride = material["G"]
        start = material["mgxs_chi_s_offset"] + g * stride
        chi_s = data[start : start + stride]
        # Above is equivalent to: chi_s = mcdc_get.multigroup_material.mgxs_chi_s_vector(g, material, data)

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
            particle_bank_module.bank_active_particle(particle_container_new, program)
