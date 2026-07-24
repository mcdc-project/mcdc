import math
import numpy as np

from numba import njit

###

import mcdc.transport.rng as rng
import mcdc.transport.physics.electron as electron
import mcdc.transport.physics.neutron as neutron
import mcdc.transport.physics.proton as proton

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import *

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.particle_speed(particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        return electron.particle_speed(particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        return proton.particle_speed(particle_container, simulation, data)
    return -1.0


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.macro_xs(reaction_type, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        return electron.macro_xs(reaction_type, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        return proton.macro_xs(reaction_type, particle_container, simulation, data)
    return -1.0


@njit
def neutron_production_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        return neutron.neutron_production_xs(
            reaction_type, particle_container, simulation, data
        )
    return -1.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision_distance(particle_container, simulation, data):
    particle = particle_container[0]

    # Get total cross-section
    SigmaT = 0.0
    if particle["particle_type"] == PARTICLE_NEUTRON:
        SigmaT = macro_xs(NEUTRON_REACTION_TOTAL, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        SigmaT = macro_xs(ELECTRON_REACTION_TOTAL, particle_container, simulation, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        SigmaT = macro_xs(PROTON_REACTION_TOTAL, particle_container, simulation, data)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng.lcg(particle_container)
    distance = -math.log(xi) / SigmaT
    return distance


@njit
def collision(particle_container, collision_data_container, program, data):
    particle = particle_container[0]

    if particle["particle_type"] == PARTICLE_NEUTRON:
        neutron.collision(particle_container, collision_data_container, program, data)
    elif particle["particle_type"] == PARTICLE_ELECTRON:
        electron.collision(particle_container, collision_data_container, program, data)
    elif particle["particle_type"] == PARTICLE_PROTON:
        proton.collision(particle_container, collision_data_container, program, data)


# ======================================================================================
# Continuous Slowing Down Approximation
# ======================================================================================


@njit
def csda_distance(particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["native_materials"][particle["material_ID"]]
    E = particle["E"]
    total_rho = 0.0
    total_dedx = 0.0

    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = simulation["nuclides"][nuclide_ID]

        if not material["stopping_power_provided"]:
            dedx_values = mcdc_get.nuclide.stopping_power_all(nuclide, data)
            dedx_energies = mcdc_get.nuclide.stopping_power_energy_grid_all(nuclide, data)
            dedx = np.interp(E / 1e6, dedx_energies, dedx_values)
            total_dedx += dedx * 1e6            

        atomic_mass = nuclide["atomic_weight_ratio"]
        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        density_gcm3 = nuclide_density * 1e24 * atomic_mass / (6.022e23)
        total_rho += density_gcm3
    
    if material["stopping_power_provided"]:
        dedx_values = mcdc_get.native_material.stopping_power_all(material, data)
        dedx_energies = mcdc_get.native_material.stopping_power_energy_grid_all(material, data)
        dedx = np.interp(E / 1e6, dedx_energies, dedx_values)
        total_dedx = dedx * 1e6

    
    max_fractional_e_loss = simulation["settings"]["csda_max_fractional_e_loss"]
    return max_fractional_e_loss * E / total_dedx / total_rho


@njit
def csda_edep(particle_container, collision_data_container, distance, simulation, data):
    particle = particle_container[0]
    if particle["particle_type"] == PARTICLE_NEUTRON:
        raise ValueError("CSDA not supported for neutrons")
    if particle["particle_type"] == PARTICLE_ELECTRON:
        raise ValueError("CSDA not supported for electrons")
    if particle["particle_type"] == PARTICLE_PROTON:
        proton.csda_edep(
            particle_container, collision_data_container, distance, simulation, data
        )
