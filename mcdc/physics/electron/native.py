import math
import numpy as np

from numba import njit

####

from mcdc import adapt
from mcdc import data_processor
from mcdc import kernel
from mcdc import mcdc_get
from mcdc import type_

from mcdc.constant import (DATA_MULTIPDF,
                           DATA_TABLE,
                           PI,
                           ELECTRON_REST_MASS_ENERGY,
                           SPEED_OF_LIGHT,
                           EEDL_THRESHOLD,
                           TINY
                           )
from mcdc.physics.util import scatter_direction


from mcdc.constant import (
    REACTION_TOTAL,
    REACTION_ELECTRON_ELASTIC_SCATTERING,
    REACTION_ELECTRON_ELASTIC_SMALL_ANGLE,
    REACTION_ELECTRON_ELASTIC_LARGE_ANGLE,
    REACTION_ELECTRON_EXCITATION,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_IONIZATION
)
from mcdc.physics.util import evaluate_xs_energy_grid
from mcdc.util import linear_interpolation

# ======================================================================================
# Particle attributes
# ======================================================================================

@njit
def particle_speed(particle_container):
    E = particle_container[0]["E"]

    gamma = 1.0 + E / ELECTRON_REST_MASS_ENERGY
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    return math.sqrt(beta_sq) * SPEED_OF_LIGHT


# ======================================================================================
# Material properties
# ======================================================================================

@njit
def macro_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    E = particle["E"]

    # Sum over all elements
    total = 0.0
    for i in range(material["N_element"]):
        element = mcdc_get.element.from_material(i, material, mcdc, data)
        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
        xs = micro_xs_element(E, reaction_type, element, mcdc, data)
        total += nuclide_density * xs
    return total


@njit
def micro_xs_element(E, reaction_type, element, mcdc, data):
    # Total reaction
    if reaction_type == REACTION_TOTAL:
        idx, E0, E1 = evaluate_xs_energy_grid(E, element, data)
        xs0 = mcdc_get.element.total_xs(idx, element, data)
        xs1 = mcdc_get.element.total_xs(idx + 1, element, data)
        return linear_interpolation(E, E0, E1, xs0, xs1)

    # Search if the reaction exists
    for i in range(element["N_reaction"]):
        the_type = int(mcdc_get.element.reaction_type(i, element, data))

        if the_type == reaction_type:
            # Reaction exists!
            reaction_idx = int(mcdc_get.element.reaction_index(i, element, data))
            idx, E0, E1 = evaluate_xs_energy_grid(E, element, data)

            if reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
                reaction = mcdc["electron_bremsstrahlung_reactios"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            if reaction_type == REACTION_ELECTRON_EXCITATION:
                reaction = mcdc["electron_excitation_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            if reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
                reaction = mcdc["electron_elastic_scattering_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

            if reaction_type == REACTION_ELECTRON_IONIZATION:
                reaction = mcdc["electron_ionization_reactions"][reaction_idx]
                xs0 = mcdc_get.reaction.xs(idx, reaction, data)
                xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
                return linear_interpolation(E, E0, E1, xs0, xs1)

    return 0.0


@njit
def electron_production_xs(reaction_type, material, particle_container, mcdc, data):
    particle = particle_container[0]
    E = particle["E"]

    if E < EEDL_THRESHOLD:
        return 0.0

    if reaction_type in (REACTION_ELECTRON_ELASTIC_SMALL_ANGLE, 
                         REACTION_ELECTRON_ELASTIC_LARGE_ANGLE):
        reaction_type = REACTION_ELECTRON_ELASTIC_SCATTERING

    if reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
        return macro_xs(reaction_type, material, particle_container, mcdc, data)

    if reaction_type == REACTION_ELECTRON_EXCITATION:
        return macro_xs(reaction_type, material, particle_container, mcdc, data)

    if reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
        return macro_xs(reaction_type, material, particle_container, mcdc, data)

    if reaction_type == REACTION_ELECTRON_IONIZATION:
        return 2.0 * macro_xs(reaction_type, material, particle_container, mcdc, data)

    if reaction_type == REACTION_TOTAL:
        total = 0.0
        total += electron_production_xs(REACTION_ELECTRON_ELASTIC_SCATTERING, 
                                        material, particle_container, mcdc, data)
        total += electron_production_xs(REACTION_ELECTRON_EXCITATION, 
                                        material, particle_container, mcdc, data)
        total += electron_production_xs(REACTION_ELECTRON_BREMSSTRAHLUNG, 
                                        material, particle_container, mcdc, data)
        total += electron_production_xs(REACTION_ELECTRON_IONIZATION, 
                                        material, particle_container, mcdc, data)
        return total


# ======================================================================================
# Collision
# ======================================================================================

@njit
def collision(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)
    material = mcdc["materials"][particle["material_ID"]]

    # ==================================================================================
    # Sample colliding element
    # ==================================================================================

    SigmaT = macro_xs_element(REACTION_TOTAL, material, particle_container, mcdc, data)
    xi = kernel.rng(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_element"]):
        element = mcdc_get.element.from_material(i, material, mcdc, data)
        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
        sigmaT = micro_xs_element(particle["E"], REACTION_TOTAL, element, mcdc, data)
        SigmaT_element = nuclide_density * sigmaT
        total += SigmaT_element
        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    xi = kernel.rng(particle_container) * sigmaT
    total = 0.0
    for i in range(element["N_reaction"]):
        reaction_type = int(mcdc_get.element.reaction_type(i, element, data))
        reaction_xs = micro_xs_element(particle["E"], reaction_type, element, mcdc, data)
        total += reaction_xs
        if total < xi:
            continue

        # Reaction is sampled
        reaction_idx = int(mcdc_get.element.reaction_index(i, element, data))
        if reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
            reaction = mcdc["electron_bremsstrahlung_reaction"][reaction_idx]
            bremsstrahlung(particle_container, element, reaction, prog, data)

        elif reaction_type == REACTION_ELECTRON_EXCITATION:
            reaction = mcdc["electron_excitation_reactions"][reaction_idx]
            excitation(particle_container, element, reaction, prog, data
                       )
        elif reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
            reaction = mcdc["electron_elastic_scattering_reactions"][reaction_idx]
            elastic_scattering(particle_container, element, reaction, prog, data)

        elif reaction_type == REACTION_ELECTRON_IONIZATION:
            reaction = mcdc["electron_ionization_reactions"][reaction_idx]
            ionization(particle_container, element, reaction, prog, data)


# ======================================================================================
# Elastic scattering
# ======================================================================================
'''
@njit
def elastic_scattering(particle_container, element, reaction, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]
    E0 = ELECTRON_REST_MASS_ENERGY  # eV
    E_e = E0 + E  # Total energy eV
    p_c = math.sqrt(E_e*E_e - E0*E0)  # p_e * c eV
    A = element["atomic_weight_ratio"]
    M_c2 = A * ELECTRON_REST_MASS_ENERGY

    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Branch XS
    xs_L = data_processor.evaluate_data(
        E, DATA_TABLE, int(reaction["large_angle_xs_index"]), mcdc, data
    )
    xs_S = data_processor.evaluate_data(
        E, DATA_TABLE, int(reaction["small_angle_xs_index"]), mcdc, data
    )

    xs_tot = xs_L + xs_S
    use_large = True
    if xs_tot > 0.0:
        if kernel.rng(particle_container) > xs_L / xs_tot:
            use_large = False

    if E <= EEDL_THRESHOLD:
        raise ValueError("Electron elastic scattering below 100 eV is not supported.")

    # =========================================================================
    # COM kinematics
    # =========================================================================

    # Particle speed
    particle_speed = electron_speed(particle_container)

    # Electron velocity - LAB
    vx = particle_speed * ux
    vy = particle_speed * uy
    vz = particle_speed * uz

    # COM velocity
    beta_COM = p_c / (E_e + M_c2)
    V_COM = beta_COM * SPEED_OF_LIGHT
    COM_x = V_COM * ux
    COM_y = V_COM * uy
    COM_z = V_COM * uz

    # Electron velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Electron speed - COM
    particle_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Electron initial direction - COM
    ux = vx / particle_speed
    uy = vy / particle_speed
    uz = vz / particle_speed

    # Sample the scattering cosine from the multi-PDF distribution
    index = int(reaction["mu_large_angle_index"]) if use_large else int(reaction["mu_small_angle_index"])
    mu0 = data_processor.sample_distribution(E, DATA_MULTIPDF, index, particle_container, mcdc, data)

    # Scatter the direction in COM
    azi = 2.0 * PI * kernel.rng(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Electron final velocity - COM
    vx = particle_speed * ux_new
    vy = particle_speed * uy_new
    vz = particle_speed * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx += COM_x
    vy += COM_y
    vz += COM_z

    # Final relativistic kinetic energy from speed - LAB
    particle_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    gamma_out = 1.0 / math.sqrt(1.0 - (particle_speed / SPEED_OF_LIGHT) ** 2)
    particle["E"] = (gamma_out - 1.0) * ELECTRON_REST_MASS_ENERGY

    # Final direction - LAB
    particle["ux"] = vx / particle_speed
    particle["uy"] = vy / particle_speed
    particle["uz"] = vz / particle_speed
'''

# ======================================================================================
# Excitation
# ======================================================================================

@njit
def excitation(particle_container, element, reaction, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]

    # Sample energy loss
    dE = data_processor.evaluate_data(E, reaction["eloss_index"], particle_container, mcdc, data)

    particle["E"] -= dE


# ======================================================================================
# Bremmstrahlung
# ======================================================================================

@njit
def bremsstrahlung(particle_container, element, reaction, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]

    # Sample energy loss
    dE = data_processor.evaluate_data(E, reaction["eloss_index"], particle_container, mcdc, data)

    particle["E"] -= dE


# ======================================================================================
# Ionization
# ======================================================================================

@njit
def compute_mu_delta(T_delta, T_prim):
    me = ELECTRON_REST_MASS_ENERGY
    pd = (T_delta * (T_delta + 2 * me)) ** 0.5
    pp = (T_prim * (T_prim + 2 * me)) ** 0.5
    mu = (T_delta * (T_prim + 2.0 * me)) / (pd * pp)
    if mu < -1.0:
        mu = -1.0
    if mu >  1.0:
        mu =  1.0
    return mu


@njit
def sample_delta_direction(T_delta, T_prim, particle_container):
    mu = compute_mu_delta(T_delta, T_prim)
    phi = 2.0 * PI * kernel.rng(particle_container)
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    ux = s * math.cos(phi)
    uy = s * math.sin(phi)
    uz = mu
    return ux, uy, uz

'''
@njit
def ionization(particle_container, element, reaction, prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]

    N = int(reaction["N_subshells"])
    total = 0.0
    
    for i in range(N):
        idx = int(reaction["subshell_xs_index"][i])
        total += data_processor.evaluate_data(E, DATA_TABLE, idx, mcdc, data)

    xi = kernel.rng(particle_container) * total
    chosen = 0
    run = 0.0

    for i in range(N):
        idx = int(reaction["subshell_xs_index"][i])
        run += data_processor.evaluate_data(E, DATA_TABLE, idx, mcdc, data)

    # Binding energy
    B = reaction["subshell_binding_energy"][chosen]
    if E <= B + TINY:
        return

    # Sample energy of the secondary electron
    prod_index = int(reaction["secondary_energy_index"][chosen])
    T_delta = data_processor.sample_distribution(E, DATA_MULTIPDF, prod_index, particle_container, mcdc, data, True)

    E_out = E - B - T_delta
    particle["E"] = E_out
    ux_d, uy_d, uz_d = sample_delta_direction(T_delta, E, particle_container)

    particle_container_new = adapt.local_array(1, type_.particle_record)
    particle_new = particle_container_new[0]
    kernel.split_as_record(particle_container_new, particle_container)

    particle_new["E"] = T_delta
    particle_new["ux"] = ux_d
    particle_new["uy"] = uy_d
    particle_new["uz"] = uz_d
'''
