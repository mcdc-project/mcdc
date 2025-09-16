import math
import numpy as np

#from numba import njit

####

import mcdc.adapt as adapt
import mcdc.data_processor as data_processor
import mcdc.kernel as kernel
import mcdc.objects as objects
import mcdc.type_ as type_

from mcdc.constant import (DATA_MULTIPDF,
                           DATA_TABLE,
                           PI,
                           ELECTRON_REST_MASS_ENERGY,
                           SPEED_OF_LIGHT,
                           EEDL_THRESHOLD,
                           TINY
                           )
from mcdc.physics.util import scatter_direction
from mcdc.util import binary_search

from mcdc.constant import (
    REACTION_TOTAL,
    REACTION_ELECTRON_ELASTIC_SCATTERING,
    REACTION_ELECTRON_ELASTIC_SMALL_ANGLE,
    REACTION_ELECTRON_ELASTIC_LARGE_ANGLE,
    REACTION_ELECTRON_EXCITATION,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_IONIZATION
)

from mcdc.util import linear_interpolation

# ======================================================================================
# Particle attributes
# ======================================================================================

#@njit
def particle_speed(particle_container):
    E = particle_container[0]["E"]

    gamma = 1.0 + E / ELECTRON_REST_MASS_ENERGY
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    return math.sqrt(beta_sq) * SPEED_OF_LIGHT

# ======================================================================================
# Temporary functions for testing without numba
# ======================================================================================

def evaluate_xs_energy_grid(e, element):
    energy_grid = element.xs_energy_grid
    idx = binary_search(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1

def reaction_index(index, element):
    offset = element["reaction_index_offset"]
    return offset + index

def atomic_densities(index, material):
    offset = material["atomic_densities_offset"]
    return offset + index

def sample_multipdf(x, rng_state, multipdf, scale=False):
    grid = multipdf.grid

    # Edge cases
    if x < grid[0]:
        idx = 0
        scale = False
    elif x > grid[-1]:
        idx = len(grid) - 1
        scale = False
    else:
        # Interpolation factor
        idx = binary_search(x, grid)
        x0 = grid[idx]
        x1 = grid[idx + 1]
        f = (x - x0) / (x1 - x0)
    
        # Min and max values for scaling
        val_min = 0.0
        val_max = 1.0
        if scale:
            # First table
            start = int(multipdf.offset[idx])
            end = int(multipdf.offset[idx + 1]) if (idx + 1) < len(multipdf.offset) else len(multipdf.value)
            val0_min = multipdf.value[start]
            val0_max = multipdf.value[end - 1]
            
            # Second table
            start = end
            if idx + 2 == len(grid):
                end = len(multipdf.value)
            else:
                end = int(multipdf.offset[idx + 2])
            val1_min = multipdf.value[start]
            val1_max = multipdf.value[end - 1]

            # Both
            val_min = val0_min + f * (val1_min - val0_min)
            val_max = val0_max + f * (val1_max - val0_max)

        # Sample which table to choose
        if kernel.rng(rng_state) > f:
            idx += 1

    # Get the table range
    start = int(multipdf.offset[idx])
    if idx + 1 == len(grid):
        end = len(multipdf.value)
    else:
        end = int(multipdf.offset[idx + 1])

    # The CDF
    cdf = multipdf.cdf[start:end]

    # Generate random numbers
    xi = kernel.rng(rng_state)

    # Sample bin index
    idx = binary_search(xi, cdf)
    c = cdf[idx]

    # Get the other values
    idx += start
    p0 = multipdf.pdf[idx]
    p1 = multipdf.pdf[idx + 1]
    val0 = multipdf.value[idx]
    val1 = multipdf.value[idx + 1]

    m = (p1 - p0) / (val1 - val0)

    if m == 0.0:
        sample = val0 + (xi - c) / p0
    else:
        sample = val0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi - c)) - p0)

    if not scale:
        return sample
    
    # Scale against the bounds
    val_low = multipdf.value[start]
    val_high = multipdf.value[end - 1]
    return val_min + (sample - val_low) / (val_high - val_low) * (val_max - val_min)

#@njit
def sample_distribution(x, data_type, index, rng_state, scale=False):
    if data_type == DATA_MULTIPDF:
        multipdf = index
        return sample_multipdf(x, rng_state, multipdf, scale)
    else:
        return 0.0
    
#@njit
def evaluate_table(x, table):
    grid = table.x
    idx = binary_search(x, grid)
    x1 = grid[idx]
    x2 = grid[idx + 1]
    y = table.y   
    y1 = y[idx]
    y2 = y[idx + 1]
    return linear_interpolation(x, x1, x2, y1, y2)


def evaluate_data(x, data_type, index):
    if data_type == DATA_TABLE:
        table = index
        return evaluate_table(x, table)
    else:
        return 0.0

# ======================================================================================
# Material properties
# ======================================================================================

#@njit
def macro_xs(reaction_type, material, particle_container):
    particle = particle_container[0]
    E = particle["E"]

    # Sum over all elements
    total = 0.0
    elements = material.elements
    N_elements = len(elements)
    element_densities = material.element_densities
    for i in range(N_elements):
        element = elements[i]
        element_density = element_densities[i]
        xs = micro_xs(E, reaction_type, element)
        total += element_density * xs
    return total


#@njit
def micro_xs(E, reaction_type, element):
    idx, E0, E1 = evaluate_xs_energy_grid(E, element)
 
    for reaction in element.reactions:
        the_type = int(reaction.type)
        if the_type == reaction_type:
            # Reaction exists!
            xs0 = reaction.xs[idx]
            xs1 = reaction.xs[idx + 1]
            return linear_interpolation(E, E0, E1, xs0, xs1)
        
    return 0.0


#@njit
def electron_production_xs(reaction_type, material, particle_container):
    particle = particle_container[0]
    E = particle["E"]

    if E < EEDL_THRESHOLD:
        return 0.0

    if reaction_type in (REACTION_ELECTRON_ELASTIC_SMALL_ANGLE,
                         REACTION_ELECTRON_ELASTIC_LARGE_ANGLE):
        reaction_type = REACTION_ELECTRON_ELASTIC_SCATTERING

    if reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
        return macro_xs(reaction_type, material, particle_container)

    if reaction_type == REACTION_ELECTRON_EXCITATION:
        return macro_xs(reaction_type, material, particle_container)

    if reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
        return macro_xs(reaction_type, material, particle_container)

    if reaction_type == REACTION_ELECTRON_IONIZATION:
        return 2.0 * macro_xs(reaction_type, material, particle_container)

    if reaction_type == REACTION_TOTAL:
        total = 0.0
        total += electron_production_xs(REACTION_ELECTRON_ELASTIC_SCATTERING,
                                        material, particle_container)
        total += electron_production_xs(REACTION_ELECTRON_EXCITATION,
                                        material, particle_container)
        total += electron_production_xs(REACTION_ELECTRON_BREMSSTRAHLUNG,
                                        material, particle_container)
        total += electron_production_xs(REACTION_ELECTRON_IONIZATION,
                                        material, particle_container)
        return total


# ======================================================================================
# Collision
# ======================================================================================

#@njit
def collision(particle_container, prog):
    particle = particle_container[0]
    material = objects.materials[particle["material_ID"]]

    # ==================================================================================
    # Sample colliding element
    # ==================================================================================

    SigmaT = macro_xs(REACTION_TOTAL, material, particle_container)
    xi = kernel.rng(particle_container) * SigmaT
    total = 0.0

    elements = material.elements
    N_elements = len(elements)
    element_densities = material.element_densities

    for i in range(N_elements): 
        element = elements[i]
        element_density = element_densities[i]
        sigmaT = micro_xs(particle["E"], REACTION_TOTAL, element)
        SigmaT_element = element_density * sigmaT
        total += SigmaT_element
        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    xi = kernel.rng(particle_container) * sigmaT
    total = 0.0

    reactions = element.reactions
    N_reaction = len(reactions)
    for i in range(N_reaction):
        reaction = reactions[i]
        E = particle["E"]
        reaction_type = int(reaction.type)
        reaction_xs = micro_xs(E, reaction_type, element)
        total += reaction_xs
        if total < xi:
            continue

        # Reaction is sampled
        if reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
            #reaction = "electron_bremsstrahlung_reaction"
            bremsstrahlung(particle_container, element, reaction)

        elif reaction_type == REACTION_ELECTRON_EXCITATION:
            #reaction = "electron_excitation_reaction"
            excitation(particle_container, element, reaction)
        elif reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
            #reaction = "electron_elastic_scattering_reaction"
            elastic_scattering(particle_container, element, reaction)

        elif reaction_type == REACTION_ELECTRON_IONIZATION:
            #reaction = "electron_ionization_reaction"
            ionization(particle_container, element, reaction, prog)


# ======================================================================================
# Elastic scattering
# ======================================================================================

#@njit
def elastic_scattering(particle_container, element, reaction):

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
    xs_L = element.large_angle_xs(E, particle_container)
    xs_S = element.small_angle_xs(E, particle_container)

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
    speed_lab = particle_speed(particle_container)

    # Electron velocity - LAB
    vx = speed_lab * ux
    vy = speed_lab * uy
    vz = speed_lab * uz

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
    speed_com = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Electron initial direction - COM
    ux = vx / speed_com
    uy = vy / speed_com
    uz = vz / speed_com

    # Sample the scattering cosine from the multi-PDF distribution
    index = reaction.mu_large_angle if use_large else reaction.mu_small_angle
    mu0 = sample_distribution(E, DATA_MULTIPDF, index, particle_container)

    # Scatter the direction in COM
    azi = 2.0 * PI * kernel.rng(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Electron final velocity - COM
    vx = speed_com * ux_new
    vy = speed_com * uy_new
    vz = speed_com * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx += COM_x
    vy += COM_y
    vz += COM_z

    # Final relativistic kinetic energy from speed - LAB
    speed_lab_out = math.sqrt(vx * vx + vy * vy + vz * vz)
    gamma_out = 1.0 / math.sqrt(1.0 - (speed_lab_out / SPEED_OF_LIGHT) ** 2)
    particle["E"] = (gamma_out - 1.0) * ELECTRON_REST_MASS_ENERGY

    # Final direction - LAB
    particle["ux"] = vx / speed_lab_out
    particle["uy"] = vy / speed_lab_out
    particle["uz"] = vz / speed_lab_out


# ======================================================================================
# Excitation
# ======================================================================================

#@njit
def excitation(particle_container, element, reaction):
    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]

    # Sample energy loss
    e_loss = reaction.eloss
    dE = evaluate_data(E, DATA_TABLE, e_loss)
    particle["E"] -= dE


# ======================================================================================
# Bremmstrahlung
# ======================================================================================

#@njit
def bremsstrahlung(particle_container, element, reaction):
    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]

    # Sample energy loss
    e_loss = reaction.eloss
    dE = evaluate_data(E, DATA_TABLE, e_loss)
    particle["E"] -= dE


# ======================================================================================
# Ionization
# ======================================================================================

#@njit
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


#@njit
def sample_delta_direction(T_delta, T_prim, particle_container):
    mu = compute_mu_delta(T_delta, T_prim)
    phi = 2.0 * PI * kernel.rng(particle_container)
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    ux = s * math.cos(phi)
    uy = s * math.sin(phi)
    uz = mu
    return ux, uy, uz


#@njit
def ionization(particle_container, element, reaction, prog):
    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]

    N = reaction.N_subshell
    total = 0.0
    
    for i in range(N):
        subshell_xs = reaction.subshell_xs[i]
        total += evaluate_data(E, DATA_TABLE, subshell_xs)

    xi = kernel.rng(particle_container) * total
    chosen = 0
    run = 0.0

    for i in range(N):
        subshell_xs = reaction.subshell_xs[i]
        run += evaluate_data(E, DATA_TABLE, subshell_xs)
        if run >= xi:
            chosen = i
            break

    # Binding energy
    B = reaction.subshell_binding_energy[chosen]
    if E <= B + TINY:
        return

    # Sample energy of the secondary electron
    secondary_multipdf = reaction.subshell_product[chosen]
    T_delta = sample_distribution(E, DATA_MULTIPDF, secondary_multipdf, particle_container, True)

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
    particle_new["w"] = particle["w"]
    particle_new["alive"] = True

    adapt.add_active(particle_container_new, prog)
