import math
import numpy as np

from numba import njit

####

import mcdc.code_factory.adapt as adapt
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
from mcdc.print_ import print_structure
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng

from mcdc.constant import (
    ANGLE_DISTRIBUTED,
    ANGLE_ENERGY_CORRELATED,
    ANGLE_ISOTROPIC,
    LIGHT_SPEED,
    PI,
    PI_HALF,
    PI_SQRT,
    ELECTRON_MASS,
    REACTION_TOTAL,
    REACTION_ELECTRON_IONIZATION,
    REACTION_ELECTRON_ELASTIC_SCATTERING,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_EXCITATION,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_correlated_distribution,
    sample_distribution,
    sample_isotropic_cosine,
    sample_isotropic_direction,
    sample_multi_table,
)
from mcdc.transport.physics.util import evaluate_xs_energy_grid, scatter_direction
from mcdc.transport.util import find_bin, linear_interpolation


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    particle = particle_container[0]
    E = particle["E"]
    mass = ELECTRON_MASS
    return LIGHT_SPEED * math.sqrt(E * (E + 2.0 * mass)) / (E + mass)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["native_materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0

    for i in range(material["N_element"]):
        element_ID = int(mcdc_get.native_material.element_IDs(i, material, data))
        element = mcdc["elements"][element_ID]

        element_density = mcdc_get.native_material.element_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, element, data)

        total += element_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, element, data):
    idx, E0, E1 = evaluate_xs_energy_grid(E, element, data)
    if reaction_type == REACTION_TOTAL:
        xs0 = mcdc_get.element.total_xs(idx, element, data)
        xs1 = mcdc_get.element.total_xs(idx + 1, element, data)
    elif reaction_type == REACTION_ELECTRON_IONIZATION:
        xs0 = mcdc_get.element.ionization_xs(idx, element, data)
        xs1 = mcdc_get.element.ionization_xs(idx + 1, element, data)
    elif reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
        xs0 = mcdc_get.element.elastic_xs(idx, element, data)
        xs1 = mcdc_get.element.elastic_xs(idx + 1, element, data)
    elif reaction_type == REACTION_ELECTRON_EXCITATION:
        xs0 = mcdc_get.element.excitation_xs(idx, element, data)
        xs1 = mcdc_get.element.excitation_xs(idx + 1, element, data)
    elif reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
        xs0 = mcdc_get.element.bremsstrahlung_xs(idx, element, data)
        xs1 = mcdc_get.element.bremsstrahlung_xs(idx + 1, element, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction_base, element, data):
    idx, E0, E1 = evaluate_xs_energy_grid(E, element, data)

    # Apply offset
    offset = reaction_base["xs_offset_"]
    if idx < offset:
        return 0.0
    else:
        idx -= offset

    xs0 = mcdc_get.reaction.xs(idx, reaction_base, data)
    xs1 = mcdc_get.reaction.xs(idx + 1, reaction_base, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def electron_production_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material_base = mcdc["materials"][particle["material_ID"]]
    material = mcdc["native_materials"][material_base["child_ID"]]

    if reaction_type == REACTION_TOTAL:
        ionization_type = REACTION_ELECTRON_IONIZATION
        elastic_type = REACTION_ELECTRON_ELASTIC_SCATTERING
        bremmstrahlung_type = REACTION_ELECTRON_BREMSSTRAHLUNG
        excitation_type = REACTION_ELECTRON_EXCITATION

        ionization_xs = electron_production_xs(ionization_type, particle_container, mcdc, data)        
        elastic_xs = electron_production_xs(elastic_type, particle_container, mcdc, data)
        bremmstrahlung_xs = electron_production_xs(bremmstrahlung_type, particle_container, mcdc, data)
        excitation_xs = electron_production_xs(excitation_type, particle_container, mcdc, data)
        return ionization_xs + elastic_xs + bremmstrahlung_xs + excitation_xs

    elif reaction_type == REACTION_ELECTRON_IONIZATION:
        return 2 * macro_xs(reaction_type, particle_container, mcdc, data)

    elif reaction_type == REACTION_ELECTRON_ELASTIC_SCATTERING:
        return macro_xs(reaction_type, particle_container, mcdc, data)
    
    elif reaction_type == REACTION_ELECTRON_BREMSSTRAHLUNG:
        return macro_xs(reaction_type, particle_container, mcdc, data)

    elif reaction_type == REACTION_ELECTRON_EXCITATION:
        return macro_xs(reaction_type, particle_container, mcdc, data)
    
    else:
        return -1.0


# ======================================================================================
# Collision
# ======================================================================================
