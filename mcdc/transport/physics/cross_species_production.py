import math

from numba import njit

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    ANGLE_ENERGY_CORRELATED,
    ANGLE_ISOTROPIC,
    EVENT_TIME_CENSUS,
    PARTICLE_ELECTRON,
    PARTICLE_NEUTRON,
    PARTICLE_PROTON,
    PI,
    REFERENCE_FRAME_COM,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_correlated_distribution_with_scale,
    sample_distribution,
    sample_distribution_with_scale,
    sample_isotropic_cosine,
)
from mcdc.transport.physics.util import scatter_direction


@njit
def produce_cross_species(
    reaction,
    atomic_weight_ratio,
    particle_container,
    interaction_data_container,
    program,
    data,
):
    """Sample and bank transported products of a different species.

    The reaction supplies the available-energy deposition balance. Subtract only
    energy carried by transported products. Use the saved incident state for
    distributions and the active particle's RNG stream for production sampling.
    Currently supports the proton library's prompt, single-spectrum products and
    its existing nuclear center-of-mass transformation.
    """
    simulation = util.access_simulation(program)
    settings = simulation["settings"]
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    incident = interaction_data["incident_particle"]
    E = incident["E"]
    ux, uy, uz = incident["ux"], incident["uy"], incident["uz"]

    for i in range(reaction["N_secondary_product"]):
        product_ID = int(data[reaction["secondary_product_IDs_offset"] + i])
        product = simulation["secondary_products"][product_ID]

        # Same-species products belong to the particle-specific reaction.
        product_type = product["particle_type"]
        if product_type == incident["particle_type"]:
            continue

        # Untransported products retain their energy in the deposition balance.
        active = False
        if product_type == PARTICLE_NEUTRON:
            active = settings["neutron_transport"]["active"]
        elif product_type == PARTICLE_ELECTRON:
            active = settings["electron_transport"]["active"]
        elif product_type == PARTICLE_PROTON:
            active = settings["proton_transport"]["active"]
        if not active:
            continue

        # Get the product count
        #   Multiplicity is represented by yield rather than repeated product entries.
        yield_data = simulation["data"][product["production_yield_ID"]]
        production_yield = evaluate_data(E, yield_data, simulation, data)
        N_product = int(math.floor(production_yield + rng.lcg(particle_container)))

        # Get the spectrum
        spectrum_ID = mcdc_get.secondary_product.energy_spectrum_IDs(0, product, data)
        spectrum = simulation["distributions"][spectrum_ID]

        # Create the products
        for _ in range(N_product):
            particle_container_new = util.local_array(1, type_.particle_data)
            particle_module.copy_as_child(particle_container_new, particle_container)
            particle_new = particle_container_new[0]
            particle_new["x"] = incident["x"]
            particle_new["y"] = incident["y"]
            particle_new["z"] = incident["z"]
            particle_new["t"] = incident["t"]
            particle_new["w"] = incident["w"]

            # Sample energy and angle
            if product["angle_type"] == ANGLE_ENERGY_CORRELATED:
                E_new, mu = sample_correlated_distribution_with_scale(
                    E,
                    spectrum,
                    particle_container_new,
                    simulation,
                    data,
                )
            else:
                E_new = sample_distribution_with_scale(
                    E,
                    spectrum,
                    particle_container_new,
                    simulation,
                    data,
                )
                if product["angle_type"] == ANGLE_ISOTROPIC:
                    mu = sample_isotropic_cosine(particle_container_new)
                else:
                    mu_distribution = simulation["distributions"][product["mu_ID"]]
                    mu = sample_distribution(
                        E, mu_distribution, particle_container_new, simulation, data
                    )

            # Frame transformation
            if product["reference_frame"] == REFERENCE_FRAME_COM:
                A = atomic_weight_ratio
                E_COM = E_new
                E_new = (
                    E_COM
                    + (E + 2.0 * mu * (A + 1.0) * math.sqrt(E * E_COM)) / (A + 1.0) ** 2
                )
                mu = mu * math.sqrt(E_COM / E_new) + math.sqrt(E / E_new) / (A + 1.0)

            azi = 2.0 * PI * rng.lcg(particle_container_new)
            ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu, azi)

            # Finalize the product and account for energy carried away.
            particle_new["ux"] = ux_new
            particle_new["uy"] = uy_new
            particle_new["uz"] = uz_new
            particle_new["E"] = E_new
            particle_new["particle_type"] = product_type
            interaction_data["energy_deposition"] -= E_new * particle_new["w"]

            # Prompt products follow the parent's census routing.
            if particle["event"] & EVENT_TIME_CENSUS:
                particle_bank_module.bank_census_particle(
                    particle_container_new, program
                )
            else:
                particle_bank_module.bank_active_particle(
                    particle_container_new, program
                )
