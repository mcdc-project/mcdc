import math

from numba import njit

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    DISTRIBUTION_NONE,
    EVENT_TIME_CENSUS,
    PARTICLE_ELECTRON,
    PARTICLE_NEUTRON,
    PARTICLE_PHOTON,
    PARTICLE_POSITRON,
    PARTICLE_PROTON,
    NEUTRON_MASS,
    PROTON_MASS,
    PI,
    REFERENCE_FRAME_COM,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_correlated_distribution_with_scale,
    sample_distribution,
    sample_distribution_with_scale,
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
        if not is_transported(product_type, settings):
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
            mu_distribution = simulation["distributions"][product["mu_ID"]]
            if mu_distribution["sub_type"] == DISTRIBUTION_NONE:
                E_new, mu = sample_correlated_distribution_with_scale(
                    E,
                    spectrum,
                    particle_container_new,
                    simulation,
                    data,
                    atomic_weight_ratio * NEUTRON_MASS / PROTON_MASS,
                    reaction["q_value"],
                )
            else:
                E_new = sample_distribution_with_scale(
                    E,
                    spectrum,
                    particle_container_new,
                    simulation,
                    data,
                )
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

            bank_cross_species(
                particle_container_new,
                product_type,
                E_new,
                ux_new,
                uy_new,
                uz_new,
                particle_container,
                interaction_data_container,
                program,
            )


@njit
def is_transported(particle_type, settings):
    """Whether products of ``particle_type`` are transported in this simulation."""
    if particle_type == PARTICLE_NEUTRON:
        return settings["neutron_transport"]["active"]
    elif particle_type == PARTICLE_ELECTRON or particle_type == PARTICLE_POSITRON:
        return settings["electron_transport"]["active"]
    elif particle_type == PARTICLE_PROTON:
        return settings["proton_transport"]["active"]
    elif particle_type == PARTICLE_PHOTON:
        return settings["photon_transport"]["active"]
    return False


@njit
def emit_cross_species(
    product_type,
    E,
    ux,
    uy,
    uz,
    particle_container,
    interaction_data_container,
    program,
):
    """Bank one kinematically determined product at the active particle's site.

    For products whose energy and direction follow from the reaction itself
    rather than from a tabulated spectrum, such as a Compton recoil electron or
    a bremsstrahlung photon. The caller checks ``is_transported`` first, so an
    untransported product leaves its energy, and the RNG stream, untouched.
    """
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_module.copy_as_child(particle_container_new, particle_container)
    bank_cross_species(
        particle_container_new,
        product_type,
        E,
        ux,
        uy,
        uz,
        particle_container,
        interaction_data_container,
        program,
    )


@njit
def bank_cross_species(
    particle_container_new,
    product_type,
    E,
    ux,
    uy,
    uz,
    particle_container,
    interaction_data_container,
    program,
):
    """Finalize a cross-species product and bank it.

    The product's energy is subtracted from the deposition balance, and it
    follows the parent's census routing.
    """
    particle = particle_container[0]
    particle_new = particle_container_new[0]
    interaction_data = interaction_data_container[0]

    particle_new["ux"] = ux
    particle_new["uy"] = uy
    particle_new["uz"] = uz
    particle_new["E"] = E
    particle_new["particle_type"] = product_type
    interaction_data["energy_deposition"] -= E * particle_new["w"]

    # Prompt products follow the parent's census routing.
    if particle["event"] & EVENT_TIME_CENSUS:
        particle_bank_module.bank_census_particle(particle_container_new, program)
    else:
        particle_bank_module.bank_active_particle(particle_container_new, program)
