import math
import numpy as np
import numba as nb
from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.rng as rng
import mcdc.config as config
import mcdc.numba_types as type_
import mcdc.transport.util as util
import mcdc.transport.linalg as linalg

from mcdc.constant import PROTON_CUTOFF_ENERGY, PROTON_MASS
from mcdc.transport.distribution import sample_normal
from mcdc.transport.data import evaluate_data_clamped


@njit
def max_condensed_step_distance(particle_container, simulation, data):
    """Return the maximum proton condensed step length."""
    condensed_interactions = simulation["settings"]["condensed_interactions"]
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]
    E = particle["E"]
    total_rho = 0.0
    total_dedx = 0.0

    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.material.nuclide_IDs(i, material, data))
        nuclide = simulation["nuclides"][nuclide_ID]

        if not material["stopping_power_provided"]:
            stopping_power = simulation["data"][nuclide["stopping_power_ID"]]
            dedx = evaluate_data_clamped(E / 1e6, stopping_power, simulation, data)
            total_dedx += dedx * 1e6

        atomic_mass = nuclide["atomic_weight_ratio"]
        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
        density_gcm3 = nuclide_density * 1e24 * atomic_mass / (6.022e23)
        total_rho += density_gcm3

    if material["stopping_power_provided"]:
        stopping_power = simulation["data"][material["stopping_power_ID"]]
        dedx = evaluate_data_clamped(E / 1e6, stopping_power, simulation, data)
        total_dedx = dedx * 1e6

    max_fractional_energy_loss = condensed_interactions["max_fractional_energy_loss"]
    return max_fractional_energy_loss * E / total_dedx / total_rho


@njit
def condensed_interactions(
    particle_container, interaction_data_container, distance, simulation, data
):
    """Apply proton condensed interactions over the traveled distance."""
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    material = simulation["materials"][particle["material_ID"]]
    E = particle["E"]

    # Check for cutoff energy
    if E <= PROTON_CUTOFF_ENERGY:
        interaction_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    average_A, average_Z, total_stopping_power, total_rho_gcm3 = (
        calculate_total_stopping_power(particle_container, simulation, data)
    )
    energy_loss = total_stopping_power * total_rho_gcm3 * distance

    # Energy straggling variance in units of MeV^2
    energy_straggling_variance = (
        0.1569 * total_rho_gcm3 * average_Z / average_A * distance
    )
    # Convert to units of eV^2
    energy_straggling_variance *= (1e6) ** 2
    energy_straggling_modifier = math.sqrt(energy_straggling_variance) * sample_normal(
        particle_container
    )
    energy_loss += energy_straggling_modifier

    # Clamping the energy loss to be between [0, particle["E"]]
    energy_loss = min(max(energy_loss, 0.0), E)
    particle["E"] = E - energy_loss
    interaction_data["energy_deposition"] += energy_loss * particle["w"]

    X0 = material["radiation_length"]

    # Angular scattering according to MCS theory
    phi, theta = sample_mcs_angle(
        particle["E"], distance, total_rho_gcm3, X0, particle_container
    )

    rotate_direction(particle, phi, theta)

    return


def negative_sigma_error(sigma):
    raise ValueError(f"negative sigma = {sigma}")


@nb.extending.overload(negative_sigma_error, target="cpu")
def nse_cpu_overload(sigma):
    def impl(sigma):
        raise ValueError(f"negative sigma = {sigma}")

    return impl


@nb.extending.overload(negative_sigma_error, target="gpu")
def nse_cuda_overload(sigma):
    def impl(sigma):
        pass

    return impl


if config.ROCM_AVAILABLE:

    @nb.extending.overload(negative_sigma_error, target="hip")
    def nse_rocm_overload(sigma):
        def impl(sigma):
            pass

        return impl


@njit
def sample_mcs_angle(E, distance, density, X0, particle_container):
    sigma = highland_lynch_dahl_sigma(E, distance, density, X0)

    if sigma < 0.0:
        negative_sigma_error(sigma)

    # Sample theta from the Highland distribution; phi uniformly from (0, 2pi)
    theta = abs(sigma * sample_normal(particle_container))
    phi = 2.0 * np.pi * rng.lcg(particle_container)

    return phi, theta


@njit
def highland_lynch_dahl_sigma(E, distance, density, X0):
    p = math.sqrt(E * (E + 2.0 * PROTON_MASS))
    beta = p / (E + PROTON_MASS)
    z = 1  # Incident particle is a proton, Z=1

    # X0 is measured in g/cm^2
    # Highland formula, modified by Lynch & Dahl
    radiation_distance_fraction = density * distance / X0
    sigma = (
        (13.6e6 / p * beta)
        * z
        * math.sqrt(radiation_distance_fraction)
        * (1 + 0.088 * math.log10(radiation_distance_fraction))
    )
    sigma = abs(sigma)

    if sigma < 0.0:
        negative_sigma_error(sigma)

    return sigma


@njit
def rotate_direction(particle, phi, theta):
    """
    Rotate direction vector (ux, uy, uz) by polar angle theta
    and azimuthal angle phi in the local frame.
    Returns new (ux, uy, uz).
    """

    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Build local perpendicular axes
    d = util.local_array(3, type_.float64)
    d[0] = ux
    d[1] = uy
    d[2] = uz

    perp = util.local_array(3, type_.float64)
    if abs(ux) < 0.9:
        perp[0] = 1.0
        perp[1] = 0.0
        perp[2] = 0.0
    else:
        perp[0] = 0.0
        perp[1] = 1.0
        perp[2] = 0.0

    u = util.local_array(3, type_.float64)
    v = util.local_array(3, type_.float64)

    linalg.cross(u, d, perp)
    linalg.normalize(u)
    linalg.cross(v, d, u)

    d_new = util.local_array(3, type_.float64)
    for i in range(3):
        d_new[i] = (
            cos_theta * d[i] + sin_theta * cos_phi * u[i] + sin_theta * sin_phi * v[i]
        )
    linalg.normalize(d_new)

    particle["ux"] = d_new[0]
    particle["uy"] = d_new[1]
    particle["uz"] = d_new[2]


@njit
def calculate_total_stopping_power(particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]
    E = particle["E"]

    total_stopping_power = 0.0
    total_rho_gcm3 = 0.0
    total_Z = 0.0
    total_A = 0.0
    # Find the total stopping power by summing over every nuclide in the material
    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.material.nuclide_IDs(i, material, data))
        nuclide = simulation["nuclides"][nuclide_ID]

        # If no stopping power provided, we calculate it ourselves here
        if not material["stopping_power_provided"]:
            stopping_power = simulation["data"][nuclide["stopping_power_ID"]]
            dedx = evaluate_data_clamped(E / 1e6, stopping_power, simulation, data)
            total_stopping_power += dedx * 1e6

        # Convert atoms/barn-cm to g/cm3:
        atomic_mass = nuclide["atomic_weight_ratio"]  # mass in amu
        nuclide_density = mcdc_get.material.nuclide_densities(i, material, data)
        density_gcm3 = nuclide_density * 1e24 * atomic_mass / (6.022e23)
        total_rho_gcm3 += density_gcm3

        element = simulation["elements"][nuclide["element_ID"]]
        total_Z += element["atomic_number"]
        total_A += nuclide["mass_number"]

    average_Z = total_Z / material["N_nuclide"]
    average_A = total_A / material["N_nuclide"]

    if material["stopping_power_provided"]:
        stopping_power = simulation["data"][material["stopping_power_ID"]]
        dedx = evaluate_data_clamped(E / 1e6, stopping_power, simulation, data)
        total_stopping_power = dedx * 1e6

    return average_A, average_Z, total_stopping_power, total_rho_gcm3
