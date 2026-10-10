import math

from numba import njit

####

import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    LIGHT_SPEED,
    PHOTON_REACTION_COHERENT,
    PHOTON_REACTION_INCOHERENT,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    PHOTON_REACTION_TOTAL,
    PI,
)

# ======================================================================================
# Applicability
# ======================================================================================


@njit
def applicable(particle_container, simulation, data):
    """Whether the colliding material carries constant photon cross sections."""
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]
    return material["has_photon_constant_xs"]


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container, simulation, data):
    return LIGHT_SPEED


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    """Return the energy-independent macroscopic cross section.

    The constant-cross-section treatment has only a scattering and an absorption
    channel, so the two tabulated photon channels that have no analogue here --
    coherent scattering and pair production -- return zero. Incoherent
    scattering stands in for scattering and photoelectric absorption for
    absorption, which keeps the reaction constants meaningful to a tally filter.
    """
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]
    constant_xs = simulation["photon_constant_xs_data"][
        material["photon_constant_xs_ID"]
    ]

    if reaction_type == PHOTON_REACTION_TOTAL:
        return constant_xs["total"]
    elif reaction_type == PHOTON_REACTION_INCOHERENT:
        return constant_xs["scatter"]
    elif reaction_type == PHOTON_REACTION_PHOTOELECTRIC:
        return constant_xs["absorb"]
    elif reaction_type == PHOTON_REACTION_COHERENT:
        return 0.0
    elif reaction_type == PHOTON_REACTION_PAIR_PRODUCTION:
        return 0.0

    return 0.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, interaction_data_container, program, data):
    """Sample scatter or absorb, with isotropic elastic scattering.

    This is the transport problem the analytical benchmarks are posed with: the
    cross sections do not depend on energy, scattering is isotropic and does not
    change the photon energy, and absorption is analog capture. Because
    scattering is elastic, a scatter deposits nothing and a capture deposits the
    whole incident energy.
    """
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    material = simulation["materials"][particle["material_ID"]]
    constant_xs = simulation["photon_constant_xs_data"][
        material["photon_constant_xs_ID"]
    ]

    sigma_total = constant_xs["total"]
    if sigma_total <= 0.0:
        return

    xi = rng.lcg(particle_container) * sigma_total

    if xi < constant_xs["absorb"]:
        # Analog capture: the full incident energy deposits locally.
        interaction_data["energy_deposition"] += particle["E"] * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # Isotropic elastic scattering: the energy is unchanged, so nothing is
    # deposited and only the direction changes.
    mu = 2.0 * rng.lcg(particle_container) - 1.0
    azi = 2.0 * PI * rng.lcg(particle_container)
    sin_polar = math.sqrt(max(0.0, 1.0 - mu * mu))

    particle["ux"] = mu
    particle["uy"] = math.cos(azi) * sin_polar
    particle["uz"] = math.sin(azi) * sin_polar
