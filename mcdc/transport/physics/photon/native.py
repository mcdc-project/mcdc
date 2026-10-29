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
    DATA_TABLE,
    ELECTRON_MASS,
    LIGHT_SPEED,
    PHOTON_CUTOFF_ENERGY,
    PHOTON_REACTION_COHERENT_SCATTERING,
    PHOTON_REACTION_INCOHERENT_SCATTERING,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    PHOTON_REACTION_TOTAL,
    PI,
)
from mcdc.transport.data import evaluate_data, evaluate_data_clamped
from mcdc.transport.distribution import sample_isotropic_direction
from mcdc.transport.physics.util import (
    evaluate_photon_xs_energy_grid,
    evaluate_reaction_xs,
    interpolate_xs,
    scatter_direction,
)

# ======================================================================================
# Local physical constants
#
# Every energy below is in eV, matching the runtime unit used throughout MC/DC
# and the unit the photon library declares on disk.
# ======================================================================================

#: Pair-production threshold, ``2 m_e c^2`` = 1021997.90138 eV.
PAIR_PRODUCTION_THRESHOLD = 2.0 * ELECTRON_MASS

#: ``h c`` in eV-angstrom. Converts a photon energy to the wavenumber
#: ``k = E / (h c)`` in inverse angstroms, which is the unit the library's
#: coherent form-factor momentum-transfer grid is tabulated against.
PLANCK_LIGHT_SPEED_EV_ANGSTROM = 12398.42


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    """Photons travel at the speed of light regardless of energy."""
    return LIGHT_SPEED


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0
    for i in range(material["N_element"]):
        element_ID = mcdc_get.material.element_IDs(i, material, data)
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.material.element_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, element, simulation, data)
        total += element_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, element, simulation, data):
    """Use summed linear tables, or sum individually interpolated reactions."""
    grid = mcdc_get.element.photon_xs_energy_grid_all(element, data)
    E = min(max(E, grid[0]), grid[-1])
    if not element["photon_xs_is_linear"]:
        total = 0.0
        for i in range(element["N_photon_reaction"]):
            ID = mcdc_get.element.photon_reaction_IDs(i, element, data)
            reaction = simulation["photon_reactions"][ID]
            if (
                reaction_type == PHOTON_REACTION_TOTAL
                or reaction["sub_type"] == reaction_type
            ):
                total += reaction_micro_xs(E, reaction, element, data)
        return total
    idx, E0, E1 = evaluate_photon_xs_energy_grid(E, element, data)
    if reaction_type == PHOTON_REACTION_TOTAL:
        xs0 = mcdc_get.element.photon_total_xs(idx, element, data)
        xs1 = mcdc_get.element.photon_total_xs(idx + 1, element, data)
    elif reaction_type == PHOTON_REACTION_COHERENT_SCATTERING:
        xs0 = mcdc_get.element.photon_coherent_xs(idx, element, data)
        xs1 = mcdc_get.element.photon_coherent_xs(idx + 1, element, data)
    elif reaction_type == PHOTON_REACTION_INCOHERENT_SCATTERING:
        xs0 = mcdc_get.element.photon_incoherent_xs(idx, element, data)
        xs1 = mcdc_get.element.photon_incoherent_xs(idx + 1, element, data)
    elif reaction_type == PHOTON_REACTION_PHOTOELECTRIC:
        xs0 = mcdc_get.element.photon_photoelectric_xs(idx, element, data)
        xs1 = mcdc_get.element.photon_photoelectric_xs(idx + 1, element, data)
    elif reaction_type == PHOTON_REACTION_PAIR_PRODUCTION:
        if E < PAIR_PRODUCTION_THRESHOLD:
            return 0.0
        xs0 = mcdc_get.element.photon_pair_production_xs(idx, element, data)
        xs1 = mcdc_get.element.photon_pair_production_xs(idx + 1, element, data)
    else:
        # Should be unreachable
        return 0.0

    if E >= grid[-1]:
        return xs1
    return interpolate_xs(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction, element, data):
    """Evaluate one shared-grid segment with right-hand values at absorption edges."""
    grid = mcdc_get.element.photon_xs_energy_grid_all(element, data)
    return evaluate_reaction_xs(
        E,
        grid,
        reaction["xs_offset_"],
        mcdc_get.photon_reaction.xs_all(reaction, data),
        mcdc_get.photon_reaction.xs_interpolations_all(reaction, data),
        mcdc_get.photon_reaction.xs_interpolation_boundaries_all(reaction, data),
    )


@njit
def sample_reaction(reaction_type, particle_container, element, simulation, data):
    """Select a reaction within a category using its microscopic cross section."""
    E = particle_container[0]["E"]
    total = total_micro_xs(reaction_type, E, element, simulation, data)
    xi = rng.lcg(particle_container) * total
    cumulative = 0.0
    selected = -1
    for i in range(element["N_photon_reaction"]):
        ID = mcdc_get.element.photon_reaction_IDs(i, element, data)
        reaction = simulation["photon_reactions"][ID]
        if reaction["sub_type"] != reaction_type:
            continue
        xs = reaction_micro_xs(E, reaction, element, data)
        if xs <= 0.0:
            continue
        selected = ID
        cumulative += xs
        if cumulative > xi:
            return ID
    return selected


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, interaction_data_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]
    material = simulation["materials"][particle["material_ID"]]

    # Particle properties
    E = particle["E"]

    # Check for cutoff energy
    if E <= PHOTON_CUTOFF_ENERGY:
        interaction_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # ==================================================================================
    # Sample colliding element
    # ==================================================================================

    SigmaT = macro_xs(PHOTON_REACTION_TOTAL, particle_container, simulation, data)

    if SigmaT == 0.0:
        return

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    element = simulation["elements"][0]
    for i in range(material["N_element"]):
        element_ID = mcdc_get.material.element_IDs(i, material, data)
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.material.element_densities(i, material, data)
        sigmaT = total_micro_xs(PHOTON_REACTION_TOTAL, E, element, simulation, data)

        total += element_density * sigmaT
        if total > xi:
            break

    # ==================================================================================
    # Sample and perform the reaction
    # ==================================================================================

    sigma_coherent = total_micro_xs(
        PHOTON_REACTION_COHERENT_SCATTERING, E, element, simulation, data
    )
    sigma_incoherent = total_micro_xs(
        PHOTON_REACTION_INCOHERENT_SCATTERING, E, element, simulation, data
    )
    sigma_photoelectric = total_micro_xs(
        PHOTON_REACTION_PHOTOELECTRIC, E, element, simulation, data
    )
    sigma_total = (
        sigma_coherent + sigma_incoherent + sigma_photoelectric
    ) + total_micro_xs(PHOTON_REACTION_PAIR_PRODUCTION, E, element, simulation, data)

    if sigma_total == 0.0:
        return

    xi = rng.lcg(particle_container) * sigma_total

    if xi < sigma_coherent:
        coherent_scattering(particle_container, element, simulation, data)
    elif xi < sigma_coherent + sigma_incoherent:
        incoherent_scattering(
            particle_container, interaction_data_container, program, data
        )
    elif xi < sigma_coherent + sigma_incoherent + sigma_photoelectric:
        photoelectric(
            particle_container,
            interaction_data_container,
            element,
            program,
            data,
        )
    else:
        pair_production(particle_container, interaction_data_container, program, data)


# ======================================================================================
# Coherent (Rayleigh) scattering
# ======================================================================================


@njit
def coherent_scattering(particle_container, element, simulation, data):
    """Deflect the photon without changing its energy.

    Elastic, so nothing is deposited and no secondary is produced. The
    scattering cosine comes from the tabulated atomic form factor, which
    reproduces the strong forward peaking that grows with energy and with Z.
    """
    particle = particle_container[0]

    mu = sample_coherent_scattering_cosine(
        particle_container, element, simulation, data
    )
    azi = 2.0 * PI * rng.lcg(particle_container)

    ux, uy, uz = scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu, azi
    )
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz


# Cap on the coherent-scattering rejection loop. At high energy F(q)^2 is
# negligible outside a narrow forward cone, the acceptance rate collapses, and
# exhausting the cap falls back to mu = 1. That fallback is a small forward bias,
# to be replaced by direct inversion or a better envelope (tracked upstream).
COHERENT_SCATTERING_MAX_REJECTION_ATTEMPTS = 1000


@njit
def sample_coherent_scattering_cosine(particle_container, element, simulation, data):
    """Sample ``mu`` from ``(1 + mu^2) F(q, Z)^2`` by rejection.

    The Thomson factor ``(1 + mu^2) / 2`` is applied by rejection, which is at
    least 50% efficient at any energy. ``F(q, Z)^2`` is applied by rejection
    against the forward-scattering maximum ``F(0, Z) = Z``, which the library
    tabulates as the first form-factor point.

    Momentum transfer and angle are related through the wavenumber
    ``k = E / (h c)`` by ``q^2 = 2 k^2 (1 - mu)``, so ``q`` ranges from 0 at
    forward scattering to ``2 k`` at backscatter.
    """
    particle = particle_container[0]
    E = particle["E"]

    N_reaction = element["N_photon_coherent_scattering_reaction"]
    if N_reaction == 0:
        return 1.0

    reaction_ID = int(
        mcdc_get.element.photon_coherent_scattering_reaction_IDs(0, element, data)
    )
    if N_reaction > 1:
        reaction_ID = sample_reaction(
            PHOTON_REACTION_COHERENT_SCATTERING,
            particle_container,
            element,
            simulation,
            data,
        )
        if reaction_ID < 0:
            return 1.0
    reaction_base = simulation["photon_reactions"][reaction_ID]
    reaction = simulation["photon_coherent_scattering_reactions"][
        reaction_base["sub_ID"]
    ]
    form_factor = simulation["data"][reaction["form_factor_ID"]]

    # Wavenumber in inverse angstroms, matching the tabulated q grid.
    k = E / PLANCK_LIGHT_SPEED_EV_ANGSTROM
    q_max = 2.0 * k

    # Forward-scattering maximum of the form factor, F(q = 0, Z) = Z.
    form_factor_max = evaluate_data(0.0, form_factor, simulation, data)
    if form_factor_max <= 0.0:
        return 1.0

    # Rejection on F(q)^2 and on the Thomson factor together.
    for _ in range(COHERENT_SCATTERING_MAX_REJECTION_ATTEMPTS):
        mu = 2.0 * rng.lcg(particle_container) - 1.0
        q = q_max * math.sqrt(0.5 * (1.0 - mu))
        form_factor_value = evaluate_data(q, form_factor, simulation, data)
        weight = (form_factor_value / form_factor_max) ** 2
        weight *= 0.5 * (1.0 + mu * mu)
        if rng.lcg(particle_container) <= weight:
            return mu

    # Every attempt rejected: fall back to forward scattering (see the cap above).
    # TODO: Replace capped rejection with an unbiased sampler efficient for
    # sharply forward-peaked form factors; this fallback adds artificial mass at mu=1.
    return 1.0


# ======================================================================================
# Incoherent (Compton) scattering
# ======================================================================================


@njit
def incoherent_scattering(
    particle_container, interaction_data_container, program, data
):
    """Scatter the photon off a free electron with Klein-Nishina kinematics.

    The tabulated incoherent cross section set the interaction probability; the
    outgoing energy and angle come from the Klein-Nishina distribution, sampled
    by Kahn's composition-rejection method. The recoil electron is not
    transported, so its energy deposits at the collision site.
    """
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    E = particle["E"]
    # TODO: Load and apply the incoherent scattering function to weight
    # the Klein-Nishina angular distribution for bound atomic electrons.
    eps, mu = sample_klein_nishina(E, particle_container)

    # The scattered photon carries away E * eps; the recoil electron's energy
    # deposits locally, because electrons are not transported.
    particle["E"] = E * eps
    interaction_data["energy_deposition"] += E * (1.0 - eps) * particle["w"]

    azi = 2.0 * PI * rng.lcg(particle_container)
    ux, uy, uz = scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu, azi
    )
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz


@njit
def sample_klein_nishina(E, particle_container):
    """Sample ``(eps, mu)`` from Klein-Nishina by Kahn (1954) rejection.

    ``eps = E_out / E_in`` is drawn from a two-term envelope, and ``mu`` follows
    from Compton kinematics, ``mu = 1 - (1 / eps - 1) / kappa``. Acceptance is at
    least 50% at every energy.
    """
    kappa = E / ELECTRON_MASS
    tau = 1.0 / (1.0 + 2.0 * kappa)

    # Normalization of the two envelope terms, 1/eps and eps, over [tau, 1].
    a1 = math.log(1.0 / tau)
    a2 = (1.0 - tau * tau) / 2.0

    eps = 1.0
    mu = 1.0
    while True:
        xi1 = rng.lcg(particle_container)
        xi2 = rng.lcg(particle_container)
        xi3 = rng.lcg(particle_container)

        if xi1 * (a1 + a2) < a1:
            eps = tau * math.exp(xi2 * a1)
        else:
            eps = math.sqrt(tau * tau + xi2 * (1.0 - tau * tau))

        mu = 1.0 - (1.0 / eps - 1.0) / kappa
        if mu < -1.0:
            mu = -1.0
        elif mu > 1.0:
            mu = 1.0

        sin2_theta = 1.0 - mu * mu
        if sin2_theta < 0.0:
            sin2_theta = 0.0

        probability_accept = 1.0 - eps * sin2_theta / (1.0 + eps * eps)
        if xi3 <= probability_accept:
            break

    return eps, mu


# ======================================================================================
# Photoelectric absorption
# ======================================================================================


@njit
def photoelectric(
    particle_container, interaction_data_container, element, program, data
):
    """Absorb the photon, then relax the resulting inner-shell vacancy.

    The photoelectron is not transported, so its kinetic energy deposits at the
    collision site. The vacancy de-excites radiatively with probability equal to
    the shell's fluorescence yield, emitting a characteristic X-ray that *is*
    transported because it is the same species; a non-radiative (Auger) outcome
    deposits locally for the same reason the photoelectron does.

    A radiative transition leaves a new vacancy in the subshell the filling
    electron came from, and that vacancy relaxes in turn. One further step is
    followed, so a photoelectric event emits up to **two** photons: the primary
    line and one cascade line. In a high-Z medium the second is the L cascade
    following a K capture, and it is not negligible -- MCNP's own accounting for
    10 MeV photons in lead logs 1.82 first-fluorescence and 0.38
    second-fluorescence photons per source particle.
    """
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    E = particle["E"]

    subshell = sample_photoelectric_subshell(
        particle_container, element, simulation, data
    )

    E_primary = 0.0
    E_cascade = 0.0
    if subshell >= 0:
        E_primary, origin = relax_subshell(
            particle_container, subshell, E, element, data
        )
        if E_primary > 0.0 and origin > 0:
            # The vacancy has moved to the subshell the filling electron left.
            next_subshell = find_subshell_by_designator(origin, element, data)
            if next_subshell >= 0:
                E_cascade, _ = relax_subshell(
                    particle_container,
                    next_subshell,
                    E - E_primary,
                    element,
                    data,
                )

    # Everything the fluorescence photons do not carry away deposits locally.
    interaction_data["energy_deposition"] += (E - E_primary - E_cascade) * particle["w"]

    if E_primary <= 0.0:
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # Revive the current history in place as the primary line, emitted
    # isotropically from the absorption site. Cheaper than killing this history
    # and banking a fresh one.
    ux, uy, uz = sample_isotropic_direction(particle_container)
    particle["E"] = E_primary
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz

    if E_cascade <= 0.0:
        return

    # Bank the cascade line as a new active particle from the same site,
    # inheriting the current weight and emitted in its own direction.
    ux, uy, uz = sample_isotropic_direction(particle_container)
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_module.copy_as_child(particle_container_new, particle_container)
    particle_new = particle_container_new[0]
    particle_new["E"] = E_cascade
    particle_new["ux"] = ux
    particle_new["uy"] = uy
    particle_new["uz"] = uz
    particle_bank_module.bank_active_particle(particle_container_new, program)


@njit
def find_subshell_by_designator(designator, element, data):
    """Map an EADL subshell designator to this element's subshell index.

    Returns ``-1`` when no matching relaxation shell is available. Terminal
    shells have valid indices and zero transition counts.
    """
    N_subshell = int(element["photon_relaxation_subshell_designator_length"])
    for index in range(N_subshell):
        tabulated = int(
            mcdc_get.element.photon_relaxation_subshell_designator(index, element, data)
        )
        if tabulated == designator:
            return index
    return -1


@njit
def sample_photoelectric_subshell(particle_container, element, simulation, data):
    """Choose which subshell the photoelectric event ionizes.

    The vacancy subshell is drawn in proportion to its shell-resolved
    photoelectric cross section at the incident energy.

    Returns
    -------
    int
        Atomic-relaxation subshell index, or ``-1`` when no accessible shell
        has a matching relaxation record.
    """
    particle = particle_container[0]
    E = particle["E"]

    N_reaction = element["N_photon_photoelectric_reaction"]
    if N_reaction == 0:
        return -1

    reaction_ID = int(
        mcdc_get.element.photon_photoelectric_reaction_IDs(0, element, data)
    )
    if N_reaction > 1:
        reaction_ID = sample_reaction(
            PHOTON_REACTION_PHOTOELECTRIC, particle_container, element, simulation, data
        )
        if reaction_ID < 0:
            return -1
    reaction_base = simulation["photon_reactions"][reaction_ID]
    reaction = simulation["photon_photoelectric_reactions"][reaction_base["sub_ID"]]

    N_subshell = reaction["N_subshell"]
    if N_subshell == 0:
        return -1
    if element["photon_relaxation_subshell_count_length"] == 0:
        return -1

    # Total shell-resolved cross section at this energy.
    sigma_total = 0.0
    for i in range(N_subshell):
        subshell_ID = int(
            mcdc_get.photon_photoelectric_reaction.subshell_x_IDs(i, reaction, data)
        )
        subshell_xs = simulation["data"][subshell_ID]
        sigma_total += subshell_micro_xs(E, subshell_xs, simulation, data)

    if sigma_total <= 0.0:
        return -1

    # Select the vacancy subshell.
    xi = rng.lcg(particle_container) * sigma_total
    total = 0.0
    subshell = N_subshell - 1
    for i in range(N_subshell):
        subshell_ID = int(
            mcdc_get.photon_photoelectric_reaction.subshell_x_IDs(i, reaction, data)
        )
        subshell_xs = simulation["data"][subshell_ID]
        total += subshell_micro_xs(E, subshell_xs, simulation, data)
        if total > xi:
            subshell = i
            break

    designator = mcdc_get.photon_photoelectric_reaction.subshell_designators(
        subshell, reaction, data
    )
    return find_subshell_by_designator(designator, element, data)


@njit
def subshell_micro_xs(E, data_object, simulation, data):
    """Return zero below a tabulated subshell threshold and clamp above its grid."""
    if data_object["sub_type"] == DATA_TABLE:
        table = simulation["table_data"][data_object["sub_ID"]]
        if E < mcdc_get.table_data.x(0, table, data):
            return 0.0
    return evaluate_data_clamped(E, data_object, simulation, data)


@njit
def relax_subshell(particle_container, subshell, E_available, element, data):
    """Draw one de-excitation transition for a vacancy in ``subshell``.

    A radiative transition emits a characteristic X-ray at the tabulated line
    energy; a non-radiative (Auger) transition emits nothing, because its
    electron is not transported and so deposits locally.

    A line energy is always the difference between two binding energies and is
    therefore strictly below the binding energy of the shell that produced it,
    so an emitted photon can never re-ionize that shell. The cascade terminates
    by construction.

    Returns
    -------
    (float, int)
        Line energy in eV -- ``0.0`` when the outcome is non-radiative or no
        transition is accessible -- and the EADL designator of the subshell the
        vacancy moves to, or ``-1`` when there is none to follow.
    """
    if subshell < 0:
        return 0.0, -1
    if subshell >= int(element["photon_relaxation_subshell_count_length"]):
        return 0.0, -1

    N_transition = int(
        mcdc_get.element.photon_relaxation_subshell_count(subshell, element, data)
    )
    if N_transition == 0:
        return 0.0, -1

    start = int(
        mcdc_get.element.photon_relaxation_subshell_start(subshell, element, data)
    )

    # Sample the full branching distribution, including radiative and Auger channels.
    xi = rng.lcg(particle_container)
    total = 0.0
    for i in range(N_transition):
        total += mcdc_get.element.photon_relaxation_transition_probability(
            start + i, element, data
        )
        if total > xi:
            radiative = mcdc_get.element.photon_relaxation_transition_radiative(
                start + i, element, data
            )
            if radiative == 0.0:
                # Non-radiative (Auger): the electron is not transported.
                return 0.0, -1

            E_line = mcdc_get.element.photon_relaxation_transition_energy(
                start + i, element, data
            )
            if E_line <= 0.0 or E_line > E_available:
                return 0.0, -1
            origin = int(
                mcdc_get.element.photon_relaxation_transition_origin(
                    start + i, element, data
                )
            )
            return E_line, origin

    # No transition selected after rounding: deposit locally.
    return 0.0, -1


# ======================================================================================
# Pair production
# ======================================================================================


@njit
def pair_production(particle_container, interaction_data_container, program, data):
    """Convert the photon to an electron-positron pair.

    The pair's kinetic energy, ``E - 2 m_e c^2``, deposits at the collision site
    because electrons are not transported and bremsstrahlung is neglected. The
    positron annihilates at rest and emits two 511 keV photons, which *are*
    transported: they carry the deep-penetration buildup that treating pair
    production as pure absorption would discard.
    """
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    E = particle["E"]

    # The pair's kinetic energy deposits locally; the two annihilation photons
    # deposit their energy at their own later collision sites.
    interaction_data["energy_deposition"] += (E - PAIR_PRODUCTION_THRESHOLD) * particle[
        "w"
    ]

    # Revive the current history in place as the first annihilation photon.
    ux, uy, uz = sample_isotropic_direction(particle_container)
    particle["E"] = ELECTRON_MASS
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz

    # Bank the second annihilation photon, emitted back to back with the first
    # and inheriting the current weight.
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_module.copy_as_child(particle_container_new, particle_container)
    particle_new = particle_container_new[0]
    particle_new["E"] = ELECTRON_MASS
    particle_new["ux"] = -ux
    particle_new["uy"] = -uy
    particle_new["uz"] = -uz
    particle_bank_module.bank_active_particle(particle_container_new, program)
