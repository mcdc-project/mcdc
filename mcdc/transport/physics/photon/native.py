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
    ELECTRON_CUTOFF_ENERGY,
    ELECTRON_MASS,
    LIGHT_SPEED,
    PARTICLE_ELECTRON,
    PARTICLE_POSITRON,
    PHOTON_CUTOFF_ENERGY,
    PHOTON_REACTION_COHERENT_SCATTERING,
    PHOTON_REACTION_INCOHERENT_SCATTERING,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    PHOTON_REACTION_TOTAL,
    PI,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.physics.cross_species_production import (
    emit_cross_species,
    is_transported,
)
from mcdc.transport.distribution import sample_isotropic_direction
from mcdc.transport.util import linear_interpolation
from mcdc.transport.physics.util import (
    evaluate_photon_xs_energy_grid,
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
# Interpolation
# ======================================================================================


@njit
def interpolate_xs(E, E0, E1, xs0, xs1):
    """Linearly interpolate a cross section, as EPDL declares (INT=2).

    The library repeats each absorption-edge energy so the discontinuity is
    represented exactly, which leaves a zero-width bin at every edge. The upper
    value is returned there, which is the value on the high-energy side.
    """
    if E1 <= E0:
        return xs1
    return linear_interpolation(E, E0, E1, xs0, xs1)


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
        xs = total_micro_xs(reaction_type, E, element, data)
        total += element_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, element, data):
    idx, E0, E1 = evaluate_photon_xs_energy_grid(E, element, data)
    return _total_micro_xs(reaction_type, E, idx, E0, E1, element, data)


@njit
def _total_micro_xs(reaction_type, E, idx, E0, E1, element, data):
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

    return interpolate_xs(E, E0, E1, xs0, xs1)


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
    idx = 0
    E0 = 0.0
    E1 = 0.0
    for i in range(material["N_element"]):
        element_ID = mcdc_get.material.element_IDs(i, material, data)
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.material.element_densities(i, material, data)
        idx, E0, E1 = evaluate_photon_xs_energy_grid(E, element, data)
        sigmaT = _total_micro_xs(PHOTON_REACTION_TOTAL, E, idx, E0, E1, element, data)

        total += element_density * sigmaT
        if total > xi:
            break

    # ==================================================================================
    # Sample and perform the reaction
    # ==================================================================================

    sigma_coherent = _total_micro_xs(
        PHOTON_REACTION_COHERENT_SCATTERING, E, idx, E0, E1, element, data
    )
    sigma_incoherent = _total_micro_xs(
        PHOTON_REACTION_INCOHERENT_SCATTERING, E, idx, E0, E1, element, data
    )
    sigma_photoelectric = _total_micro_xs(
        PHOTON_REACTION_PHOTOELECTRIC, E, idx, E0, E1, element, data
    )
    sigma_total = (
        sigma_coherent + sigma_incoherent + sigma_photoelectric
    ) + _total_micro_xs(PHOTON_REACTION_PAIR_PRODUCTION, E, idx, E0, E1, element, data)

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
        pair_production(
            particle_container, interaction_data_container, element, program, data
        )


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

    # The ID list holds base-array indices; the base record carries sub_ID into
    # the concrete subtype array.
    reaction_ID = int(
        mcdc_get.element.photon_coherent_scattering_reaction_IDs(0, element, data)
    )
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
    by Kahn's composition-rejection method. The recoil electron carries
    ``E (1 - eps)``: it is banked when electrons are transported, and otherwise
    its energy deposits at the collision site.
    """
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    E = particle["E"]
    eps, mu = sample_klein_nishina(E, particle_container)

    # The scattered photon carries away E * eps; the recoil electron takes the
    # rest, which deposits locally unless the electron is banked below.
    particle["E"] = E * eps
    interaction_data["energy_deposition"] += E * (1.0 - eps) * particle["w"]

    ux_in = particle["ux"]
    uy_in = particle["uy"]
    uz_in = particle["uz"]
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux, uy, uz = scatter_direction(ux_in, uy_in, uz_in, mu, azi)
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz

    # The recoil electron takes the momentum the photon lost, p_e = p - p'.
    T = E * (1.0 - eps)
    if T > ELECTRON_CUTOFF_ENERGY and is_transported(
        PARTICLE_ELECTRON, simulation["settings"]
    ):
        px = E * (ux_in - eps * ux)
        py = E * (uy_in - eps * uy)
        pz = E * (uz_in - eps * uz)
        norm = math.sqrt(px * px + py * py + pz * pz)
        emit_cross_species(
            PARTICLE_ELECTRON,
            T,
            px / norm,
            py / norm,
            pz / norm,
            particle_container,
            interaction_data_container,
            program,
        )


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

    The photoelectron carries ``E - B`` for the vacated subshell. The vacancy
    de-excites radiatively with probability equal to the shell's fluorescence
    yield, emitting a characteristic X-ray that is always transported because it
    is the same species; a non-radiative outcome emits an Auger electron. The
    photoelectron and Auger electrons are banked, isotropically, when electrons
    are transported, and otherwise deposit at the collision site.

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

    # Photon lines and Auger electrons from the first and second relaxation steps
    E_primary = 0.0
    E_cascade = 0.0
    E_auger_primary = 0.0
    E_auger_cascade = 0.0
    if subshell >= 0:
        E_emitted, origin, radiative = relax_subshell(
            particle_container, subshell, E, element, data
        )
        if radiative:
            E_primary = E_emitted
        else:
            E_auger_primary = E_emitted
        if E_primary > 0.0 and origin > 0:
            # The vacancy has moved to the subshell the filling electron left.
            next_subshell = find_subshell_by_designator(origin, element, data)
            if next_subshell >= 0:
                E_emitted, _, radiative = relax_subshell(
                    particle_container,
                    next_subshell,
                    E - E_primary,
                    element,
                    data,
                )
                if radiative:
                    E_cascade = E_emitted
                else:
                    E_auger_cascade = E_emitted

    # Everything the fluorescence photons do not carry away deposits locally,
    # less whatever the electrons banked below carry away.
    interaction_data["energy_deposition"] += (E - E_primary - E_cascade) * particle["w"]

    if subshell >= 0 and is_transported(PARTICLE_ELECTRON, simulation["settings"]):
        B = mcdc_get.element.photon_photoelectric_subshell_binding_energy(
            subshell, element, data
        )
        # Line and Auger energies come from the relaxation tables and can
        # slightly exceed this binding energy, so each electron takes at most
        # what is left, keeping the local deposit non-negative.
        E_left = E - E_primary - E_cascade
        T = min(E - B, E_left)
        emit_isotropic_electron(
            T, particle_container, interaction_data_container, program
        )
        E_left -= max(T, 0.0)
        T = min(E_auger_primary, E_left)
        emit_isotropic_electron(
            T, particle_container, interaction_data_container, program
        )
        E_left -= max(T, 0.0)
        T = min(E_auger_cascade, E_left)
        emit_isotropic_electron(
            T, particle_container, interaction_data_container, program
        )

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
def emit_isotropic_electron(T, particle_container, interaction_data_container, program):
    """Bank an electron of kinetic energy ``T`` emitted isotropically, if above cutoff."""
    if T <= ELECTRON_CUTOFF_ENERGY:
        return
    ux, uy, uz = sample_isotropic_direction(particle_container)
    emit_cross_species(
        PARTICLE_ELECTRON,
        T,
        ux,
        uy,
        uz,
        particle_container,
        interaction_data_container,
        program,
    )


@njit
def find_subshell_by_designator(designator, element, data):
    """Map an EADL subshell designator to this element's subshell index.

    Returns ``-1`` when the element carries no relaxation table for that
    subshell, which is how outer shells are written: their vacancy relaxes
    through unmodeled channels and so deposits locally.
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
        Subshell index, or ``-1`` when no subshell is accessible at this energy
        or the element carries no shell-resolved data.
    """
    particle = particle_container[0]
    E = particle["E"]

    N_reaction = element["N_photon_photoelectric_reaction"]
    if N_reaction == 0:
        return -1

    # The ID list holds base-array indices; the base record carries sub_ID into
    # the concrete subtype array.
    reaction_ID = int(
        mcdc_get.element.photon_photoelectric_reaction_IDs(0, element, data)
    )
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
        sigma_total += evaluate_data(E, subshell_xs, simulation, data)

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
        total += evaluate_data(E, subshell_xs, simulation, data)
        if total > xi:
            subshell = i
            break

    if subshell >= element["photon_relaxation_subshell_count_length"]:
        return -1
    return subshell


@njit
def relax_subshell(particle_container, subshell, E_available, element, data):
    """Draw one de-excitation transition for a vacancy in ``subshell``.

    A radiative transition emits a characteristic X-ray at the tabulated line
    energy; a non-radiative (Auger) transition emits an electron at the
    tabulated Auger energy, and the cascade is not followed past it.

    A line energy is always the difference between two binding energies and is
    therefore strictly below the binding energy of the shell that produced it,
    so an emitted photon can never re-ionize that shell. The cascade terminates
    by construction.

    Returns
    -------
    (float, int, bool)
        Emitted energy in eV -- ``0.0`` when no transition is accessible -- the
        EADL designator of the subshell the vacancy moves to, or ``-1`` when there
        is none to follow, and whether the emission is a photon (``True``) or an
        Auger electron (``False``).
    """
    if subshell < 0:
        return 0.0, -1, True
    if subshell >= int(element["photon_relaxation_subshell_count_length"]):
        return 0.0, -1, True

    N_transition = int(
        mcdc_get.element.photon_relaxation_subshell_count(subshell, element, data)
    )
    if N_transition == 0:
        return 0.0, -1, True

    start = int(
        mcdc_get.element.photon_relaxation_subshell_start(subshell, element, data)
    )

    # The probabilities are absolute, so they need not sum to one: whatever is
    # left over is an unmodeled channel, which deposits locally like Auger does.
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
            E_line = mcdc_get.element.photon_relaxation_transition_energy(
                start + i, element, data
            )
            if radiative == 0.0:
                # Non-radiative (Auger): the cascade is not followed further.
                if E_line <= 0.0 or E_line > E_available:
                    return 0.0, -1, False
                return E_line, -1, False

            if E_line <= 0.0 or E_line > E_available:
                return 0.0, -1, True
            origin = int(
                mcdc_get.element.photon_relaxation_transition_origin(
                    start + i, element, data
                )
            )
            return E_line, origin, True

    # Beyond the tabulated probability: unmodeled channel, deposit locally.
    return 0.0, -1, True


# ======================================================================================
# Pair production
# ======================================================================================


@njit
def pair_production(
    particle_container, interaction_data_container, element, program, data
):
    """Convert the photon to an electron-positron pair.

    The pair's kinetic energy, ``E - 2 m_e c^2``, deposits at the collision site.
    When electrons are transported, the electron and positron are banked
    instead, with the Bethe-Heitler split of that energy, and the positron
    annihilates into two 511 keV photons where it stops. Otherwise the positron
    annihilates at rest here; the annihilation photons are transported either
    way, since they carry the deep-penetration buildup that treating pair
    production as pure absorption would discard. Pair production in the
    electron field (triplet) is sampled the same way.
    """
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    interaction_data = interaction_data_container[0]

    E = particle["E"]

    # The pair's kinetic energy deposits locally, less the shares that are
    # banked; the two annihilation photons deposit their energy at their own
    # later collision sites.
    interaction_data["energy_deposition"] += (E - PAIR_PRODUCTION_THRESHOLD) * particle[
        "w"
    ]

    if is_transported(PARTICLE_ELECTRON, simulation["settings"]):
        eps = sample_bethe_heitler(E, element["atomic_number"], particle_container)
        T = eps * E - ELECTRON_MASS
        azi = 2.0 * PI * rng.lcg(particle_container)
        if T > ELECTRON_CUTOFF_ENERGY:
            # Fixed polar angle 1/gamma about the photon direction, as in EGS4.
            mu = math.cos(ELECTRON_MASS / (T + ELECTRON_MASS))
            ux, uy, uz = scatter_direction(
                particle["ux"], particle["uy"], particle["uz"], mu, azi
            )
            emit_cross_species(
                PARTICLE_ELECTRON,
                T,
                ux,
                uy,
                uz,
                particle_container,
                interaction_data_container,
                program,
            )

        # The positron takes the rest, on the opposite azimuth, and annihilates
        # where it stops.
        T_positron = (1.0 - eps) * E - ELECTRON_MASS
        if T_positron > ELECTRON_CUTOFF_ENERGY:
            mu = math.cos(ELECTRON_MASS / (T_positron + ELECTRON_MASS))
            ux, uy, uz = scatter_direction(
                particle["ux"], particle["uy"], particle["uz"], mu, azi + PI
            )
            emit_cross_species(
                PARTICLE_POSITRON,
                T_positron,
                ux,
                uy,
                uz,
                particle_container,
                interaction_data_container,
                program,
            )
            particle["alive"] = False
            particle["E"] = 0.0
            return

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


#: Below this photon energy the electron's share is sampled uniformly.
BETHE_HEITLER_LOW_ENERGY = 2.0e6

#: Above this photon energy the Coulomb correction is applied.
BETHE_HEITLER_COULOMB_ENERGY = 5.0e7


@njit
def sample_bethe_heitler(E, Z, particle_container):
    """Sample the electron's share of the photon energy in pair production.

    Returns ``eps``, the electron's total energy over ``E``, in
    ``[m c^2 / E, 1 - m c^2 / E]``. The screened Bethe-Heitler distribution with
    the Coulomb correction is sampled by the Butcher-Messel composition-rejection
    method, as in Geant4's G4BetheHeitlerModel; below 2 MeV, ``eps`` is sampled
    uniformly.
    """
    eps0 = ELECTRON_MASS / E

    if E < BETHE_HEITLER_LOW_ENERGY:
        eps = eps0 + (0.5 - eps0) * rng.lcg(particle_container)
    else:
        F_Z = 8.0 * math.log(Z) / 3.0
        if E > BETHE_HEITLER_COULOMB_ENERGY:
            a2 = (Z / 137.035999) ** 2
            F_Z += (
                8.0
                * a2
                * (
                    1.0 / (1.0 + a2)
                    + 0.20206
                    - 0.0369 * a2
                    + 0.0083 * a2 * a2
                    - 0.002 * a2 * a2 * a2
                )
            )

        screen_factor = 136.0 * eps0 / Z ** (1.0 / 3.0)
        screen_max = math.exp((42.24 - F_Z) / 8.368) - 0.952
        screen_min = min(4.0 * screen_factor, screen_max)
        eps1 = 0.5 - 0.5 * math.sqrt(1.0 - screen_min / screen_max)
        eps_min = max(eps0, eps1)
        eps_range = 0.5 - eps_min

        F10 = _screen_function_1(screen_min) - F_Z
        F20 = _screen_function_2(screen_min) - F_Z
        norm_F1 = max(F10 * eps_range * eps_range, 0.0)
        norm_F2 = max(1.5 * F20, 0.0)

        eps = 0.5
        while True:
            if norm_F1 / (norm_F1 + norm_F2) > rng.lcg(particle_container):
                eps = 0.5 - eps_range * rng.lcg(particle_container) ** (1.0 / 3.0)
                screen = screen_factor / (eps * (1.0 - eps))
                rejection = (_screen_function_1(screen) - F_Z) / F10
            else:
                eps = eps_min + eps_range * rng.lcg(particle_container)
                screen = screen_factor / (eps * (1.0 - eps))
                rejection = (_screen_function_2(screen) - F_Z) / F20
            if rejection >= rng.lcg(particle_container):
                break

    # The distribution is symmetric; the electron takes either share.
    if rng.lcg(particle_container) > 0.5:
        eps = 1.0 - eps
    return eps


@njit
def _screen_function_1(delta):
    if delta > 1.0:
        return 42.24 - 8.368 * math.log(delta + 0.952)
    return 42.392 - delta * (7.796 - 1.961 * delta)


@njit
def _screen_function_2(delta):
    if delta > 1.0:
        return 42.24 - 8.368 * math.log(delta + 0.952)
    return 41.405 - delta * (5.828 - 0.8945 * delta)
