"""Pair production: threshold, deposition balance, and the annihilation pair.

Above ``2 m_e c^2`` the photon converts to an electron-positron pair. The pair's
kinetic energy deposits at the collision site, because electrons are not
transported and bremsstrahlung is neglected, and the positron annihilates at
rest into two 511 keV photons which *are* transported -- they are the same
species, so they are banked directly rather than going through cross-species
production.

Those two annihilation photons carry the deep-penetration buildup that treating
pair production as pure absorption would discard, which is why their presence or
absence changes a 10 MeV deck's flux by tens of percent. These tests pin the
behaviour so that it cannot drift again unnoticed.
"""

import numpy as np
import pytest

import mcdc.transport.physics.photon.native as native

from mcdc.constant import (
    ELECTRON_MASS,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_TOTAL,
)

from conftest import make_interaction_data, make_photon

PAIR_THRESHOLD = 2.0 * ELECTRON_MASS


def _bank(simulation):
    """The active particle bank, which is where the second photon lands."""
    return simulation["bank_active"]


def _drain(simulation):
    """Empty the active bank and return nothing, so each call starts clean."""
    _bank(simulation)["size"][0] = 0


def _pair_event(simulation, data, E, seed=7, weight=1.0):
    """Run one pair-production event and return ``(particle, banked, deposited)``.

    The second annihilation photon goes into the real active bank -- the kernel
    is compiled, so the bank call cannot be monkeypatched -- and is read back
    out of it here.
    """
    _drain(simulation)
    container = make_photon(E, seed=seed, weight=weight)
    interaction = make_interaction_data()

    native.pair_production(container, interaction, simulation, data)

    bank = _bank(simulation)
    size = int(bank["size"][0])
    banked = bank["particle_data"][size - 1].copy() if size else None
    return container[0], banked, interaction[0]["energy_deposition"], size


# ======================================================================================
# Threshold
# ======================================================================================


def test_threshold_is_twice_the_electron_mass():
    assert native.PAIR_PRODUCTION_THRESHOLD == pytest.approx(1021997.90138, rel=1e-9)


def test_cross_section_is_zero_below_threshold(photon_model):
    simulation, data = photon_model()
    for E in (1.0e5, PAIR_THRESHOLD * 0.999):
        assert (
            native.macro_xs(
                PHOTON_REACTION_PAIR_PRODUCTION, make_photon(E), simulation, data
            )
            == 0.0
        )


def test_cross_section_is_positive_above_threshold(photon_model):
    simulation, data = photon_model()
    assert (
        native.macro_xs(
            PHOTON_REACTION_PAIR_PRODUCTION, make_photon(1.0e7), simulation, data
        )
        > 0.0
    )


# ======================================================================================
# The annihilation pair
# ======================================================================================


def test_both_annihilation_photons_are_at_the_electron_rest_mass(photon_model):
    """511 keV each, exactly -- a positron annihilating at rest."""
    simulation, data = photon_model()
    container = make_photon(5.0e6, seed=7)
    interaction = make_interaction_data()

    native.pair_production(container, interaction, simulation, data)

    assert container[0]["E"] == pytest.approx(ELECTRON_MASS, rel=1e-12)
    assert container[0]["alive"]


def test_the_second_photon_is_banked_back_to_back(photon_model):
    """Momentum conservation at rest: the two photons leave in opposite directions."""
    simulation, data = photon_model()
    particle, banked, _, size = _pair_event(simulation, data, 5.0e6)

    assert size == 1
    assert banked["E"] == pytest.approx(ELECTRON_MASS, rel=1e-12)

    first = np.array([particle["ux"], particle["uy"], particle["uz"]])
    second = np.array([banked["ux"], banked["uy"], banked["uz"]])
    assert np.allclose(second, -first, rtol=1e-12)
    assert np.linalg.norm(second) == pytest.approx(1.0, rel=1e-12)


def test_the_banked_photon_inherits_the_weight(photon_model):
    simulation, data = photon_model()
    weight = 0.375
    particle, banked, _, _ = _pair_event(simulation, data, 5.0e6, weight=weight)

    assert banked["w"] == pytest.approx(weight)
    assert particle["w"] == pytest.approx(weight)


def test_the_banked_photon_starts_at_the_collision_site(photon_model):
    simulation, data = photon_model()
    particle, banked, _, _ = _pair_event(simulation, data, 5.0e6)
    for axis in ("x", "y", "z"):
        assert banked[axis] == pytest.approx(particle[axis])


# ======================================================================================
# Energy balance
# ======================================================================================


@pytest.mark.parametrize("E", [1.5e6, 5.0e6, 1.0e7, 1.0e8])
def test_energy_is_conserved_exactly(photon_model, E):
    """Deposited + both annihilation photons = the incident energy.

    The deposited part is the pair's kinetic energy, ``E - 2 m_e c^2``; the two
    photons carry ``2 m_e c^2`` back out. Nothing is created or lost.
    """
    simulation, data = photon_model()
    particle, banked, deposited, _ = _pair_event(simulation, data, E)

    carried = particle["E"] + banked["E"]
    assert deposited == pytest.approx(E - PAIR_THRESHOLD, rel=1e-12)
    assert deposited + carried == pytest.approx(E, rel=1e-12)


def test_deposition_is_weight_included(photon_model):
    simulation, data = photon_model()
    E, weight = 5.0e6, 0.25
    _, _, deposited, _ = _pair_event(simulation, data, E, weight=weight)
    assert deposited == pytest.approx((E - PAIR_THRESHOLD) * weight)


def test_deposition_is_never_negative_just_above_threshold(photon_model):
    # At E just above 2 m_e c^2 the pair has almost no kinetic energy, so the
    # deposited amount approaches zero from above and must not cross it.
    simulation, data = photon_model()
    _, _, deposited, _ = _pair_event(simulation, data, PAIR_THRESHOLD * 1.000001)
    assert deposited >= 0.0


# ======================================================================================
# Branching rate
# ======================================================================================


def test_pair_production_is_sampled_at_its_cross_section_fraction(photon_model):
    """The branch must be taken with probability ``sigma_pair / sigma_total``.

    This is the check that a correct cross section and a correct per-event
    treatment would both pass while the *rate* was still wrong -- which is the
    one way pair production could silently change a deck's flux without any of
    the other tests noticing.

    Detected by the signature a pair event leaves: the photon survives at
    exactly 511 keV.
    """
    simulation, data = photon_model(
        coherent=1.0, incoherent=1.0, photoelectric=1.0, pair=1.0
    )
    E = 1.0e7

    sigma_pair = native.macro_xs(
        PHOTON_REACTION_PAIR_PRODUCTION, make_photon(E), simulation, data
    )
    sigma_total = native.macro_xs(
        PHOTON_REACTION_TOTAL, make_photon(E), simulation, data
    )
    expected = sigma_pair / sigma_total

    trials = 4000
    pair_events = 0
    container = make_photon(E, seed=12345)
    for _ in range(trials):
        # One advancing RNG stream; reset only the particle state, and drain the
        # bank so the annihilation photons do not fill it.
        seed = container[0]["rng_seed"]
        container = make_photon(E)
        container[0]["rng_seed"] = seed
        interaction = make_interaction_data()
        _drain(simulation)
        native.collision(container, interaction, simulation, data)
        if (
            container[0]["alive"]
            and abs(container[0]["E"] - ELECTRON_MASS) < 1e-6 * ELECTRON_MASS
        ):
            pair_events += 1
    _drain(simulation)

    observed = pair_events / trials
    sigma = np.sqrt(expected * (1.0 - expected) / trials)

    assert expected == pytest.approx(0.25, rel=1e-9)
    assert abs(observed - expected) < 4.0 * sigma, (
        f"pair branch taken {observed:.4f} of the time, expected "
        f"{expected:.4f} +/- {sigma:.4f}"
    )
