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

from mcdc.constant import ELECTRON_MASS

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

    native.pair_production(
        container, interaction, simulation["elements"][0], simulation, data
    )

    bank = _bank(simulation)
    size = int(bank["size"][0])
    banked = bank["particle_data"][size - 1].copy() if size else None
    return container[0], banked, interaction[0]["energy_deposition"], size


# ======================================================================================
# The annihilation pair
# ======================================================================================


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
