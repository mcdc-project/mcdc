"""Photoelectric atomic relaxation and characteristic X-ray emission.

These tests carry real weight: fluorescence photons are the only photoelectric
product that is transported -- every charged product deposits locally -- so this
is the only check on the one photoelectric branch that creates a particle.
Ported from the pre-refactor ``fluorescence.py``.
"""

import numpy as np
import pytest

import mcdc.transport.physics.photon.native as native

from conftest import (
    K_ALPHA_ENERGY,
    make_interaction_data,
    make_photon,
)

#: Above the K edge, so the K shell can be ionized.
E_ABOVE_K_EDGE = 1.0e6


def _element(simulation):
    return simulation["elements"][0]


def _sample(simulation, data, E, seed):
    """The primary line energy, as the single-photon sampler used to return."""
    container = make_photon(E, seed=seed)
    element = _element(simulation)
    subshell = native.sample_photoelectric_subshell(
        container, element, simulation, data
    )
    E_line, _, _ = native.relax_subshell(container, subshell, E, element, data)
    return E_line


def _drain(simulation):
    simulation["bank_active"]["size"][0] = 0


# ======================================================================================
# Deposition balance
# ======================================================================================


def test_photoelectric_with_a_line_revives_the_history(photon_model):
    simulation, data = photon_model(fluorescence_yield=1.0)
    container = make_photon(E_ABOVE_K_EDGE, seed=5)
    interaction = make_interaction_data()
    native.photoelectric(container, interaction, _element(simulation), simulation, data)

    assert container[0]["alive"]
    assert container[0]["E"] == pytest.approx(K_ALPHA_ENERGY)
    # Emitted isotropically, so the direction is renormalized but not preserved.
    direction = np.array([container[0]["ux"], container[0]["uy"], container[0]["uz"]])
    assert np.linalg.norm(direction) == pytest.approx(1.0)


# ======================================================================================
# The de-excitation cascade
#
# A radiative transition leaves a vacancy in the shell the filling electron came
# from, and that vacancy relaxes in turn. One further step is followed, so a
# photoelectric event emits up to two photons. The pre-refactor code did this and
# MCNP does it too (1.82 first-fluorescence and 0.38 second-fluorescence photons
# per source particle for 10 MeV photons in lead), so the port must as well.
# ======================================================================================


def test_the_cascade_is_energy_conserving(photon_model):
    """Deposited + both lines = the incident energy, weight included."""
    simulation, data = photon_model(cascade=True)
    _drain(simulation)
    weight = 0.25
    container = make_photon(E_ABOVE_K_EDGE, seed=5, weight=weight)
    interaction = make_interaction_data()

    native.photoelectric(container, interaction, _element(simulation), simulation, data)

    banked = simulation["bank_active"]["particle_data"][0]
    carried = (container[0]["E"] + banked["E"]) * weight
    deposited = interaction[0]["energy_deposition"]
    assert deposited + carried == pytest.approx(E_ABOVE_K_EDGE * weight)
    _drain(simulation)
