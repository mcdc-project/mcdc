"""Photon cross-section lookup, summation and dispatch.

Ported from the pre-refactor ``cross_sections.py``, which tested the same
physics against the hand-built flat-data buffer that the object model replaced.
"""

import numpy as np
import pytest

import mcdc.transport.physics.photon.native as native

from mcdc.constant import (
    PHOTON_REACTION_COHERENT_SCATTERING,
    PHOTON_REACTION_INCOHERENT_SCATTERING,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    PHOTON_REACTION_TOTAL,
)

from conftest import K_BINDING_ENERGY, make_photon

#: Above the K edge and the pair threshold, so every channel is open.
E_ALL_OPEN = 1.0e7

#: Above the K edge but below the pair threshold.
E_NO_PAIR = 1.0e6


def _macro(simulation, data, reaction_type, E):
    return native.macro_xs(reaction_type, make_photon(E), simulation, data)


def test_total_is_sum_of_four_channels(photon_model):
    simulation, data = photon_model()
    total = _macro(simulation, data, PHOTON_REACTION_TOTAL, E_ALL_OPEN)
    channels = sum(
        _macro(simulation, data, reaction_type, E_ALL_OPEN)
        for reaction_type in (
            PHOTON_REACTION_COHERENT_SCATTERING,
            PHOTON_REACTION_INCOHERENT_SCATTERING,
            PHOTON_REACTION_PHOTOELECTRIC,
            PHOTON_REACTION_PAIR_PRODUCTION,
        )
    )
    assert total == pytest.approx(channels, rel=1e-12)


def test_channels_match_the_library_values(photon_model):
    # Unit atom density, so the macroscopic cross section equals the
    # microscopic one and each channel must come back as written.
    simulation, data = photon_model(
        coherent=1.5, incoherent=2.5, photoelectric=3.5, pair=4.5
    )
    expected = {
        PHOTON_REACTION_COHERENT_SCATTERING: 1.5,
        PHOTON_REACTION_INCOHERENT_SCATTERING: 2.5,
        PHOTON_REACTION_PHOTOELECTRIC: 3.5,
        PHOTON_REACTION_PAIR_PRODUCTION: 4.5,
    }
    for reaction_type, value in expected.items():
        assert _macro(simulation, data, reaction_type, E_ALL_OPEN) == pytest.approx(
            value
        )


def test_pair_is_zero_below_threshold(photon_model):
    simulation, data = photon_model()
    assert _macro(simulation, data, PHOTON_REACTION_PAIR_PRODUCTION, E_NO_PAIR) == 0.0
    assert _macro(simulation, data, PHOTON_REACTION_PAIR_PRODUCTION, E_ALL_OPEN) > 0.0


# ======================================================================================
# Interpolation
# ======================================================================================


def test_interpolation_is_linear():
    # EPDL declares lin-lin (INT=2) for every photon cross section.
    value = native.interpolate_xs(2.0, 1.0, 4.0, 1.0, 16.0)
    assert value == pytest.approx(6.0, rel=1e-12)


def test_cross_section_is_finite_across_a_duplicated_edge(photon_model):
    # The grid repeats the K-edge energy. Evaluating either side must stay
    # finite and non-negative, which is what the zero-width guard buys.
    simulation, data = photon_model()
    for E in (
        K_BINDING_ENERGY * (1 - 1e-9),
        K_BINDING_ENERGY,
        K_BINDING_ENERGY * 1.001,
    ):
        value = _macro(simulation, data, PHOTON_REACTION_TOTAL, E)
        assert np.isfinite(value)
        assert value >= 0.0
