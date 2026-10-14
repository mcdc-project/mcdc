"""Photon cross-section lookup, summation and dispatch.

Ported from the pre-refactor ``cross_sections.py``, which tested the same
physics against the hand-built flat-data buffer that the object model replaced.
"""

import numpy as np
import pytest

import mcdc.transport.physics.photon.native as native
import mcdc.transport.physics.photon as photon

from mcdc.constant import (
    PHOTON_REACTION_COHERENT,
    PHOTON_REACTION_INCOHERENT,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    PHOTON_REACTION_TOTAL,
)

from conftest import K_BINDING_ENERGY, make_photon

#: Above the K edge and the pair threshold, so every channel is open.
E_ALL_OPEN = 1.0e7

#: Above the K edge but below the pair threshold.
E_NO_PAIR = 1.0e6

#: Below the K edge, so photoelectric is closed too.
E_SCATTER_ONLY = 1.0e3


def _macro(simulation, data, reaction_type, E):
    return native.macro_xs(reaction_type, make_photon(E), simulation, data)


def test_total_is_sum_of_four_channels(photon_model):
    simulation, data = photon_model()
    total = _macro(simulation, data, PHOTON_REACTION_TOTAL, E_ALL_OPEN)
    channels = sum(
        _macro(simulation, data, reaction_type, E_ALL_OPEN)
        for reaction_type in (
            PHOTON_REACTION_COHERENT,
            PHOTON_REACTION_INCOHERENT,
            PHOTON_REACTION_PHOTOELECTRIC,
            PHOTON_REACTION_PAIR_PRODUCTION,
        )
    )
    assert total == pytest.approx(channels, rel=1e-12)


def test_total_includes_coherent(photon_model):
    # Coherent scattering is a real channel, not an omission: dropping it
    # understates the total, which is how it was missed in an earlier version.
    simulation, data = photon_model(coherent=3.0)
    total = _macro(simulation, data, PHOTON_REACTION_TOTAL, E_ALL_OPEN)
    without_coherent = total - _macro(
        simulation, data, PHOTON_REACTION_COHERENT, E_ALL_OPEN
    )
    assert total > without_coherent
    assert _macro(simulation, data, PHOTON_REACTION_COHERENT, E_ALL_OPEN) == (
        pytest.approx(3.0)
    )


def test_channels_match_the_library_values(photon_model):
    # Unit atom density, so the macroscopic cross section equals the
    # microscopic one and each channel must come back as written.
    simulation, data = photon_model(
        coherent=1.5, incoherent=2.5, photoelectric=3.5, pair=4.5
    )
    expected = {
        PHOTON_REACTION_COHERENT: 1.5,
        PHOTON_REACTION_INCOHERENT: 2.5,
        PHOTON_REACTION_PHOTOELECTRIC: 3.5,
        PHOTON_REACTION_PAIR_PRODUCTION: 4.5,
    }
    for reaction_type, value in expected.items():
        assert _macro(simulation, data, reaction_type, E_ALL_OPEN) == pytest.approx(
            value
        )


def test_density_scales_the_macroscopic_cross_section(photon_model):
    simulation, data = photon_model(density=0.25, incoherent=2.0)
    assert _macro(simulation, data, PHOTON_REACTION_INCOHERENT, E_ALL_OPEN) == (
        pytest.approx(0.5)
    )


def test_pair_is_zero_below_threshold(photon_model):
    simulation, data = photon_model()
    assert _macro(simulation, data, PHOTON_REACTION_PAIR_PRODUCTION, E_NO_PAIR) == 0.0
    assert _macro(simulation, data, PHOTON_REACTION_PAIR_PRODUCTION, E_ALL_OPEN) > 0.0


def test_photoelectric_is_zero_below_the_edge(photon_model):
    simulation, data = photon_model()
    assert (
        _macro(simulation, data, PHOTON_REACTION_PHOTOELECTRIC, E_SCATTER_ONLY) == 0.0
    )
    assert _macro(simulation, data, PHOTON_REACTION_PHOTOELECTRIC, E_NO_PAIR) > 0.0


def test_scattering_survives_below_every_threshold(photon_model):
    # Coherent and incoherent have no threshold, so a low-energy photon still
    # interacts rather than streaming forever.
    simulation, data = photon_model()
    total = _macro(simulation, data, PHOTON_REACTION_TOTAL, E_SCATTER_ONLY)
    assert total > 0.0


def test_interface_dispatches_to_native(photon_model):
    simulation, data = photon_model()
    container = make_photon(E_ALL_OPEN)
    assert photon.macro_xs(
        PHOTON_REACTION_TOTAL, container, simulation, data
    ) == pytest.approx(
        native.macro_xs(PHOTON_REACTION_TOTAL, container, simulation, data)
    )


def test_unknown_reaction_type_returns_zero(photon_model):
    simulation, data = photon_model()
    assert _macro(simulation, data, 999, E_ALL_OPEN) == 0.0


# ======================================================================================
# Interpolation
# ======================================================================================


def test_log_log_interpolation_reproduces_a_power_law():
    # Two points on y = x^2 must interpolate exactly under log-log.
    value = native.log_log_interpolation(2.0, 1.0, 4.0, 1.0, 16.0)
    assert value == pytest.approx(4.0, rel=1e-12)


def test_log_log_interpolation_falls_back_across_a_zero():
    # A zero ordinate occurs below every absorption edge, where log is
    # undefined, so the fallback is linear rather than a crash.
    value = native.log_log_interpolation(1.5, 1.0, 2.0, 0.0, 10.0)
    assert value == pytest.approx(5.0)


def test_log_log_interpolation_handles_a_zero_width_bin():
    # A repeated abscissa is how an absorption edge is written; the upper value
    # is the one on the high-energy side of the discontinuity.
    value = native.log_log_interpolation(5.0, 5.0, 5.0, 1.0, 100.0)
    assert value == 100.0


def test_cross_section_is_continuous_across_a_duplicated_edge(photon_model):
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


# ======================================================================================
# Constant cross sections
# ======================================================================================


def test_constant_xs_is_energy_independent(constant_xs_model):
    simulation, data = constant_xs_model(scatter=0.9, absorb=0.1)
    values = [
        photon.macro_xs(PHOTON_REACTION_TOTAL, make_photon(E), simulation, data)
        for E in (1.0e2, 1.0e5, 1.0e8)
    ]
    assert values == pytest.approx([1.0, 1.0, 1.0])


def test_constant_xs_channel_mapping(constant_xs_model):
    simulation, data = constant_xs_model(scatter=0.7, absorb=0.3)
    container = make_photon(E_ALL_OPEN)

    def value(reaction_type):
        return photon.macro_xs(reaction_type, container, simulation, data)

    assert value(PHOTON_REACTION_TOTAL) == pytest.approx(1.0)
    assert value(PHOTON_REACTION_INCOHERENT) == pytest.approx(0.7)
    assert value(PHOTON_REACTION_PHOTOELECTRIC) == pytest.approx(0.3)
    # Neither channel has an analogue in the constant-XS treatment.
    assert value(PHOTON_REACTION_COHERENT) == 0.0
    assert value(PHOTON_REACTION_PAIR_PRODUCTION) == 0.0


def test_constant_xs_applicability_is_what_selects_the_treatment(
    constant_xs_model, photon_model
):
    import mcdc.transport.physics.photon.constant_xs as constant_xs

    simulation, data = constant_xs_model()
    assert constant_xs.applicable(make_photon(E_ALL_OPEN), simulation, data)

    simulation, data = photon_model()
    assert not constant_xs.applicable(make_photon(E_ALL_OPEN), simulation, data)


def test_photon_speed_is_the_speed_of_light(photon_model, constant_xs_model):
    from mcdc.constant import LIGHT_SPEED

    for build in (photon_model, constant_xs_model):
        simulation, data = build()
        for E in (1.0e2, 1.0e8):
            assert photon.particle_speed(
                make_photon(E), simulation, data
            ) == pytest.approx(LIGHT_SPEED)
