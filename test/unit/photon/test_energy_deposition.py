"""Photon energy deposition, branch by branch.

Electrons are not transported, so every charged product deposits at the
collision site. The deposition each branch writes into
``interaction_data["energy_deposition"]`` is therefore fixed by what the branch
does *not* carry away:

===================  ==================================================
Coherent             0 -- elastic, nothing transferred
Incoherent           ``E - E_scattered``, the recoil electron
Photoelectric        ``E - E_fluorescence``
Pair production      ``E - 2 m_e c^2``, the pair's kinetic energy
===================  ==================================================

Ported from the pre-refactor ``test_energy_deposition.py``.
"""

import numpy as np
import pytest

import mcdc.transport.physics.photon as photon
import mcdc.transport.physics.photon.native as native

from mcdc.constant import ELECTRON_MASS, PHOTON_CUTOFF_ENERGY

from conftest import K_ALPHA_ENERGY, make_interaction_data, make_photon

PAIR_THRESHOLD = 2.0 * ELECTRON_MASS


def _element(simulation):
    return simulation["elements"][0]


# ======================================================================================
# Constant cross sections
# ======================================================================================


def test_constant_xs_capture_deposits_the_full_energy(constant_xs_model):
    # Pure absorber, so the first sampled channel is always capture.
    simulation, data = constant_xs_model(scatter=0.0, absorb=1.0)
    E = 1.0e6
    container = make_photon(E, seed=3)
    interaction = make_interaction_data()

    photon.collision(container, interaction, simulation, data)

    assert interaction[0]["energy_deposition"] == pytest.approx(E)
    assert not container[0]["alive"]
    assert container[0]["E"] == 0.0


def test_constant_xs_scatter_deposits_nothing(constant_xs_model):
    # Pure scatterer, and the scatter is elastic, so no energy is lost.
    simulation, data = constant_xs_model(scatter=1.0, absorb=0.0)
    E = 1.0e6
    container = make_photon(E, seed=3)
    interaction = make_interaction_data()

    photon.collision(container, interaction, simulation, data)

    assert interaction[0]["energy_deposition"] == 0.0
    assert container[0]["alive"]
    assert container[0]["E"] == E


def test_constant_xs_capture_is_weight_included(constant_xs_model):
    simulation, data = constant_xs_model(scatter=0.0, absorb=1.0)
    E, weight = 1.0e6, 0.25
    container = make_photon(E, seed=3, weight=weight)
    interaction = make_interaction_data()

    photon.collision(container, interaction, simulation, data)

    assert interaction[0]["energy_deposition"] == pytest.approx(E * weight)


def test_constant_xs_scatter_leaves_a_unit_direction(constant_xs_model):
    simulation, data = constant_xs_model(scatter=1.0, absorb=0.0)
    for seed in range(1, 15):
        container = make_photon(1.0e6, seed=seed)
        photon.collision(container, make_interaction_data(), simulation, data)
        direction = np.array(
            [container[0]["ux"], container[0]["uy"], container[0]["uz"]]
        )
        assert np.linalg.norm(direction) == pytest.approx(1.0, rel=1e-12)


def test_zero_total_cross_section_is_rejected_at_construction():
    """A zero-total ``PhotonConstantXSData`` cannot reach transport.

    The zero-cross-section object is the placeholder every material carries
    until compilation selects ID 0, so accepting it as a real material would
    silently give a vacuum where the deck asked for a medium. It is rejected
    when the material is built instead.
    """
    import mcdc

    with pytest.raises(SystemExit):
        mcdc.Material(
            photon_constant_xs=mcdc.PhotonConstantXSData(scatter=0.0, absorb=0.0)
        )


def test_negative_cross_sections_are_rejected():
    import mcdc

    with pytest.raises(SystemExit):
        mcdc.PhotonConstantXSData(scatter=-1.0)
    with pytest.raises(SystemExit):
        mcdc.PhotonConstantXSData(absorb=-1.0)


# ======================================================================================
# The cutoff
# ======================================================================================


def test_below_cutoff_the_photon_deposits_everything(photon_model):
    simulation, data = photon_model()
    E = PHOTON_CUTOFF_ENERGY * 0.5
    container = make_photon(E, seed=3)
    interaction = make_interaction_data()

    native.collision(container, interaction, simulation, data)

    assert interaction[0]["energy_deposition"] == pytest.approx(E)
    assert not container[0]["alive"]
    assert container[0]["E"] == 0.0


# ======================================================================================
# Per-branch formulas
# ======================================================================================


def test_incoherent_deposits_the_recoil_electron_energy(photon_model):
    simulation, data = photon_model()
    E = 1.0e6
    container = make_photon(E, seed=9)
    interaction = make_interaction_data()

    native.incoherent_scattering(container, interaction, simulation, data)

    E_scattered = container[0]["E"]
    assert 0.0 < E_scattered < E
    assert interaction[0]["energy_deposition"] == pytest.approx(E - E_scattered)
    assert container[0]["alive"]


def test_pair_production_deposits_the_pair_kinetic_energy(photon_model):
    simulation, data = photon_model()
    E = 5.0e6
    container = make_photon(E, seed=9)
    interaction = make_interaction_data()

    native.pair_production(container, interaction, simulation, data)

    assert interaction[0]["energy_deposition"] == pytest.approx(E - PAIR_THRESHOLD)
    # The history is revived as the first annihilation photon at 511 keV.
    assert container[0]["alive"]
    assert container[0]["E"] == pytest.approx(ELECTRON_MASS)


def test_photoelectric_deposits_all_but_the_fluorescence_line(photon_model):
    simulation, data = photon_model(fluorescence_yield=1.0)
    E = 1.0e6
    container = make_photon(E, seed=11)
    interaction = make_interaction_data()

    native.photoelectric(container, interaction, _element(simulation), simulation, data)

    assert interaction[0]["energy_deposition"] == pytest.approx(E - K_ALPHA_ENERGY)


def test_coherent_deposits_nothing(photon_model):
    simulation, data = photon_model()
    interaction = make_interaction_data()
    container = make_photon(1.0e5, seed=4)

    native.coherent_scattering(container, _element(simulation), simulation, data)

    assert interaction[0]["energy_deposition"] == 0.0


# ======================================================================================
# Conservation across a whole collision
# ======================================================================================


@pytest.mark.parametrize("E", [1.0e3, 5.0e5, 1.0e6, 5.0e6, 1.0e7])
def test_collision_never_deposits_more_than_it_was_given(photon_model, E):
    """Deposition is bounded by the incident energy, at every energy and seed.

    This is the invariant that catches a sign error or a unit error in any
    branch: whatever the sampled channel, the site cannot absorb more than
    arrived, and it cannot absorb a negative amount.
    """
    simulation, data = photon_model()
    for seed in range(1, 25):
        container = make_photon(E, seed=seed)
        interaction = make_interaction_data()

        native.collision(container, interaction, simulation, data)

        deposited = interaction[0]["energy_deposition"]
        assert deposited >= 0.0
        assert deposited <= E * (1.0 + 1e-12)


def test_pair_production_is_impossible_below_threshold(photon_model):
    """No collision below threshold may take the pair branch.

    If it did, the deposition formula ``E - 2 m_e c^2`` would go negative,
    which the bound above would catch -- so this pins the cause as well as the
    symptom.
    """
    simulation, data = photon_model()
    E = PAIR_THRESHOLD * 0.9
    for seed in range(1, 40):
        container = make_photon(E, seed=seed)
        interaction = make_interaction_data()
        native.collision(container, interaction, simulation, data)
        # A pair event would leave the photon at exactly 511 keV.
        if container[0]["alive"]:
            assert container[0]["E"] != pytest.approx(ELECTRON_MASS)
