"""Photon-electron coupling: the products each species banks for the other.

With both species transported, every charged product of a photon event and
every bremsstrahlung photon is banked instead of deposited. Each event must
still close its energy balance: the local deposit plus the energy the banked
products carry equals the energy the event made available.
"""

import numpy as np
import pytest

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.physics.electron.native as electron
import mcdc.transport.physics.photon.native as native

from mcdc.constant import (
    ELECTRON_MASS,
    PARTICLE_ELECTRON,
    PARTICLE_PHOTON,
    PARTICLE_POSITRON,
)

from conftest import (
    BREMSSTRAHLUNG_LOSS_FRACTION,
    K_ALPHA_ENERGY,
    K_BINDING_ENERGY,
    L_BINDING_ENERGY,
    make_interaction_data,
    make_photon,
)


def _banked(simulation):
    """Drain the active bank and return copies of what it held."""
    bank = simulation["bank_active"]
    size = int(bank["size"][0])
    banked = [bank["particle_data"][i].copy() for i in range(size)]
    bank["size"][0] = 0
    return banked


def _direction(particle):
    return np.array([particle["ux"], particle["uy"], particle["uz"]])


# ======================================================================================
# Photon -> electron
# ======================================================================================


def test_compton_recoil_electron_takes_the_lost_energy_and_momentum(coupled_model):
    """The recoil electron carries ``E - E'`` along ``p - p'``, with |p - p'| = p_e."""
    simulation, data = coupled_model()
    _banked(simulation)
    E = 1.0e6
    container = make_photon(E, seed=3)
    interaction = make_interaction_data()

    native.incoherent_scattering(container, interaction, simulation, data)

    (recoil,) = _banked(simulation)
    photon = container[0]
    T = recoil["E"]
    assert recoil["particle_type"] == PARTICLE_ELECTRON
    assert T == pytest.approx(E - photon["E"], rel=1e-12)
    assert interaction[0]["energy_deposition"] == pytest.approx(0.0, abs=1e-6)

    momentum_lost = E * np.array([0.0, 0.0, 1.0]) - photon["E"] * _direction(photon)
    assert np.linalg.norm(momentum_lost) == pytest.approx(
        np.sqrt(T * (T + 2.0 * ELECTRON_MASS)), rel=1e-9
    )
    assert np.allclose(
        _direction(recoil), momentum_lost / np.linalg.norm(momentum_lost), atol=1e-12
    )


def test_photoelectric_banks_the_photoelectron_and_auger_electron(coupled_model):
    """A K vacancy that relaxes non-radiatively emits two electrons and no photon."""
    simulation, data = coupled_model(fluorescence_yield=0.0)
    _banked(simulation)
    E = 1.0e6
    container = make_photon(E, seed=5)
    interaction = make_interaction_data()

    native.photoelectric(
        container, interaction, simulation["elements"][0], simulation, data
    )

    banked = _banked(simulation)
    assert not container[0]["alive"]
    assert all(p["particle_type"] == PARTICLE_ELECTRON for p in banked)
    energies = sorted(p["E"] for p in banked)
    if len(energies) == 2:  # K shell: photoelectron plus Auger electron
        assert energies == pytest.approx([K_ALPHA_ENERGY, E - K_BINDING_ENERGY])
    else:  # L shell: no transitions, the photoelectron only
        assert energies == pytest.approx([E - L_BINDING_ENERGY])
    deposited = interaction[0]["energy_deposition"]
    assert deposited >= 0.0
    assert deposited + sum(energies) == pytest.approx(E, rel=1e-12)


def test_pair_production_banks_the_electron_and_positron(coupled_model):
    """Deposit + electron + positron + rest masses = the incident energy."""
    simulation, data = coupled_model()
    _banked(simulation)
    E = 1.0e7
    container = make_photon(E, seed=7)
    interaction = make_interaction_data()

    native.pair_production(
        container, interaction, simulation["elements"][0], simulation, data
    )

    # The photon is gone; the positron annihilates where it stops
    assert not container[0]["alive"]
    electron_, positron = sorted(_banked(simulation), key=lambda p: p["particle_type"])
    assert electron_["particle_type"] == PARTICLE_ELECTRON
    assert positron["particle_type"] == PARTICLE_POSITRON
    T = electron_["E"] + positron["E"]
    assert interaction[0]["energy_deposition"] == pytest.approx(0.0, abs=1e-6)
    assert T == pytest.approx(E - 2.0 * ELECTRON_MASS, rel=1e-12)

    # Both leave forward, on opposite azimuths
    assert electron_["uz"] > 0.0 and positron["uz"] > 0.0
    assert electron_["ux"] * positron["ux"] + electron_["uy"] * positron["uy"] < 0.0


def test_stopped_positron_annihilates_into_two_back_to_back_photons(coupled_model):
    """The stopped positron's rest mass leaves as two 511 keV photons."""
    simulation, data = coupled_model()
    _banked(simulation)
    T = 50.0  # Below the electron cutoff
    container = make_photon(T, seed=9)
    container[0]["particle_type"] = PARTICLE_POSITRON
    interaction = make_interaction_data()

    electron.collision(container, interaction, simulation, data)

    assert not container[0]["alive"]
    first, second = _banked(simulation)
    assert first["particle_type"] == second["particle_type"] == PARTICLE_PHOTON
    assert first["E"] == second["E"] == pytest.approx(ELECTRON_MASS)
    assert np.allclose(_direction(first), -_direction(second))
    assert interaction[0]["energy_deposition"] == pytest.approx(T, rel=1e-9)


@pytest.mark.parametrize("E", [1.5e6, 1.0e7, 1.0e8])
def test_bethe_heitler_share_is_bounded_and_symmetric(E):
    """``eps`` stays in ``[m/E, 1 - m/E]`` and averages one half."""
    container = make_photon(E, seed=11)
    eps = np.array([native.sample_bethe_heitler(E, 82, container) for _ in range(4000)])
    eps0 = ELECTRON_MASS / E
    assert eps.min() >= eps0
    assert eps.max() <= 1.0 - eps0
    assert eps.mean() == pytest.approx(0.5, abs=0.02)


# ======================================================================================
# Electron -> photon
# ======================================================================================


def test_bremsstrahlung_banks_the_photon_along_the_electron(coupled_model):
    """The photon carries the tabulated mean loss; nothing is deposited for it."""
    simulation, data = coupled_model()
    _banked(simulation)
    element = simulation["elements"][0]
    reaction_ID = int(
        mcdc_get.element.electron_bremsstrahlung_reaction_IDs(0, element, data)
    )
    reaction = simulation["electron_reactions"][reaction_ID]

    E = 1.0e6
    container = make_photon(E, seed=13)
    container[0]["particle_type"] = PARTICLE_ELECTRON
    interaction = make_interaction_data()

    electron.sample_bremsstrahlung(reaction, container, interaction, simulation, data)

    (photon,) = _banked(simulation)
    assert photon["particle_type"] == PARTICLE_PHOTON
    assert photon["E"] == pytest.approx(BREMSSTRAHLUNG_LOSS_FRACTION * E, rel=1e-9)
    assert np.allclose(_direction(photon), _direction(container[0]))
    assert container[0]["E"] + photon["E"] == pytest.approx(E, rel=1e-12)
    assert interaction[0]["energy_deposition"] == pytest.approx(0.0, abs=1e-6)


def test_bremsstrahlung_photon_energy_follows_the_spectrum(coupled_model):
    """With a spectrum, k is sampled from it, and the electron loses k."""
    simulation, data = coupled_model(brems_spectrum=True)
    element = simulation["elements"][0]
    reaction_ID = int(
        mcdc_get.element.electron_bremsstrahlung_reaction_IDs(0, element, data)
    )
    reaction = simulation["electron_reactions"][reaction_ID]

    E = 1.0e6
    k = []
    for seed in range(1, 2001):
        _banked(simulation)
        container = make_photon(E, seed=seed)
        container[0]["particle_type"] = PARTICLE_ELECTRON
        interaction = make_interaction_data()
        electron.sample_bremsstrahlung(
            reaction, container, interaction, simulation, data
        )
        banked = _banked(simulation)
        k_i = banked[0]["E"] if banked else 0.0
        E_left = container[0]["E"] + interaction[0]["energy_deposition"]
        assert E_left + k_i == pytest.approx(E, rel=1e-12)
        k.append(k_i)

    # Uniform in [0, E]: mean E / 2, against the tabulated mean loss of 0.1 E
    k = np.array(k)
    assert k.max() <= E
    assert k.mean() == pytest.approx(0.5 * E, rel=0.05)
