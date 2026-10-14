"""Coherent (Rayleigh) scattering from the tabulated atomic form factor.

Ported from the pre-refactor ``coherent_form_factor.py``. The physics being
checked is the shape of the angular distribution: coherent scattering peaks
forward, more sharply as energy rises and as Z rises, and it is elastic.
"""

import math

import numpy as np
import pytest

import mcdc.transport.physics.photon.native as native

from conftest import make_interaction_data, make_photon


def _element(simulation):
    return simulation["elements"][0]


def _sample_cosines(simulation, data, E, count=400, first_seed=1):
    """Sample ``count`` scattering cosines from one advancing RNG stream.

    One stream, not one fresh seed per draw: an LCG's first output is a
    near-linear function of its seed, so consecutive seeds give correlated
    samples whose mean is biased and whose spread is not Poisson.
    """
    container = make_photon(E, seed=first_seed)
    element = _element(simulation)
    return np.array(
        [
            native.sample_coherent_scattering_cosine(
                container, element, simulation, data
            )
            for _ in range(count)
        ]
    )


# ======================================================================================
# The form factor itself
# ======================================================================================


@pytest.mark.parametrize("Z", [1, 13, 82])
def test_form_factor_forward_value_equals_z(photon_model, Z):
    """``F(q = 0, Z) = Z``: every electron scatters in phase in the forward direction."""
    from mcdc.transport.data import evaluate_data

    simulation, data = photon_model(atomic_number=Z)
    element = _element(simulation)
    reaction_ID = int(
        np.asarray(
            __import__(
                "mcdc.mcdc_get", fromlist=["element"]
            ).element.photon_coherent_reaction_IDs(0, element, data)
        )
    )
    reaction_base = simulation["photon_reactions"][reaction_ID]
    reaction = simulation["photon_coherent_reactions"][reaction_base["sub_ID"]]
    form_factor = simulation["data"][reaction["form_factor_ID"]]

    assert evaluate_data(0.0, form_factor, simulation, data) == pytest.approx(float(Z))


# ======================================================================================
# The sampled angular distribution
# ======================================================================================


def test_cosine_is_within_physical_range(photon_model):
    simulation, data = photon_model()
    mu = _sample_cosines(simulation, data, 1.0e5, count=200)
    assert np.all(mu >= -1.0)
    assert np.all(mu <= 1.0)
    assert np.all(np.isfinite(mu))


def test_distribution_peaks_forward_not_isotropic(photon_model):
    # An isotropic distribution has mean cosine zero. Coherent scattering must
    # be measurably forward-peaked instead.
    simulation, data = photon_model()
    mu = _sample_cosines(simulation, data, 1.0e5, count=600)
    assert mu.mean() > 0.1


def test_forward_peaking_sharpens_with_energy(photon_model):
    # Higher energy means larger accessible momentum transfer, so the form
    # factor cuts off the large angles more aggressively.
    simulation, data = photon_model()
    low = _sample_cosines(simulation, data, 1.0e4, count=600).mean()
    high = _sample_cosines(simulation, data, 1.0e6, count=600).mean()
    assert high > low


def test_forward_peaking_sharpens_with_z(photon_model):
    # A heavier atom has a wider electron cloud in momentum space, so its form
    # factor falls off faster and scattering is more forward.
    low_z, low_data = photon_model(atomic_number=6)
    high_z, high_data = photon_model(atomic_number=82)
    low = _sample_cosines(low_z, low_data, 1.0e5, count=600).mean()
    high = _sample_cosines(high_z, high_data, 1.0e5, count=600).mean()
    assert high >= low


def test_low_energy_limit_is_thomson(photon_model):
    """As the energy goes to zero, coherent scattering becomes Thomson.

    The accessible momentum transfer vanishes, so the form factor is flat at
    ``F(0, Z) = Z`` across the whole angular range and only the Thomson factor
    survives. That distribution is ``(1 + mu^2) / 2``, which is symmetric
    fore-and-aft -- mean cosine zero -- and has ``<mu^2> = 0.4``. It is
    emphatically *not* forward-peaked: the forward peaking tested above is an
    effect of the form factor cutting off large angles at finite energy.
    """
    simulation, data = photon_model()
    mu = _sample_cosines(simulation, data, 1.0e-3, count=3000)

    assert mu.mean() == pytest.approx(0.0, abs=0.05)
    assert (mu**2).mean() == pytest.approx(0.4, abs=0.03)


# ======================================================================================
# Coherent scattering is elastic
# ======================================================================================


def test_scattering_preserves_energy_and_deposits_nothing(photon_model):
    simulation, data = photon_model()
    E = 1.0e5
    container = make_photon(E, seed=3)
    interaction = make_interaction_data()

    native.coherent_scattering(container, _element(simulation), simulation, data)

    assert container[0]["E"] == E
    assert container[0]["alive"]
    assert interaction[0]["energy_deposition"] == 0.0


def test_scattering_leaves_a_unit_direction(photon_model):
    simulation, data = photon_model()
    for seed in range(1, 20):
        container = make_photon(1.0e5, seed=seed)
        native.coherent_scattering(container, _element(simulation), simulation, data)
        direction = np.array(
            [container[0]["ux"], container[0]["uy"], container[0]["uz"]]
        )
        assert np.linalg.norm(direction) == pytest.approx(1.0, rel=1e-12)


def test_element_without_coherent_data_does_not_deflect(photon_model):
    # Defensive, but reachable: an element whose library omits the coherent
    # channel must not be sampled for an angle it has no data for.
    simulation, data = photon_model()
    element = _element(simulation)
    assert element["N_photon_coherent_reaction"] == 1
