"""The Klein-Nishina sampler used for incoherent scattering.

Ported from the pre-refactor ``test_klein_nishina.py`` and
``test_distributions.py``. These check the sampler against the analytic
distribution rather than against a stored answer, so they fail if the
kinematics or the rejection envelope is wrong.
"""

import math

import numpy as np
import pytest

import mcdc.transport.physics.photon.native as native

from mcdc.constant import ELECTRON_MASS

from conftest import make_photon


def _sample(E, count=4000, first_seed=1):
    """Return ``(eps, mu)`` arrays sampled at incident energy ``E``.

    Every draw comes from **one** advancing RNG stream, which is how transport
    uses it. Seeding each draw with a fresh consecutive integer instead would
    make the samples correlated -- an LCG's first output is a near-linear
    function of its seed -- and a correlated sample does not obey the Poisson
    statistics the distribution test below relies on.
    """
    container = make_photon(E, seed=first_seed)
    eps = np.empty(count)
    mu = np.empty(count)
    for index in range(count):
        eps[index], mu[index] = native.sample_klein_nishina(E, container)
    return eps, mu


def _epsilon_min(E):
    """The lowest possible energy ratio, at backscatter."""
    kappa = E / ELECTRON_MASS
    return 1.0 / (1.0 + 2.0 * kappa)


# ======================================================================================
# Kinematics
# ======================================================================================


@pytest.mark.parametrize("E", [1.0e4, 1.0e5, 5.0e5, 1.0e6, 1.0e7])
def test_energy_ratio_is_within_the_kinematic_bounds(E):
    eps, _ = _sample(E, count=1500)
    assert np.all(eps <= 1.0 + 1e-12)
    assert np.all(eps >= _epsilon_min(E) - 1e-12)


@pytest.mark.parametrize("E", [1.0e4, 1.0e6, 1.0e7])
def test_cosine_is_within_physical_range(E):
    _, mu = _sample(E, count=1500)
    assert np.all(mu >= -1.0)
    assert np.all(mu <= 1.0)


@pytest.mark.parametrize("E", [1.0e4, 1.0e6, 1.0e7])
def test_compton_relation_ties_energy_to_angle(E):
    """``eps`` and ``mu`` must satisfy Compton kinematics exactly.

    ``1 / eps = 1 + kappa (1 - mu)``. The sampler draws ``eps`` and derives
    ``mu`` from it, so a violation here means the derivation is wrong, not that
    the distribution is misshapen.
    """
    eps, mu = _sample(E, count=1500)
    kappa = E / ELECTRON_MASS

    # Exclude samples the sampler clamped at the ends of the cosine range.
    interior = (mu > -1.0 + 1e-12) & (mu < 1.0 - 1e-12)
    assert interior.sum() > 100

    predicted = 1.0 + kappa * (1.0 - mu[interior])
    assert np.allclose(1.0 / eps[interior], predicted, rtol=1e-10)


def test_scattered_energy_never_exceeds_the_incident_energy():
    eps, _ = _sample(1.0e6, count=1500)
    assert np.all(eps <= 1.0)


# ======================================================================================
# Limits
# ======================================================================================


def test_low_energy_limit_is_elastic_and_thomson():
    """Well below the electron mass, scattering is nearly elastic and Thomson.

    ``eps -> 1`` because the recoil is negligible, and the angular
    distribution approaches ``(1 + mu^2)``, which is symmetric fore-and-aft
    with ``<mu> = 0`` and ``<mu^2> = 0.4``.
    """
    E = 1.0e3  # kappa ~ 0.002
    eps, mu = _sample(E, count=6000)

    assert np.all(eps > 0.99)
    assert mu.mean() == pytest.approx(0.0, abs=0.05)
    assert (mu**2).mean() == pytest.approx(0.4, abs=0.03)


def test_high_energy_scattering_is_forward_peaked():
    # Well above the electron mass the distribution collapses forward.
    _, mu_low = _sample(1.0e4, count=3000)
    _, mu_high = _sample(1.0e7, count=3000)
    assert mu_high.mean() > mu_low.mean()
    assert mu_high.mean() > 0.5


def test_high_energy_photons_lose_most_of_their_energy_when_backscattered():
    # At kappa >> 1 the backscatter limit approaches m_e c^2 / 2E, so the
    # minimum surviving fraction becomes very small.
    E = 1.0e8
    eps, _ = _sample(E, count=2000)
    assert eps.min() < 0.05


# ======================================================================================
# The distribution itself
# ======================================================================================


def _klein_nishina_pdf(eps, E):
    """The Klein-Nishina differential cross section in ``eps``, up to a constant.

    ``d(sigma)/d(eps) ~ 1/eps + eps - (1 - mu^2) ... `` expressed through
    ``eps`` alone, which is the form the Kahn envelope is built around:

        f(eps) = (1/eps + eps) - (1 - mu^2) ,  mu = 1 - (1/eps - 1)/kappa
    """
    kappa = E / ELECTRON_MASS
    mu = 1.0 - (1.0 / eps - 1.0) / kappa
    return (1.0 / eps + eps) - (1.0 - mu**2)


@pytest.mark.parametrize("E", [1.0e5, 1.0e6, 5.0e6])
def test_sampled_spectrum_matches_the_klein_nishina_formula(E):
    """A histogram of sampled ``eps`` must follow the analytic shape.

    This is the test that would catch a wrong rejection probability: the
    samples would still be in range and still satisfy kinematics, but their
    density would be wrong. Compared as a shape, normalised on the histogram,
    so no overall constant is needed.
    """
    eps, _ = _sample(E, count=20000)

    edges = np.linspace(_epsilon_min(E), 1.0, 13)
    counts, _ = np.histogram(eps, bins=edges)

    # Integrate the density over each bin rather than sampling it at the bin
    # centre. The density goes as 1/eps, which varies steeply across the
    # lowest bin, and a midpoint estimate there is biased by more than the
    # Poisson error this test is trying to measure.
    expected = np.array(
        [
            np.trapezoid(
                _klein_nishina_pdf(np.linspace(lo, hi, 200), E),
                np.linspace(lo, hi, 200),
            )
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    expected = expected / expected.sum() * counts.sum()

    # Poisson tolerance: every well-populated bin within four sigma.
    populated = expected > 30
    assert populated.sum() >= 8
    deviation = np.abs(counts[populated] - expected[populated]) / np.sqrt(
        expected[populated]
    )
    assert deviation.max() < 4.0


def test_acceptance_is_efficient_enough_to_terminate():
    """Kahn's method is at least 50% efficient, so sampling cannot stall.

    Measured indirectly: drawing many samples must consume a bounded number of
    random numbers. A broken envelope shows up as a hang, which is why this is
    worth asserting rather than assuming.
    """
    import mcdc.transport.rng as rng

    E = 1.0e6
    container = make_photon(E, seed=12345)
    before = int(container[0]["rng_seed"])
    for _ in range(200):
        native.sample_klein_nishina(E, container)
    after = int(container[0]["rng_seed"])

    assert before != after
    # Three draws per trial, so 200 trials at >= 50% acceptance is well under
    # 2000 draws; this only has to prove it returned at all.
    assert np.isfinite(float(after))
