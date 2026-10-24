import math

import h5py
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betaincinv

import mcdc.transport.distribution as dist
from mcdc.object_.distribution import DistributionNBody
from mcdc.object_.hdf5_reader import read_distribution


def test_nbody_sample_correlated(mock_rng_sequence, prepare_simulation):
    """A normalized energy quantile is scaled by LAW 66's kinematic maximum."""
    distribution = DistributionNBody(
        3,
        10.0,
        values=[0.0, 0.5, 1.0],
        probabilities=[1.0, 1.0, 1.0],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    distribution_base = simulation["distributions"][distribution.ID]

    xi1, xi2 = 0.2, 0.75
    mock_rng = mock_rng_sequence(xi1, xi2)

    sampled_E, sampled_mu = dist.sample_correlated_distribution(
        20.0,
        distribution_base,
        mock_rng,
        simulation,
        data,
        atomic_weight_ratio=4.0,
        q_value=-2.0,
    )

    expected_E = xi1 * (9.0 / 10.0) * ((4.0 / 5.0) * 20.0 - 2.0)
    expected_mu = 2.0 * xi2 - 1.0

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sampled_mu, expected_mu, rtol=0.0, atol=1e-12)
    assert mock_rng[0]["idx"] == 2


@pytest.mark.parametrize(
    "law, name",
    [(1, "histogram"), (2, "linear"), (3, "semilog-x"), (4, "semilog-y"), (5, "log")],
)
@pytest.mark.parametrize("from_hdf5", [False, True])
def test_nbody_interpolation_quantiles(
    law, name, from_hdf5, tmp_path, mock_rng_sequence, prepare_simulation
):
    """Check every density law against independently integrated quantiles."""
    if from_hdf5:
        with h5py.File(tmp_path / "nbody.h5", "w") as group:
            group.attrs["type"] = "n-body"
            group["number-particles"] = 3
            group["total-mass-ratio"] = 10.0
            group["value"] = [0.25, 1.0]
            group["pdf"] = [0.2, 1.1]
            group.create_group("interpolation")
            group["interpolation/laws"] = np.asarray([name], dtype=h5py.string_dtype())
            group["interpolation/boundaries"] = [2]
            distribution = read_distribution(group)
    else:
        distribution = DistributionNBody(
            3, 10.0, [0.25, 1.0], [0.2, 1.1], interpolations=law
        )
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    base = simulation["distributions"][distribution.ID]

    def density(x):
        fraction = (
            math.log(x / 0.25) / math.log(4.0) if law in (3, 5) else (x - 0.25) / 0.75
        )
        if law == 1:
            return 0.2
        if law in (4, 5):
            return 0.2 * (1.1 / 0.2) ** fraction
        return 0.2 + fraction * 0.9

    integral = quad(density, 0.25, 1.0)[0]
    for probability in (0.01, 0.25, 0.9):
        expected = brentq(
            lambda x: quad(density, 0.25, x)[0] / integral - probability, 0.25, 1.0
        )
        energy, cosine = dist.sample_correlated_distribution(
            10.0,
            base,
            mock_rng_sequence(probability, 0.75),
            simulation,
            data,
            atomic_weight_ratio=4.0,
            q_value=-2.0,
        )
        assert energy == pytest.approx(expected * 0.9 * 6.0, abs=1e-10)
        assert cosine == pytest.approx(0.5)


def test_nbody_multiple_interpolation_regions(mock_rng_sequence, prepare_simulation):
    """A histogram interval and a logarithmic region share one normalized CDF."""
    distribution = DistributionNBody(
        3,
        10.0,
        [0.25, 0.5, 0.5, 1.0],
        [1.0, 1.0, 2.0, 4.0],
        interpolations=[1, 5],
        interpolation_boundaries=[2, 4],
    )
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    base = simulation["distributions"][distribution.ID]
    # The first region has mass 0.25; integrating p(x) = 4*x from 0.5 to 1 adds 1.5.
    for probability, expected in [(0.1, 0.425), (0.5, 0.75)]:
        energy, cosine = dist.sample_correlated_distribution(
            10.0,
            base,
            mock_rng_sequence(probability, 0.75),
            simulation,
            data,
            atomic_weight_ratio=4.0,
            q_value=-2.0,
        )
        assert energy == pytest.approx(expected * 0.9 * 6.0, abs=1e-10)
        assert cosine == pytest.approx(0.5)


@pytest.mark.parametrize("count", [3, 4, 5])
@pytest.mark.parametrize("incident,q_value", [(5.0e6, -2.0e6), (20.0e6, 1.0e6)])
def test_nbody_phase_space_shape_and_energy_scale(
    count, incident, q_value, mock_rng_sequence, prepare_simulation
):
    """Compare phase-space quantiles to the analytic beta distribution."""
    value = np.linspace(0, 1, 10001)
    density = np.sqrt(value) * (1 - value) ** (1.5 * count - 4)
    distribution = DistributionNBody(count, 12.0, value, density)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    base = simulation["distributions"][distribution.ID]
    maximum = (11.0 / 12.0) * ((12.0 / 13.0) * incident + q_value)
    for probability in (0.1, 0.5, 0.9):
        energy, cosine = dist.sample_correlated_distribution_with_scale(
            incident,
            base,
            mock_rng_sequence(probability, 0.75),
            simulation,
            data,
            atomic_weight_ratio=12.0,
            q_value=q_value,
        )
        fraction = betaincinv(1.5, 1.5 * count - 3, probability)
        assert energy == pytest.approx(maximum * fraction, rel=2e-5)
        assert 0 <= energy <= maximum
        assert cosine == pytest.approx(0.5)


@pytest.mark.parametrize("incident", [0.0, 2.0, 2.5])
def test_nbody_zero_available_energy(incident, mock_rng_sequence, prepare_simulation):
    distribution = DistributionNBody(3, 10.0, [0, 1], [1, 1])
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    rng = mock_rng_sequence(0.9, 0.25)
    energy, cosine = dist.sample_correlated_distribution(
        incident,
        simulation["distributions"][distribution.ID],
        rng,
        simulation,
        data,
        atomic_weight_ratio=4.0,
        q_value=-2.0,
    )
    assert energy == 0
    assert cosine == -0.5
    assert rng[0]["idx"] == 2


@pytest.mark.parametrize(
    "count,mass,values",
    [
        (2, 10, [0, 1]),
        (3.5, 10, [0, 1]),
        (3, 1, [0, 1]),
        (3, np.inf, [0, 1]),
        (3, np.nan, [0, 1]),
        (3, 10, [-0.1, 1]),
        (3, 10, [0, 1.1]),
    ],
)
def test_nbody_rejects_invalid_parameters(count, mass, values):
    with pytest.raises(ValueError):
        DistributionNBody(count, mass, values, [1, 1])


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"atomic_weight_ratio": 4.0},
        {"q_value": 0.0},
        {"atomic_weight_ratio": 0.0, "q_value": 0.0},
        {"atomic_weight_ratio": np.inf, "q_value": 0.0},
        {"atomic_weight_ratio": 4.0, "q_value": np.nan},
    ],
)
def test_nbody_requires_finite_reaction_context(
    context, mock_rng_sequence, prepare_simulation
):
    distribution = DistributionNBody(3, 10.0, [0, 1], [1, 1])
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    with pytest.raises(ValueError):
        dist.sample_correlated_distribution(
            10.0,
            simulation["distributions"][distribution.ID],
            mock_rng_sequence(),
            simulation,
            data,
            **context,
        )
