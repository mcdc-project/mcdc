import numpy as np
import pytest

import mcdc.transport.distribution as dist
from mcdc.object_.distribution import DistributionTabulated


@pytest.mark.parametrize(
    "probabilities",
    [{"cdf": [0.0, 0.4, 1.0]}, {"pdf": [1.0, 2.0, 1.0]}],
    ids=["cdf", "pdf"],
)
def test_tabulated_distribution_first_cdf_point(
    probabilities, mock_rng_sequence, prepare_simulation
):
    # xi = cdf[0] = 0 returns the first value; find_bin would return -1 here.
    distribution = DistributionTabulated(value=[1.0, 3.0, 7.0], **probabilities)
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    table = simulation["tabulated_distributions"][distribution.sub_ID]
    mock_rng = mock_rng_sequence(0.0)

    sampled = dist.sample_tabulated(table, mock_rng, simulation, data)

    assert sampled == 1.0


def test_tabulated_distribution_sample(mock_rng_sequence, prepare_simulation):
    # MCNP Theory & User Manual §2.4.3.5.4.4 (Law 4: Tabular Distribution)
    distribution = DistributionTabulated(
        value=[1.0, 3.0, 7.0],
        cdf=[0.0, 0.4, 1.0],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    table = simulation["tabulated_distributions"][distribution.sub_ID]

    xi1 = 0.2
    mock_rng = mock_rng_sequence(xi1)

    sampled_E = dist.sample_tabulated(table, mock_rng, simulation, data)

    # This is the single-table inverse-CDF interpolation used by the tabulated sampler:
    # xi_1 = 0.2 lies in the first bin, so linear interpolation between
    # (c_0, E_0) = (0.0, 1.0) and (c_1, E_1) = (0.4, 3.0) gives the expected value.
    expected_E = 1.0 + (xi1 - 0.0) * (3.0 - 1.0) / (0.4 - 0.0)

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)


def test_tabulated_distribution_returns_interval(
    mock_rng_sequence,
    prepare_simulation,
):
    distribution = DistributionTabulated(
        value=[1.0, 3.0, 7.0],
        cdf=[0.0, 0.4, 1.0],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    table = simulation["tabulated_distributions"][distribution.sub_ID]
    mock_rng = mock_rng_sequence(0.7)

    sampled, interval = dist.sample_tabulated_with_interval(
        table,
        mock_rng,
        simulation,
        data,
    )

    assert 3.0 <= sampled <= 7.0
    assert interval == 1
