import numpy as np
import pytest

import mcdc.transport.distribution as dist
from mcdc.object_.distribution import DistributionMultiTable


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("grid", [[1.0], [1.0, 3.0]])
def test_multi_table_distribution_first_grid_point(
    grid, scale, mock_rng_sequence, prepare_simulation
):
    # At E = grid[0], the first table [10, 20] is sampled directly with xi = 0.25.
    distribution = DistributionMultiTable(
        grid=grid,
        value=[[10.0, 20.0], [100.0, 200.0]][: len(grid)],
        cdf=[[0.0, 1.0], [0.0, 1.0]][: len(grid)],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    multi_table = simulation["multi_table_distributions"][distribution.sub_ID]
    mock_rng = mock_rng_sequence(0.25, 0.75)

    sampled_E = dist._sample_multi_table(
        1.0, mock_rng, multi_table, simulation, data, scale
    )

    np.testing.assert_allclose(sampled_E, 12.5, rtol=0.0, atol=1e-12)


def test_multi_table_distribution_sample(mock_rng_sequence, prepare_simulation):
    # MCNP Theory & User Manual §2.4.3.5.4.4 (Law 4: Tabular Distribution)
    distribution = DistributionMultiTable(
        grid=[1.0, 3.0],
        value=[[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
        cdf=[[0.0, 0.5, 1.0], [0.0, 0.6, 1.0]],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    multi_table = simulation["multi_table_distributions"][distribution.sub_ID]

    # For E_in = 2.0 on the grid [1, 3], Eq. (2.62) gives r = 0.5.
    # xi_1 = 0.3 < r, so Eq. (2.64) selects l = i + 1, i.e. the second table.
    xi1, xi2 = 0.3, 0.2
    mock_rng = mock_rng_sequence(xi1, xi2)

    sampled_E = dist._sample_multi_table(
        2.0,
        mock_rng,
        multi_table,
        simulation,
        data,
        scale=True,
    )

    # In the selected table, xi_2 = 0.2 falls in the first continuous bin.
    # The CDF rises from 0.0 to 0.6 over [100, 200], giving p = 0.006.
    E_prime = 100.0 + (xi2 - 0.0) / 0.006
    # Eq. (2.67) and Eq. (2.68) give the scaled bounds:
    #   E_1 = 10 + 0.5 * (100 - 10) = 55
    #   E_K = 30 + 0.5 * (300 - 30) = 165
    # Here E_l,1 = 100 and E_l,K = 300 because the selected table is the second one.
    # Eq. (2.69) then gives
    #   E_out = E_1 + (E' - E_l,1) * (E_K - E_1) / (E_l,K - E_l,1)
    expected_E = 55.0 + (E_prime - 100.0) * (165.0 - 55.0) / (300.0 - 100.0)

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)


def test_incident_regions_preserve_child_pdf_interpolation(
    mock_rng_sequence, prepare_simulation
):
    # Incident interpolation changes from histogram to linear at the middle point.
    # Every child has a linear, rising density with a known quantile at xi = 0.25.
    distribution = DistributionMultiTable(
        grid=[1.0, 3.0, 5.0],
        value=[[10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0]],
        pdf=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        interpolations=[1, 2],
        interpolation_boundaries=[2, 3],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    multi_table = simulation["multi_table_distributions"][distribution.sub_ID]

    for energy, expected in [(2.0, 15.0), (4.0, 825.0)]:
        sampled = dist._sample_multi_table(
            energy, mock_rng_sequence(0.1, 0.25), multi_table, simulation, data, True
        )
        assert sampled == pytest.approx(expected)
