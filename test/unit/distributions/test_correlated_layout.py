"""Correlated samplers must use the lengths and offsets of compiled objects."""

import numpy as np
import pytest

from mcdc.object_.distribution import (
    DistributionKalbachMann,
    DistributionTableEnergyAngle,
)
from mcdc.transport.distribution import (
    sample_kalbach_mann,
    sample_table_energy_angle,
)


@pytest.mark.parametrize("energy", [1.0, 2.0, 3.0])
def test_kalbach_uses_flattened_outgoing_length(
    prepare_simulation, mock_rng_sequence, energy
):
    distribution = DistributionKalbachMann(
        [1, 3],
        [0, 3],
        [1, 2, 3, 2, 4, 6],
        [0.5] * 3 + [0.25] * 3,
        [0] * 6,
        [1] * 6,
    )
    container, data = prepare_simulation(objects=[distribution])
    record = container[0]["kalbach_mann_distributions"][distribution.sub_ID]
    state = mock_rng_sequence(0.3, 0.1, 0.7, 0.5)
    sampled, cosine = sample_kalbach_mann(energy, state, record, data)
    fraction = (energy - 1) / 2
    assert sampled == pytest.approx((1 + fraction) * 1.2)
    assert cosine == pytest.approx(0.0)


@pytest.mark.parametrize(
    "quantile, expected_cosine", [(0.1, -0.7), (0.4, 0.2), (0.9, 0.7)]
)
def test_energy_angle_uses_global_conditional_index(
    prepare_simulation, mock_rng_sequence, quantile, expected_cosine
):
    intervals = np.array([[-1, 1], [-1, 0], [0, 1], [-0.8, -0.4], [0.1, 0.5], [0.6, 1]])
    density = np.repeat(1 / np.diff(intervals)[:, 0], 2)
    distribution = DistributionTableEnergyAngle(
        [1, 3],
        [0, 3],
        [1, 2, 3, 2, 4, 6],
        [0.5] * 3 + [0.25] * 3,
        np.arange(6) * 2,
        intervals.ravel(),
        density,
    )
    container, data = prepare_simulation(objects=[distribution])
    record = container[0]["table_energy_angle_distributions"][distribution.sub_ID]
    state = mock_rng_sequence(0.3, quantile, 0.25)
    sampled, cosine = sample_table_energy_angle(2.0, state, record, data)
    assert sampled == pytest.approx(1.5 + 3 * quantile)
    assert cosine == pytest.approx(expected_cosine)
