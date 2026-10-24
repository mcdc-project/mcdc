"""Continuous and discrete components share one total probability measure."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from mcdc.object_.distribution import DistributionTable, DistributionMultiTable
from mcdc.object_.hdf5_reader import read_distribution
from mcdc.transport.distribution import sample_table, _sample_multi_table

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from data_library_generator.common.hdf5_writer import distribution_table, multi_table


def sample(distribution, xi, prepare_simulation, mock_rng_sequence):
    container, data = prepare_simulation(objects=[distribution])
    record = container[0]["table_distributions"][distribution.sub_ID]
    rng = mock_rng_sequence(xi)
    result = sample_table(record, rng, container[0], data)
    assert rng[0]["idx"] == 1
    return result


@pytest.mark.parametrize(
    "xi,expected", [(0, 2), (0.15, 2), (0.3, 0), (0.65, 5), (1, 10)]
)
def test_mixed_point_mass_and_uniform_density(
    xi, expected, prepare_simulation, mock_rng_sequence
):
    distribution = DistributionTable(
        discrete_value=[2], pmf=[0.3], value=[0, 10], pdf=[0.07, 0.07]
    )
    assert distribution.continuous_probability == pytest.approx(0.7)
    assert distribution.cmf[-1] == pytest.approx(0.3)
    assert sample(
        distribution, xi, prepare_simulation, mock_rng_sequence
    ) == pytest.approx(expected)


@pytest.mark.parametrize("xi,expected", [(0, 2), (0.3, 8), (1, 8)])
def test_discrete_only_skips_zero_masses(
    xi, expected, prepare_simulation, mock_rng_sequence
):
    distribution = DistributionTable(
        discrete_value=[-100, 2, 8, 100], pmf=[0, 0.3, 0.7, 0]
    )
    assert sample(distribution, xi, prepare_simulation, mock_rng_sequence) == expected


def test_joint_normalization_preserves_relative_weights():
    distribution = DistributionTable(
        discrete_value=[2], pmf=[3], value=[0, 10], pdf=[0.7, 0.7]
    )
    np.testing.assert_allclose(distribution.pmf, [0.3])
    np.testing.assert_allclose(distribution.pdf.y, [0.07, 0.07])
    np.testing.assert_allclose(distribution.pdf.aux[0], [0, 0.7])


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"discrete_value": [1]},
        {"pmf": [1]},
        {"value": [0, 1]},
        {"discrete_value": [1], "pmf": [-1]},
        {"discrete_value": [1], "pmf": [0]},
        {"discrete_value": [np.inf], "pmf": [1]},
        {"value": [0, 1], "pdf": [0, 0]},
        {"value": [0, 1], "pdf": [1, 1], "cdf": [0, 1]},
    ],
)
def test_invalid_probability_components(arguments):
    with pytest.raises(ValueError):
        DistributionTable(**arguments)


def test_mixed_hdf5_round_trip(tmp_path, prepare_simulation, mock_rng_sequence):
    with h5py.File(tmp_path / "mixed.h5", "w") as file:
        distribution_table(file, [0, 10], [0.07, 0.07], discrete_value=[2], pmf=[0.3])
        assert file["discrete-value"].attrs["unit"] == "eV"
        assert "unit" not in file["pmf"].attrs
        distribution = read_distribution(file)
    assert sample(
        distribution, 0.65, prepare_simulation, mock_rng_sequence
    ) == pytest.approx(5)


def test_packed_multi_table_has_no_child_distributions(
    tmp_path, prepare_simulation, mock_rng_sequence
):
    with h5py.File(tmp_path / "multi.h5", "w") as file:
        multi_table(
            file,
            [1, 3],
            [
                (
                    [1, 2, 2, 4],
                    np.array([1, 1, 2, 4]) / 7,
                    ["histogram", "log"],
                    [2, 4],
                ),
                ([1, 2], [1, 1], ["linear"], None),
            ],
        )
        distribution = read_distribution(file)
    assert isinstance(distribution, DistributionMultiTable)
    assert not hasattr(distribution, "tables")
    np.testing.assert_array_equal(distribution.offsets, [0, 4, 6])
    container, data = prepare_simulation(objects=[distribution])
    assert container[0]["N_table_distribution"] == 0
    record = container[0]["multi_table_distributions"][distribution.sub_ID]
    assert _sample_multi_table(
        1, mock_rng_sequence(0.5), record, container[0], data, False
    ) == pytest.approx(3)


def test_multi_table_rejects_incomplete_discrete_payload(tmp_path):
    with h5py.File(tmp_path / "discrete-multi.h5", "w") as file:
        multi_table(file, [1, 2], [([0, 1], [1, 1], None, None)] * 2)
        file["tables/pmf"] = [1]
        with pytest.raises(ValueError, match="incomplete packed discrete"):
            read_distribution(file)


def test_indexing_grid_jump_uses_right_hand_pdf(prepare_simulation, mock_rng_sequence):
    distribution = DistributionMultiTable(
        [1, 1, 3], value=[[0, 1], [10, 20], [100, 200]], pdf=[[1, 1]] * 3
    )
    container, data = prepare_simulation(objects=[distribution])
    record = container[0]["multi_table_distributions"][distribution.sub_ID]
    actual = _sample_multi_table(
        1, mock_rng_sequence(0.5), record, container[0], data, False
    )
    assert actual == pytest.approx(15)


def test_mixed_log_density_keeps_its_component_mass(
    prepare_simulation, mock_rng_sequence
):
    # Integral of 0.1*x on [1, 3] is 0.4; the atom carries the remaining 0.6.
    distribution = DistributionTable(
        [1, 3], pdf=[0.1, 0.3], discrete_value=[-2], pmf=[0.6], interpolations=5
    )
    actual = sample(distribution, 0.8, prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(np.sqrt(5))
