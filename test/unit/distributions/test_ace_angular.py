"""ACE angular encodings preserve their probabilities in packed cosine tables."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ACEtk = pytest.importorskip("ACEtk")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from data_library_generator.common.ace_converter import write_angular_distribution
from mcdc.object_.distribution import (
    DistributionIsotropic,
    DistributionMultiTable,
    DistributionTable,
)
from mcdc.object_.hdf5_reader import read_distribution
from mcdc.transport.distribution import sample_distribution

# Unequal widths distinguish equal bin probabilities from a uniform cosine PDF.
BIN_EDGES = 2.0 * np.linspace(0.0, 1.0, 33) ** 2 - 1.0


def convert(tmp_path, source):
    with h5py.File(tmp_path / "angular.h5", "w") as group:
        write_angular_distribution(group, source)
        return read_distribution(group)


def pack(model, prepare_simulation):
    container, data = prepare_simulation(objects=[model])
    simulation = container[0]
    return simulation["distributions"][model.ID], simulation, data


def test_equiprobable_bins_preserve_each_bin_mass_and_quantiles(
    tmp_path, prepare_simulation, mock_rng_sequence
):
    edges = [BIN_EDGES, -BIN_EDGES[::-1]]
    source = ACEtk.continuous.AngularDistributionData(
        [
            ACEtk.continuous.EquiprobableAngularBins(energy, values.tolist())
            for energy, values in zip([1.0, 3.0], edges)
        ]
    )
    model = convert(tmp_path, source)
    assert isinstance(model, DistributionMultiTable)
    np.testing.assert_array_equal(model.grid, [1.0e6, 3.0e6])
    with h5py.File(tmp_path / "angular.h5") as group:
        np.testing.assert_array_equal(group["tables/offsets"], [0, 33, 66])
        assert group["tables/interpolation/laws"].asstr()[()].tolist() == [
            "histogram",
            "histogram",
        ]
        assert group["grid"].attrs["unit"] == "eV"
        assert "unit" not in group["tables/value"].attrs
        assert "unit" not in group["tables/pdf"].attrs
        for i, values in enumerate(edges):
            density = group["tables/pdf"][33 * i : 33 * (i + 1)]
            np.testing.assert_allclose(density[:-1] * np.diff(values), 1.0 / 32)
            np.testing.assert_allclose(
                model.cdf[33 * i : 33 * (i + 1)], np.arange(33) / 32
            )

    record, simulation, data = pack(model, prepare_simulation)
    for energy, values in zip([1.0e6, 3.0e6], edges):
        for xi in [0.0, 1.0, *((np.arange(32) + 0.25) / 32)]:
            index = min(int(32 * xi), 31)
            expected = values[index] + (32 * xi - index) * (
                values[index + 1] - values[index]
            )
            result = sample_distribution(
                energy, record, mock_rng_sequence(xi), simulation, data
            )
            assert result == pytest.approx(expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_angular_sequence_preserves_entries_and_incident_table_selection(
    tmp_path, reverse, prepare_simulation, mock_rng_sequence
):
    factories = [
        ACEtk.continuous.IsotropicAngularDistribution,
        lambda energy: ACEtk.continuous.EquiprobableAngularBins(
            energy, BIN_EDGES.tolist()
        ),
        lambda energy: ACEtk.continuous.TabulatedAngularDistribution(
            energy, 2, [0.0, 1.0], [0.0, 2.0], [0.0, 1.0]
        ),
    ]
    expected = [-0.5, BIN_EDGES[8], 0.5]
    laws = ["histogram", "histogram", "linear"]
    if reverse:
        factories.reverse()
        expected.reverse()
        laws.reverse()
    source = ACEtk.continuous.AngularDistributionData(
        [factory(energy) for factory, energy in zip(factories, [1.0, 3.0, 5.0])]
    )
    model = convert(tmp_path, source)
    with h5py.File(tmp_path / "angular.h5") as group:
        assert group.attrs["type"] == "multi-table"
        np.testing.assert_array_equal(group["tables/offsets"], [0, 2, 35, 37])
        assert group["tables/interpolation/laws"].asstr()[()].tolist() == laws
    record, simulation, data = pack(model, prepare_simulation)

    for energy, value in zip([1.0e6, 3.0e6, 5.0e6], expected):
        draws = (0.4, 0.25) if energy == 3.0e6 else (0.25,)
        result = sample_distribution(
            energy, record, mock_rng_sequence(*draws), simulation, data
        )
        assert result == pytest.approx(value)
    for lower, energy in enumerate([2.0e6, 4.0e6]):
        for selector, selected in [(0.75, lower), (0.25, lower + 1)]:
            result = sample_distribution(
                energy, record, mock_rng_sequence(selector, 0.25), simulation, data
            )
            assert result == pytest.approx(expected[selected])


def test_fully_isotropic_angular_record_keeps_compact_representation(tmp_path):
    model = convert(tmp_path, ACEtk.continuous.FullyIsotropicDistribution())
    assert isinstance(model, DistributionIsotropic)
    with h5py.File(tmp_path / "angular.h5") as group:
        assert group.attrs["type"] == "isotropic"
        assert not list(group)


@pytest.mark.parametrize("kind", ["isotropic", "equiprobable", "tabulated"])
def test_single_incident_entry_is_energy_independent(
    tmp_path, kind, prepare_simulation, mock_rng_sequence
):
    if kind == "isotropic":
        entry = ACEtk.continuous.IsotropicAngularDistribution(3.0)
        expected = -0.5
    elif kind == "equiprobable":
        entry = ACEtk.continuous.EquiprobableAngularBins(3.0, BIN_EDGES.tolist())
        expected = BIN_EDGES[8]
    else:
        entry = ACEtk.continuous.TabulatedAngularDistribution(
            3.0, 2, [0.0, 1.0], [0.0, 2.0], [0.0, 1.0]
        )
        expected = 0.5
    model = convert(tmp_path, ACEtk.continuous.AngularDistributionData([entry]))
    assert isinstance(
        model, DistributionIsotropic if kind == "isotropic" else DistributionTable
    )
    record, simulation, data = pack(model, prepare_simulation)
    for energy in [1.0e6, 3.0e6, 5.0e6]:
        result = sample_distribution(
            energy, record, mock_rng_sequence(0.25), simulation, data
        )
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "invalid", ["duplicate", "decreasing", "nan", "infinite", "out-of-range"]
)
def test_invalid_equiprobable_boundaries_are_rejected_before_writing(tmp_path, invalid):
    values = BIN_EDGES.copy()
    if invalid == "duplicate":
        values[10] = values[9]
    elif invalid == "decreasing":
        values[10] = values[9] - 0.01
    elif invalid == "nan":
        values[10] = np.nan
    elif invalid == "infinite":
        values[10] = np.inf
    else:
        values[-1] = 1.01
    source = ACEtk.continuous.AngularDistributionData(
        [
            ACEtk.continuous.EquiprobableAngularBins(1.0, values.tolist()),
            ACEtk.continuous.IsotropicAngularDistribution(3.0),
        ]
    )
    with h5py.File(tmp_path / "invalid.h5", "w") as group:
        with pytest.raises(ValueError, match="equiprobable bins require"):
            write_angular_distribution(group, source)
        assert not list(group)
