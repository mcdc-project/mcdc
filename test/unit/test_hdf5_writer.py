"""Packed PDF tables preserve local interpolation regions and physical units."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from data_library_generator.common import hdf5_writer as writer


def test_packed_tables_round_trip_regions_and_discontinuities(tmp_path):
    tables = [
        ([0.0, 1.0, 1.0, 2.0], [0.2, 0.2, 0.8, 0.8], ["histogram", "linear"], [2, 4]),
        ([1.0, 3.0], [0.25, 0.75], ["log"], [2]),
        ([0.0, 0.5, 1.0], [0.0, 2.0, 0.0], None, None),
    ]
    path = tmp_path / "packed.h5"
    with h5py.File(path, "w") as file:
        writer.multi_table(
            file,
            [1.0, 1.0, 10.0],
            tables,
            laws=["histogram", "linear"],
            boundaries=[2, 3],
        )
    with h5py.File(path) as file:
        assert file.attrs["type"] == "multi-table"
        assert file["grid"].attrs["unit"] == "eV"
        np.testing.assert_array_equal(file["grid"], [1, 1, 10])
        np.testing.assert_array_equal(file["interpolation/boundaries"], [2, 3])
        packed = file["tables"]
        np.testing.assert_array_equal(packed["offsets"], [0, 4, 6, 9])
        np.testing.assert_array_equal(packed["interpolation/offsets"], [0, 2, 3, 4])
        assert packed["value"].attrs["unit"] == "eV"
        assert packed["pdf"].attrs["unit"] == "1/eV"
        assert "unit" not in packed["offsets"].attrs
        for i, (value, pdf, laws, boundaries) in enumerate(tables):
            start, end = packed["offsets"][i : i + 2]
            np.testing.assert_array_equal(packed["value"][start:end], value)
            np.testing.assert_array_equal(packed["pdf"][start:end], pdf)
            start, end = packed["interpolation/offsets"][i : i + 2]
            assert packed["interpolation/laws"].asstr()[start:end].tolist() == (
                laws or ["linear"]
            )
            np.testing.assert_array_equal(
                packed["interpolation/boundaries"][start:end],
                boundaries if boundaries is not None else [len(value)],
            )


def test_dimensionless_grid_and_cosines_have_no_unit(tmp_path):
    with h5py.File(tmp_path / "cosines.h5", "w") as file:
        writer.multi_table(
            file,
            [-1, 1],
            [([-1, 1], [0.5, 0.5], None, None)] * 2,
            grid_unit=None,
            unit=None,
        )
        for name in ("grid", "tables/value", "tables/pdf"):
            assert "unit" not in file[name].attrs
        writer.pdf_table(file.create_group("single"), [-1, 1], [0.5, 0.5], unit=None)
        assert file["single"].attrs["type"] == "table"


@pytest.mark.parametrize(
    "tables",
    [
        [],
        [([0, 1], [1], None, None)],
        [([0], [1], None, None)],
        [([0, 1, 2], [1, 1, 1], ["linear", "log"], [2, 1])],
    ],
)
def test_invalid_packed_tables_do_not_write_partial_datasets(tmp_path, tables):
    with h5py.File(tmp_path / "invalid.h5", "w") as file:
        with pytest.raises(ValueError):
            writer.pdf_tables(file, tables)
        assert not len(file)


@pytest.mark.parametrize("grid", [[1], [2, 1], [1, np.inf]])
def test_multi_table_rejects_invalid_indexing_grid(tmp_path, grid):
    with h5py.File(tmp_path / "invalid-grid.h5", "w") as file:
        with pytest.raises(ValueError):
            writer.multi_table(file, grid, [([0, 1], [1, 1], None, None)] * 2)


def test_exclusive_regions_cover_each_interval_once(tmp_path):
    with h5py.File(tmp_path / "regions.h5", "w") as file:
        writer.interpolation(file, 6, ["linear", "log"], [3, 6])
        laws = file["interpolation/laws"].asstr()[:]
        boundaries = file["interpolation/boundaries"][:]
        # An interval belongs to the first region whose exclusive end exceeds
        # its upper-point index; the regions share their boundary coordinate.
        regions = np.searchsorted(boundaries, np.arange(1, 6), side="right")
        assert laws[regions].tolist() == ["linear", "linear", "log", "log", "log"]
        single = file.create_group("single")
        writer.interpolation(single, 6)
        np.testing.assert_array_equal(single["interpolation/boundaries"], [6])


@pytest.mark.parametrize("boundaries", [[1, 6], [3, 5], [3, 3, 6], [3.0, 6.0], []])
def test_invalid_region_boundaries_are_rejected_before_writing(tmp_path, boundaries):
    with h5py.File(tmp_path / "invalid-regions.h5", "w") as file:
        with pytest.raises(ValueError):
            writer.interpolation(file, 6, ["linear"] * len(boundaries), boundaries)
        assert not len(file)


@pytest.mark.parametrize(
    "interpolants,boundaries,expected",
    [([2, 5], [3, 6], [3, 6]), ([], [], [6])],
)
def test_ace_regions_retain_source_point_counts(
    tmp_path, interpolants, boundaries, expected
):
    pytest.importorskip("ACEtk")
    from types import SimpleNamespace
    from data_library_generator.common.ace_converter import write_interpolation

    source = SimpleNamespace(interpolants=interpolants, boundaries=boundaries)
    with h5py.File(tmp_path / "ace-regions.h5", "w") as file:
        write_interpolation(file, source, 6)
        np.testing.assert_array_equal(file["interpolation/boundaries"], expected)


# ======================================================================================
# Probability validation shared by standalone and packed representations
# ======================================================================================


def write_probability_table(group, kind, value, pdf, laws=None, boundaries=None):
    """Exercise the same density through each public probability writer."""
    if kind == "table":
        writer.distribution_table(group, value, pdf, laws=laws, boundaries=boundaries)
    elif kind == "packed":
        writer.pdf_tables(group, [(value, pdf, laws, boundaries)])
    else:
        writer.multi_table(group, [1, 2], [(value, pdf, laws, boundaries)] * 2)


@pytest.mark.parametrize("kind", ["table", "packed", "multi-table"])
@pytest.mark.parametrize(
    "value,pdf,laws,boundaries",
    [
        ([0, np.inf], [1, 1], None, None),
        ([np.nan, 1], [1, 1], None, None),
        ([0, 1], [1, np.nan], None, None),
        ([0, 1], [1, np.inf], None, None),
        ([0, 1], [1, -1], None, None),
        ([1, 0], [1, 1], None, None),
        ([1, 1], [1, 1], None, None),
        ([0, 1], [0, 0], None, None),
        ([0, 1], [2, 2], None, None),
        ([0, 1], [1, 1], ["semilog-x"], None),
        ([-1, 1], [1, 1], ["log"], None),
        ([1, 2], [0, 1], ["semilog-y"], None),
        ([1, 2], [1, 0], ["log"], None),
        ([0, 1, 2], [1, 1, 1], ["linear", "log"], [2, 3.0]),
        ([0, 1, 2], [1, 1, 1], ["linear", "log"], [2, 2]),
    ],
)
def test_probability_writers_reject_invalid_tables_before_writing(
    tmp_path, kind, value, pdf, laws, boundaries
):
    with h5py.File(tmp_path / "invalid-probability.h5", "w") as file:
        with pytest.raises(ValueError):
            write_probability_table(file, kind, value, pdf, laws, boundaries)
        assert not len(file)
        assert not len(file.attrs)


@pytest.mark.parametrize("kind", ["table", "packed", "multi-table"])
@pytest.mark.parametrize(
    "value,pdf,law,density",
    [
        ([0, 2], [0.5, 0.99], "histogram", lambda x: 0.5),
        ([0, 1, 2], [0, 1, 0], "linear", lambda x: min(x, 2 - x)),
        ([1, 4], [0, np.log(4)], "semilog-x", np.log),
        ([0, 1], [1, np.e], "semilog-y", np.exp),
        ([1, 3], [1, 1 / 3], "log", lambda x: 1 / x),
    ],
)
def test_probability_integrals_follow_density_law(
    tmp_path, kind, value, pdf, law, density
):
    """Normalize independently so a linear approximation cannot pass the check."""
    integral = quad(density, value[0], value[-1])[0]
    expected = np.asarray(pdf) / integral
    with h5py.File(tmp_path / "normalized.h5", "w") as file:
        write_probability_table(file, kind, value, expected, [law])
        stored = file["tables/pdf"] if kind == "multi-table" else file["pdf"]
        np.testing.assert_array_equal(stored[: len(pdf)], expected)


@pytest.mark.parametrize("kind", ["table", "packed", "multi-table"])
@pytest.mark.parametrize(
    "factor,valid", [(0.5, True), (-0.5, True), (2, False), (-2, False)]
)
def test_probability_tolerance_preserves_source_values(tmp_path, kind, factor, valid):
    mass = 1 + factor * writer.PROBABILITY_TOLERANCE
    with h5py.File(tmp_path / "rounded.h5", "w") as file:
        if valid:
            write_probability_table(file, kind, [0, 1], [mass, mass])
            stored = file["tables/pdf"] if kind == "multi-table" else file["pdf"]
            assert stored[0] == mass
        else:
            with pytest.raises(ValueError, match="Total probability"):
                write_probability_table(file, kind, [0, 1], [mass, mass])
            assert not len(file)
            assert not len(file.attrs)


@pytest.mark.parametrize("kind", ["table", "packed", "multi-table"])
@pytest.mark.parametrize("law", sorted(writer.INTERPOLATION_LAWS))
def test_nearly_equal_coordinates_have_finite_integrals(tmp_path, kind, law):
    upper = np.nextafter(1.0, 2.0)
    density = 1 / (upper - 1)
    with h5py.File(tmp_path / "narrow.h5", "w") as file:
        write_probability_table(file, kind, [1, upper], [density, density], [law])


@pytest.mark.parametrize("kind", ["table", "packed", "multi-table"])
def test_probability_regions_allow_zero_width_density_jumps(tmp_path, kind):
    with h5py.File(tmp_path / "jump.h5", "w") as file:
        write_probability_table(
            file,
            kind,
            [0, 1, 1, 2],
            [0.2, 0, 0.8, 0.8],
            ["histogram", "semilog-y"],
            [3, 4],
        )


@pytest.mark.parametrize("kind", ["packed", "multi-table"])
def test_invalid_later_pdf_does_not_leave_partial_collection(tmp_path, kind):
    tables = [([0, 1], [1, 1], None, None), ([0, 1], [2, 2], None, None)]
    with h5py.File(tmp_path / "later-invalid.h5", "w") as file:
        with pytest.raises(ValueError, match="Packed PDF table 1: Total probability"):
            if kind == "packed":
                writer.pdf_tables(file, tables)
            else:
                writer.multi_table(file, [1, 2], tables)
        assert not len(file)
        assert not len(file.attrs)


@pytest.mark.parametrize("continuous_mass", [0.0, 0.7, 1.0])
def test_mixed_probability_checks_combined_mass(tmp_path, continuous_mass):
    with h5py.File(tmp_path / "mixed.h5", "w") as file:
        writer.distribution_table(
            file,
            [0, 10],
            [continuous_mass / 10] * 2,
            discrete_value=[2],
            pmf=[1 - continuous_mass],
        )
        np.testing.assert_array_equal(file["pdf"], [continuous_mass / 10] * 2)
        np.testing.assert_array_equal(file["pmf"], [1 - continuous_mass])


@pytest.mark.parametrize(
    "arguments",
    [
        {"discrete_value": [1], "pmf": [0]},
        {"discrete_value": [1], "pmf": [-1]},
        {"discrete_value": [np.inf], "pmf": [1]},
        {"discrete_value": [1], "pmf": [np.nan]},
        {"discrete_value": [1], "pmf": [np.inf]},
        {"discrete_value": [1, 2], "pmf": [0.3, 0.8]},
        {"discrete_value": [1, 2], "pmf": [1e308, 1e308]},
        {"value": [0, 1], "pdf": [0.7, 0.7], "discrete_value": [1], "pmf": [0.4]},
    ],
)
def test_invalid_discrete_and_mixed_probabilities(tmp_path, arguments):
    with h5py.File(tmp_path / "invalid-discrete.h5", "w") as file:
        with pytest.raises(ValueError):
            writer.distribution_table(file, **arguments)
        assert not len(file)
        assert not len(file.attrs)


def test_discrete_only_probability_tolerance(tmp_path):
    mass = 1 - 0.5 * writer.PROBABILITY_TOLERANCE
    with h5py.File(tmp_path / "discrete.h5", "w") as file:
        writer.distribution_table(file, discrete_value=[2], pmf=[mass])
        assert file["pmf"][0] == mass
