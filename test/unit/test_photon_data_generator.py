"""Photon preparation checks using small, self-contained EPICS evaluations."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from data_library_generator.common import epics_converter as converter
from data_library_generator.common import hdf5_writer as writer
from data_library_generator.photon.generate import convert_file


def record(values, mf, mt):
    fields = [f"{value:11.4E}" for value in values]
    return "".join(fields).ljust(66) + f"{1200:4d}{mf:2d}{mt:3d}\n"


def text_record(text):
    return text.ljust(66) + f"{1200:4d}{1:2d}{451:3d}\n"


def tab1(mt, grid, values, mf=23, boundaries=None, laws=None):
    boundaries = [len(grid)] if boundaries is None else boundaries
    laws = [2] if laws is None else laws
    lines = [record([12000, 24.1, 0, 0, 0, 0], mf, mt)]
    lines.append(record([0, 0, 0, 0, len(laws), len(grid)], mf, mt))
    metadata = [value for pair in zip(boundaries, laws) for value in pair]
    data = [value for pair in zip(grid, values) for value in pair]
    for array in (metadata, data):
        lines.extend(
            record(array[start : start + 6], mf, mt)
            for start in range(0, len(array), 6)
        )
    return lines


def header(library):
    title = f"12-Mg 2025 Evaluated Test Library ({library})"
    return [
        title.ljust(66) + "   0 0  0\n",
        record([12000, 24.1, -1, 0, 0, 3], 1, 451),
        record([0, 0, 0, 0, 0, 6], 1, 451),
        record([0, 1e11, 0, 0, 3, 6], 1, 451),
        record([0, 0, 0, 0, 2, 0], 1, 451),
        text_record("12-Mg Eval-Aug23 Reference author"),
        text_record("NDS-IAEA-225 Dist-Jan25"),
    ]


@pytest.fixture
def sources(tmp_path):
    directory = tmp_path / "epics"
    directory.mkdir()
    epdl, eadl = directory / "EPDL.ZA012000.endf", directory / "EADL.ZA012000.endf"
    lines = header("EPDL")
    grid = [1, 2, 2, 4]
    for mt in (501, *converter.PHOTON_REACTIONS):
        values = [0, 0, 10, 20] if mt == 522 else [1, 2, 2, 4]
        lines.extend(tab1(mt, grid, values))
    for mt in (502, 504):
        lines.extend(tab1(mt, [0, 1, 2], [12, 6, 1], mf=27))
    for mt in (534, 536):
        lines.extend(tab1(mt, [2, 2, 4], [0, 10, 20]))
    epdl.write_text("".join(lines))
    lines = header("EADL")
    lines += [
        record([12000, 24.1, 0, 0, 2, 0], 28, 533),
        record([1, 0, 0, 0, 18, 2], 28, 533),
        record([100, 2, 0, 0, 0, 0], 28, 533),
        record([3, 0, 90, 0.4, 0, 0], 28, 533),
        record([3, 3, 80, 0.6, 0, 0], 28, 533),
        record([3, 0, 0, 0, 6, 0], 28, 533),
        record([10, 2, 0, 0, 0, 0], 28, 533),
    ]
    eadl.write_text("".join(lines))
    output = tmp_path / "prepared"
    output.mkdir()
    return epdl, eadl, output


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field, expected",
    [(" 1.234567+6", 1234567), ("-1.23456-2", -0.0123456), ("1.25D+03", 1250), ("", 0)],
)
def test_endf_numeric_notation(field, expected):
    assert converter.endf_float(field) == pytest.approx(expected)


def test_tab1_region_boundary_and_nonlinear_projection():
    table = converter.read_tab1(
        tab1(502, [1, 2, 3, 5], [1, 4, 9, 25], boundaries=[3, 4], laws=[2, 5])
    )
    grid = np.array([1, 1.5, 2, 2.5, 3, 4, 5], dtype=float)
    values, laws = converter.project_onto_grid(grid, table)
    np.testing.assert_allclose(values, [1, 2.5, 4, 6.5, 9, 16, 25])
    np.testing.assert_array_equal(laws, [2, 2, 2, 2, 5, 5])
    with h5py.File("nonlinear-xs.h5", "w", driver="core", backing_store=False) as file:
        converter.write_cross_section(file, grid, table)
        np.testing.assert_array_equal(file["interpolation/boundaries"], [5, 7])
        assert file["interpolation/laws"].asstr()[:].tolist() == ["linear", "log"]
        converter.write_table(file.create_group("source"), table)
        np.testing.assert_array_equal(file["source/interpolation/boundaries"], [3, 4])


@pytest.mark.parametrize(
    "law,expected", [(1, 2.0), (2, 20.0 / 3.0), (3, 9.0), (4, 4.0), (5, 2.0**2.5)]
)
def test_projection_respects_each_endf_interpolation_law(law, expected):
    table = converter.read_tab1(tab1(502, [1, 4], [2, 16], laws=[law]))
    values, _ = converter.project_onto_grid(np.array([1.0, 2.0, 4.0]), table)
    np.testing.assert_allclose(values, [2.0, expected, 16.0])


def test_threshold_and_absorption_edge_limits():
    source = converter.read_tab1(tab1(522, [2, 2, 4], [0, 10, 20]))
    values, _ = converter.project_onto_grid(
        np.array([1, 2, 2, 3, 4, 5], dtype=float), source
    )
    np.testing.assert_allclose(values, [0, 0, 10, 15, 20, 20])
    source = converter.read_tab1(tab1(522, [1, 2, 2, 3], [1, 3, 7, 9]))
    values, _ = converter.project_onto_grid(np.array([1, 1.5, 2, 2, 2.5, 3]), source)
    np.testing.assert_allclose(values, [1, 2, 3, 7, 8, 9])


def test_shared_grid_keeps_missing_knots_and_edge_multiplicity(sources):
    epdl, _, _ = sources
    sections = converter.read_sections(epdl)
    sections[23, 504] = tab1(504, [1, 1.5, 4], [1, 1.5, 4])
    np.testing.assert_array_equal(
        converter.photon_energy_grid(sections), [1, 1.5, 2, 2, 4]
    )


def test_conversion_preserves_other_projectiles_and_source_records(sources):
    epdl, eadl, output = sources
    path = output / "Mg.h5"
    with writer.atomic_file(path, {"Lib81": {"title": "neutron identity"}}) as file:
        file["atomic-number"] = 12
        file["projectiles/electron/sentinel"] = [2.0, 3.0]
        timestamp = file["source-libraries/Lib81/conversion-time"].asstr()[()]
    convert_file(epdl, output, eadl)
    with h5py.File(path) as file:
        writer.check_format(file)
        assert set(file["source-libraries"]) == {"Lib81", "EPDL2025", "EADL2025"}
        assert file["source-libraries/Lib81/conversion-time"].asstr()[()] == timestamp
        np.testing.assert_array_equal(file["projectiles/electron/sentinel"], [2, 3])
        assert "atomic-weight-ratio" not in file
        assert "temperature" not in file
        assert "kT" not in file
        photon = file["projectiles/photon"]
        np.testing.assert_array_equal(photon["xs-energy-grid"], [1, 2, 2, 4])
        assert photon["xs-energy-grid"].attrs["unit"] == "eV"
        assert len(photon["reactions"]) == 5
        for mt, (name, category) in converter.PHOTON_REACTIONS.items():
            reaction = photon[f"reactions/{name}"]
            assert reaction.attrs["ENDF-MT"] == mt
            assert reaction.attrs["category"] == category
            assert reaction["reference-frame"].asstr()[()] == "LAB"
            assert not len(reaction["cross-products"])
            assert reaction["xs/value"].attrs["unit"] == "barn"
            assert "unit" not in reaction["xs/offset"].attrs
        assert (
            photon["reactions/pair-production-electron-field/field"].asstr()[()]
            == "electron"
        )
        assert (
            photon["reactions/pair-production-nuclear-field/field"].asstr()[()]
            == "nuclear"
        )
        factor = photon["reactions/coherent-scattering/form-factor"]
        assert factor.attrs["type"] == "table"
        assert factor["grid"].attrs["unit"] == "1/angstrom"
        assert "unit" not in factor["value"].attrs
        shells = photon["reactions/photoelectric/subshells"]
        assert set(shells) == {"K", "L2"}
        np.testing.assert_array_equal(shells["K/xs/grid"], [2, 2, 4])
        assert shells["L2/designator"][()] == 3
        assert shells["K/xs/grid"].attrs["unit"] == "eV"
        assert shells["K/xs/value"].attrs["unit"] == "barn"
        relaxation = file["atomic-relaxation/subshells"]
        np.testing.assert_array_equal(
            relaxation["K/transitions/secondary-designator"], [0, 3]
        )
        assert relaxation["L2/transitions/energy"].shape == (0,)
        assert relaxation["L2/transitions/energy"].attrs["unit"] == "eV"
        for name in ("probability", "origin-designator", "secondary-designator"):
            assert "unit" not in relaxation[f"K/transitions/{name}"].attrs
        for library in ("EPDL2025", "EADL2025"):
            source = file[f"source-libraries/{library}"]
            assert source["version"].asstr()[()] == "2025"
            assert source["release-date"].asstr()[()] == "2025-01"
            assert source["format-version"].asstr()[()] == "6"
            assert "temperature" not in source and "kT" not in source


def test_reuse_is_read_only_and_optional_eadl_preserves_relaxation(sources):
    epdl, eadl, output = sources
    path = output / "Mg.h5"
    convert_file(epdl, output, eadl)
    before = fingerprint(path)
    convert_file(epdl, output, eadl)
    assert fingerprint(path) == before
    with h5py.File(path) as file:
        timestamp = file["source-libraries/EADL2025/conversion-time"].asstr()[()]
    convert_file(epdl, output, rewrite=True)
    with h5py.File(path) as file:
        assert (
            file["source-libraries/EADL2025/conversion-time"].asstr()[()] == timestamp
        )
        np.testing.assert_allclose(
            file["atomic-relaxation/subshells/K/transitions/probability"], [0.4, 0.6]
        )


def test_invalid_rewrite_preserves_previous_element_file(sources):
    epdl, eadl, output = sources
    path = output / "Mg.h5"
    convert_file(epdl, output, eadl)
    before = fingerprint(path)
    epdl.write_text(
        "".join(
            line
            for line in epdl.read_text().splitlines(keepends=True)
            if (int(line[70:72]), int(line[72:75])) != (27, 502)
        )
    )
    with pytest.raises(ValueError, match="form factors"):
        convert_file(epdl, output, eadl, rewrite=True)
    assert fingerprint(path) == before
    assert sorted(p.name for p in output.iterdir()) == ["Mg.h5"]


@pytest.mark.parametrize("identity,format_name", [(13, "mcdc-data"), (12, "other")])
def test_incompatible_identity_or_format_is_not_overwritten(
    sources, identity, format_name
):
    epdl, eadl, output = sources
    path = output / "Mg.h5"
    with writer.atomic_file(path, {"Lib81": {}}) as file:
        file["atomic-number"] = identity
        file.attrs["format"] = format_name
    before = fingerprint(path)
    with pytest.raises(ValueError):
        convert_file(epdl, output, eadl, rewrite=True)
    assert fingerprint(path) == before


def test_no_eadl_writes_no_relaxation_group(sources):
    epdl, _, output = sources
    convert_file(epdl, output)
    with h5py.File(output / "Mg.h5") as file:
        assert "atomic-relaxation" not in file
        assert set(file["source-libraries"]) == {"EPDL2025"}


def test_photon_and_neutron_preparation_order_is_independent(sources):
    """A neutron element contribution and photon conversion commute."""
    epdl, eadl, output = sources
    photon_first = output / "photon-first"
    neutron_first = output / "neutron-first"
    photon_first.mkdir()
    neutron_first.mkdir()
    source = {"Lib81": {"title": "neutron contribution"}}

    convert_file(epdl, photon_first, eadl)
    writer.write_companion(photon_first / "Mg.h5", {"atomic-number": 12}, source)
    writer.write_companion(neutron_first / "Mg.h5", {"atomic-number": 12}, source)
    convert_file(epdl, neutron_first, eadl)

    with (
        h5py.File(photon_first / "Mg.h5") as first,
        h5py.File(neutron_first / "Mg.h5") as second,
    ):
        assert dict(first.attrs) == dict(second.attrs)
        assert set(first["source-libraries"]) == set(second["source-libraries"])
        names = []
        first.visit(names.append)
        for name in names:
            if name == "source-libraries" or name.startswith("source-libraries/"):
                continue
            assert name in second
            assert dict(first[name].attrs) == dict(second[name].attrs)
            if isinstance(first[name], h5py.Dataset):
                np.testing.assert_array_equal(first[name][()], second[name][()])
        other_names = []
        second.visit(other_names.append)
        assert set(names) == set(other_names)


def test_flat_directory_cli_and_element_selection(sources):
    epdl, _, output = sources
    (epdl.parent / "unrelated.endf").write_text("not an EPDL evaluation")
    environment = {
        **os.environ,
        "MCDC_EPICS_LIB": str(epdl.parent),
        "MCDC_LIB": str(output),
    }
    command = [sys.executable, str(TOOLS / "data_library_generator/photon/generate.py")]
    result = subprocess.run(
        [*command, "--elements", "Mg"], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == ["Mg.h5"]
    before = fingerprint(output / "Mg.h5")
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert fingerprint(output / "Mg.h5") == before


def test_shared_reuse_check_retains_neutron_file_requirements(tmp_path):
    sources = {"Lib81": {"temperature": 293.6, "kT": 0.0253}}
    by_file = {}
    for filename, fields in (
        (
            "H2-293.6K.h5",
            {"temperature": 293.6, "kT": 0.0253, "projectiles/neutron/reactions": None},
        ),
        (
            "H2.h5",
            {
                "mass-number": 2,
                "atomic-weight-ratio": 1.9968,
                "excitation-level": 0,
                "fissionable": False,
            },
        ),
        ("H.h5", {"atomic-number": 1}),
    ):
        by_file[filename] = sources
        with writer.atomic_file(tmp_path / filename, sources) as file:
            for name, value in fields.items():
                if value is None:
                    file.create_group(name)
                else:
                    file[name] = value
    assert writer.library_exists(tmp_path, "H2-293.6K.h5", "H2", "H", by_file)
    with h5py.File(tmp_path / "H2.h5", "a") as file:
        del file["mass-number"]
    assert not writer.library_exists(tmp_path, "H2-293.6K.h5", "H2", "H", by_file)
