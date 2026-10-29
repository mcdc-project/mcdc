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
from data_library_generator.common.epics_converter import prepare_atomic_data


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
    path = output / "photon-Mg.h5"
    unrelated = output / "neutron-Mg24-293.6K.h5"
    with writer.atomic_file(unrelated, {"Lib81": {}}) as file:
        file["atomic-number"] = 12
        file["sentinel"] = [2.0, 3.0]
    previous = fingerprint(unrelated)
    convert_file(epdl, output, eadl)
    assert fingerprint(unrelated) == previous
    with h5py.File(path) as file, h5py.File(output / "atoms.h5") as atoms:
        writer.check_format(file)
        writer.check_format(atoms)
        assert set(file["source-libraries"]) == {"EPDL2025"}
        assert set(atoms["source-libraries"]) == {"EADL2025"}
        assert "source-libraries" not in atoms["Mg"]
        assert "atomic-relaxation" not in file
        assert "atomic-weight-ratio" not in file
        assert "temperature" not in file
        assert "kT" not in file
        photon = file
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
        relaxation = atoms["Mg/atomic-relaxation/subshells"]
        np.testing.assert_array_equal(
            relaxation["K/transitions/secondary-designator"], [0, 3]
        )
        assert relaxation["L2/transitions/energy"].shape == (0,)
        assert relaxation["L2/transitions/energy"].attrs["unit"] == "eV"
        for name in ("probability", "origin-designator", "secondary-designator"):
            assert "unit" not in relaxation[f"K/transitions/{name}"].attrs
        for parent, library in ((file, "EPDL2025"), (atoms, "EADL2025")):
            source = parent[f"source-libraries/{library}"]
            assert source["version"].asstr()[()] == "2025"
            assert source["release-date"].asstr()[()] == "2025-01"
            assert source["format-version"].asstr()[()] == "6"
            assert "temperature" not in source and "kT" not in source


def test_reuse_is_read_only_and_optional_eadl_preserves_relaxation(sources):
    epdl, eadl, output = sources
    path, atoms = output / "photon-Mg.h5", output / "atoms.h5"
    convert_file(epdl, output, eadl)
    before = fingerprint(path), fingerprint(atoms)
    convert_file(epdl, output, eadl)
    assert (fingerprint(path), fingerprint(atoms)) == before
    convert_file(epdl, output, rewrite=True)
    assert fingerprint(atoms) == before[1]
    with h5py.File(atoms) as file:
        np.testing.assert_allclose(
            file["Mg/atomic-relaxation/subshells/K/transitions/probability"], [0.4, 0.6]
        )


def test_invalid_rewrite_preserves_previous_element_file(sources):
    epdl, eadl, output = sources
    path = output / "photon-Mg.h5"
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
    assert sorted(p.name for p in output.iterdir()) == ["atoms.h5", "photon-Mg.h5"]


@pytest.mark.parametrize("identity,format_name", [(13, "mcdc-data"), (12, "other")])
def test_incompatible_identity_or_format_is_not_overwritten(
    sources, identity, format_name
):
    epdl, eadl, output = sources
    path = output / "photon-Mg.h5"
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
    with h5py.File(output / "photon-Mg.h5") as file:
        assert "atomic-relaxation" not in file
        assert set(file["source-libraries"]) == {"EPDL2025"}


def test_atomic_preparation_is_independent_of_projectile_preparation(sources):
    """Shared atomic preparation can run before or through a projectile generator."""
    epdl, eadl, output = sources
    first, second = output / "first", output / "second"
    first.mkdir()
    second.mkdir()
    prepare_atomic_data(eadl, first)
    convert_file(epdl, first, eadl)
    convert_file(epdl, second, eadl)
    for filename in ("atoms.h5", "photon-Mg.h5"):
        with h5py.File(first / filename) as a, h5py.File(second / filename) as b:
            assert dict(a.attrs) == dict(b.attrs)

            def compare(name, obj):
                assert name in b
                assert dict(obj.attrs) == dict(b[name].attrs)
                if isinstance(obj, h5py.Dataset) and not name.endswith(
                    "conversion-time"
                ):
                    np.testing.assert_array_equal(obj[()], b[name][()])

            a.visititems(compare)


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
    assert sorted(path.name for path in output.iterdir()) == [
        "atoms.h5",
        "photon-Mg.h5",
    ]
    before = fingerprint(output / "photon-Mg.h5")
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert fingerprint(output / "photon-Mg.h5") == before


def test_atomic_records_require_root_provenance(sources):
    _, eadl, output = sources
    prepare_atomic_data(eadl, output)
    sources = {"EADL2025": {"version": "2025"}}
    assert writer.prepared_file_exists(
        output / "atoms.h5", ["Mg/atomic-number"], sources
    )
    with h5py.File(output / "atoms.h5", "a") as file:
        del file["source-libraries/EADL2025/title"]
    assert not writer.prepared_file_exists(
        output / "atoms.h5", ["Mg/atomic-number"], sources
    )


def test_atomic_library_preserves_other_elements_and_root_provenance(sources):
    _, magnesium, output = sources
    aluminum = magnesium.with_name("EADL.ZA013000.endf")
    aluminum.write_text(
        magnesium.read_text()
        .replace("1.2000E+04", "1.3000E+04")
        .replace("12-Mg", "13-Al")
    )
    prepare_atomic_data(magnesium, output)
    prepare_atomic_data(aluminum, output)
    with h5py.File(output / "atoms.h5") as file:
        assert {"Mg", "Al", "source-libraries"} == set(file)
        source = file["source-libraries/EADL2025"]
        for name in ("title", "comments", "processing-date"):
            assert source[name].asstr()[()] == ""
        assert source["version"].asstr()[()] == "2025"
        previous = file["Al/atomic-relaxation/subshells/K/transitions/energy"][()]
        for element in ("Mg", "Al"):
            assert set(file[element]) == {"atomic-number", "atomic-relaxation"}
    prepare_atomic_data(magnesium, output, rewrite=True)
    with h5py.File(output / "atoms.h5") as file:
        np.testing.assert_array_equal(
            file["Al/atomic-relaxation/subshells/K/transitions/energy"], previous
        )
    before = fingerprint(output / "atoms.h5")
    prepare_atomic_data(magnesium, output)
    assert fingerprint(output / "atoms.h5") == before


def test_atomic_target_mismatch_does_not_change_output(sources):
    _, eadl, output = sources
    prepare_atomic_data(eadl, output)
    before = fingerprint(output / "atoms.h5")
    with pytest.raises(ValueError, match="differs from projectile"):
        prepare_atomic_data(eadl, output, atomic_number=13, rewrite=True)
    assert fingerprint(output / "atoms.h5") == before
    with h5py.File(output / "atoms.h5", "a") as file:
        file["Mg/atomic-number"][()] = 13
    before = fingerprint(output / "atoms.h5")
    with pytest.raises(ValueError, match="inconsistent atomic-number"):
        prepare_atomic_data(eadl, output)
    assert fingerprint(output / "atoms.h5") == before


def test_atomic_preparation_repairs_missing_root_provenance(sources):
    _, eadl, output = sources
    prepare_atomic_data(eadl, output)
    with h5py.File(output / "atoms.h5", "a") as file:
        del file["source-libraries"]
    prepare_atomic_data(eadl, output)
    with h5py.File(output / "atoms.h5") as file:
        assert set(file["source-libraries"]) == {"EADL2025"}
        assert "source-libraries" not in file["Mg"]
