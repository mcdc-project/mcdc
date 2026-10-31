"""Electron ACE preparation preserves the source arrays and existing atomic data."""

from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("ACEtk")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from data_library_generator.common.hdf5_writer import atomic_file
from data_library_generator.electron.generate import convert_table
from mcdc.object_.distribution import DistributionMultiTable
from mcdc.object_.hdf5_reader import read_distribution


@pytest.fixture
def electron_ace():
    """Two nonadjacent shells with distinct cross sections identify ordering errors."""
    energy = [1e-4, 1e-2]
    spectra = SimpleNamespace(
        energies=energy,
        distributions=[
            SimpleNamespace(outgoing_energies=[1e-8, 2e-8, 4e-8], cdf=[0, 0.25, 1]),
            SimpleNamespace(outgoing_energies=[1e-6, 4e-6, 9e-6], cdf=[0, 0.5, 1]),
        ],
    )
    terminal = SimpleNamespace(number_transitions=0)
    return SimpleNamespace(
        zaid="14000.14p",
        header=SimpleNamespace(
            title="EPICS2014 test evaluation", date="07/02/15", version="1.0.0"
        ),
        electron_cross_section_block=SimpleNamespace(
            energies=energy,
            excitation=[1, 2],
            bremsstrahlung=[3, 4],
            elastic=[5, 6],
            electroionisation=lambda i: [i, 2 * i],
        ),
        electron_elastic_cross_section_block=SimpleNamespace(
            total=[7, 8], transport=[1, 2]
        ),
        electron_subshell_block=SimpleNamespace(
            number_electron_subshells=2,
            designator=lambda i: [1, 3][i - 1],
            binding_energy=lambda i: i * 1e-5,
            population=lambda i: 2,
        ),
        electron_elastic_angular_distribution_block=SimpleNamespace(
            energies=energy,
            distributions=[SimpleNamespace(cosines=[-1, 0, 1], cdf=[0, 0.2, 1])] * 2,
        ),
        electron_excitation_energy_loss_block=SimpleNamespace(
            energies=energy,
            excitation_energy_loss=[1e-5, 2e-5],
        ),
        electron_energy_after_bremsstrahlung_block=SimpleNamespace(
            energies=energy,
            energy_after_bremsstrahlung=[1e-5, 1e-3],
        ),
        bremsstrahlung_energy_distribution_block=spectra,
        electroionisation_energy_distribution_block=lambda i: spectra,
        subshell_transition_data_block=SimpleNamespace(
            transition_data=lambda i: terminal
        ),
    )


def test_electron_preparation_preserves_source_fields(tmp_path, electron_ace):
    convert_table(electron_ace, tmp_path)
    with h5py.File(tmp_path / "electron-Si.h5", "r") as file:
        assert file.attrs["format"] == "mcdc-data"
        assert "atomic-weight-ratio" not in file
        assert "temperature" not in file
        assert (
            file["source-libraries/EPRDATA14/title"].asstr()[()]
            == electron_ace.header.title
        )
        np.testing.assert_array_equal(file["xs-energy-grid"], [100, 10000])
        np.testing.assert_array_equal(file["reactions/elastic/xs/value"], [7, 8])
        np.testing.assert_array_equal(file["reactions/ionization/xs/value"], [3, 6])
        np.testing.assert_array_equal(
            file["reactions/excitation/energy-loss/value"], [10, 20]
        )
        np.testing.assert_array_equal(
            file["reactions/bremsstrahlung/energy-loss/value"], [10, 1000]
        )
        assert file["reactions/elastic/xs/value"].attrs["unit"] == "barn"
        spectrum = read_distribution(
            file["reactions/bremsstrahlung/photon-energy-distribution"]
        )
        expected = DistributionMultiTable(
            grid=[100, 10000],
            value=[[0.01, 0.02, 0.04], [1, 4, 9]],
            cdf=[[0, 0.25, 1], [0, 0.5, 1]],
        )
        for name in ("grid", "offsets", "value", "pdf", "cdf", "pdf_interpolations"):
            np.testing.assert_allclose(
                getattr(spectrum, name), getattr(expected, name), rtol=1e-14
            )
    with h5py.File(tmp_path / "atoms.h5", "r") as file:
        assert set(file["Si/atomic-relaxation/subshells"]) == {"K", "L2"}
        assert len(file["Si/atomic-relaxation/subshells/K/transitions/energy"]) == 0


def test_electron_rewrite_preserves_existing_atomic_evaluation(tmp_path, electron_ace):
    with atomic_file(tmp_path / "atoms.h5", {"EADL": {}}) as file:
        file["Si/atomic-number"] = 14
        file["Si/atomic-relaxation/source-marker"] = 123
    previous = (tmp_path / "atoms.h5").read_bytes()
    convert_table(electron_ace, tmp_path, rewrite=True)
    assert (tmp_path / "atoms.h5").read_bytes() == previous


def test_electron_reuse_and_failed_rewrite_preserve_output(tmp_path, electron_ace):
    convert_table(electron_ace, tmp_path)
    path = tmp_path / "electron-Si.h5"
    previous = path.read_bytes()
    timestamp = path.stat().st_mtime_ns
    convert_table(electron_ace, tmp_path)
    assert path.stat().st_mtime_ns == timestamp
    electron_ace.bremsstrahlung_energy_distribution_block.distributions[0].cdf = [
        0,
        0.8,
        0.2,
    ]
    with pytest.raises(ValueError, match="invalid EPR CDF"):
        convert_table(electron_ace, tmp_path, rewrite=True)
    assert path.read_bytes() == previous


def test_epr_atomic_probabilities_are_cumulative(tmp_path, electron_ace):
    transition = SimpleNamespace(
        number_transitions=2,
        probability=lambda i: [0.3, 1][i - 1],
        energy=lambda i: i * 1e-6,
        primary_designator=lambda i: 3,
        secondary_designator=lambda i: [0, 3][i - 1],
    )
    terminal = SimpleNamespace(number_transitions=0)
    electron_ace.subshell_transition_data_block.transition_data = lambda i: (
        transition if i == 1 else terminal
    )
    convert_table(electron_ace, tmp_path)
    with h5py.File(tmp_path / "atoms.h5", "r") as file:
        np.testing.assert_allclose(
            file["Si/atomic-relaxation/subshells/K/transitions/probability"], [0.3, 0.7]
        )


@pytest.mark.parametrize("keep_going", [False, True])
def test_electron_batch_reports_table_context(
    tmp_path, monkeypatch, capsys, electron_ace, keep_going
):
    """A malformed table does not hide its element or silently end a batch."""
    from copy import copy
    from data_library_generator.electron import generate

    tables = [copy(electron_ace) for _ in range(3)]
    for table, za in zip(tables, [14000, 15000, 16000]):
        table.zaid = f"{za}.14p"
    monkeypatch.setattr(
        generate,
        "ACEtk",
        SimpleNamespace(
            PhotoatomicTable=SimpleNamespace(from_concatenated_file=lambda path: tables)
        ),
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "eprdata14").touch()
    monkeypatch.delenv("MCDC_ACE_LIB", raising=False)
    monkeypatch.setenv("MCDC_ACE_EPRDATA_LIB", str(source))
    monkeypatch.setenv("MCDC_LIB", str(tmp_path / "prepared"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--verbose", *(["--keep-going"] if keep_going else [])],
    )
    seen = []

    def convert(table, *args, **kwargs):
        seen.append(table.zaid)
        if table.zaid == "15000.14p":
            raise ValueError("invalid EPR CDF")

    monkeypatch.setattr(generate, "convert_table", convert)
    with pytest.raises(SystemExit) as error:
        generate.main()
    assert error.value.code == 1
    assert "eprdata14:15000.14p: invalid EPR CDF" in capsys.readouterr().err
    assert len(seen) == (3 if keep_going else 2)
