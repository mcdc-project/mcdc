from types import SimpleNamespace

import h5py
import numpy as np

import mcdc
from mcdc.main import prepare
from mcdc.output import create_runtime_dataset, generate_output
from mcdc.constant import SCORE_FLUX
from mcdc.output import clear_census_based_tally_files, recombine_tallies


def _write_census_tally(path, values):
    with h5py.File(path, "w") as file:
        tally = file.require_group("tallies/tracklength_tally_0")
        tally.require_group("grid").create_dataset("time", data=[0.0, 1.0])
        score = tally.require_group("flux")
        score.create_dataset("mean", data=values)
        score.create_dataset("sdev", data=np.zeros_like(values))


def test_runtime_output_includes_data_loading(tmp_path):
    runtimes = {
        "runtime_total": 5.0,
        "runtime_preparation": 2.0,
        "runtime_data_loading": 1.5,
        "runtime_simulation": 2.5,
        "runtime_output": 0.5,
        "runtime_bank_management": 0.25,
    }

    with h5py.File(tmp_path / "runtime.h5", "w") as file:
        create_runtime_dataset(file, runtimes)

    with h5py.File(tmp_path / "runtime.h5", "r") as file:
        assert file["runtime/data_loading"][0] == 1.5


def test_clear_census_based_tally_files(tmp_path):
    base_name = tmp_path / "output"
    settings = SimpleNamespace(output_name=str(base_name), N_batch=2, N_census=3)
    expected_files = [
        tmp_path / f"output-batch_{idx_batch}-census_{idx_census}.h5"
        for idx_batch in range(settings.N_batch)
        for idx_census in range(settings.N_census)
    ]
    for path in expected_files:
        path.touch()
    unrelated = tmp_path / "output-unrelated.h5"
    unrelated.touch()

    clear_census_based_tally_files(settings)

    assert not any(path.exists() for path in expected_files)
    assert unrelated.exists()


def test_recombine_tallies_zero_fills_censuses_missing_after_extinction(tmp_path):
    base_name = tmp_path / "output"
    settings = SimpleNamespace(
        output_name=str(base_name),
        use_census_based_tally=True,
        census_tally_frequency=1,
        census_time=np.array([1.0, 2.0, np.inf]),
        N_batch=2,
        N_census=3,
    )
    tally = SimpleNamespace(
        name="tracklength_tally_0",
        scores=[SCORE_FLUX],
        bin_shape=[1, 1, 1, 1, 2, 1],
    )
    simulation_python = SimpleNamespace(settings=settings, tallies=[tally])
    simulation = {"mpi_master": True}

    with h5py.File(f"{base_name}.h5", "w"):
        pass

    _write_census_tally(f"{base_name}-batch_0-census_0.h5", np.array([2.0, 4.0]))
    _write_census_tally(f"{base_name}-batch_1-census_0.h5", np.array([4.0, 6.0]))
    _write_census_tally(f"{base_name}-batch_1-census_1.h5", np.array([8.0, 10.0]))

    recombine_tallies(simulation_python, simulation)

    with h5py.File(f"{base_name}.h5", "r") as file:
        tally_path = "tallies/tracklength_tally_0"
        np.testing.assert_array_equal(
            file[f"{tally_path}/grid/time"][:], [0.0, 1.0, 2.0]
        )
        np.testing.assert_array_equal(
            file[f"{tally_path}/flux/mean"][:],
            [[3.0, 5.0], [4.0, 5.0]],
        )
        np.testing.assert_array_equal(
            file[f"{tally_path}/flux/sdev"][:],
            [[1.0, 1.0], [4.0, 5.0]],
        )


def test_output_serializes_nested_transport_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    simulation.set_sources([mcdc.Source(particle_type="electron")])
    simulation.settings.electron_transport.prioritize_low_energy = True
    simulation.settings.output_name = "output"
    simulation.compile()
    revision = ("123456abcdef", "2026-11-12T12:00:00+00:00")
    monkeypatch.setattr("mcdc.main.git_revision", lambda: revision)
    container, data = prepare(simulation)

    assert simulation.version == mcdc.__version__
    assert (simulation.git_commit, simulation.git_commit_time) == revision
    for name in ("version", "git_commit", "git_commit_time"):
        assert name not in container.dtype.names
    generate_output(container[0], data, simulation, no_tally_output=True)

    with h5py.File(tmp_path / "output.h5", "r") as output:
        version = output["version"]
        assert version.asstr()[()] == mcdc.__version__
        commit, commit_time = revision
        assert version.attrs["git_commit"] == commit
        assert version.attrs["git_commit_time"] == commit_time
        settings = output["settings"]
        assert not settings["neutron_transport/active"][()]
        assert not settings["neutron_transport/prioritize_low_energy"][()]
        assert settings["electron_transport/active"][()]
        assert settings["electron_transport/prioritize_low_energy"][()]
        assert settings["N_particle"][()] == simulation.settings.N_particle
        np.testing.assert_array_equal(
            settings["census_time"][()], simulation.settings.census_time
        )
