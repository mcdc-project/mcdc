"""Delayed fission products use their group's probability, spectrum, and direction."""

import h5py
import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.transport.physics.neutron.native import sample_fission


def _spectrum(group, fraction):
    group.attrs["type"] = "level-scattering"
    group["C1"] = 0.0
    group["C2"] = fraction


def _library(path, prompt_yield):
    with h5py.File(path, "w") as file:
        for name, value in [
            ("atomic-number", 1),
            ("mass-number", 2),
            ("atomic-weight-ratio", 2.0),
            ("fissionable", True),
            ("excitation-level", 0),
        ]:
            file[name] = value
        file.attrs["format"], file.attrs["format-version"] = "mcdc-data", "0.1.0"
        file["temperature"], file["kT"] = 293.6, 0.0253
        file["xs-energy-grid"] = [1.0, 20.0]
        group = file.create_group("reactions/fission")
        group.attrs["ENDF-MT"], group.attrs["category"] = 18, "fission"
        for name, value in [
            ("prompt-multiplicity", prompt_yield),
            ("delayed-multiplicity", 1 - prompt_yield),
        ]:
            parameter = group.create_group(name)
            parameter.attrs["type"] = "polynomial"
            parameter["coefficients"] = [value]
        group["xs/value"] = [1.0, 1.0]
        group["xs/offset"] = 0
        group["xs/interpolation/laws"] = np.array(["linear"], dtype=h5py.string_dtype())
        group["xs/interpolation/boundaries"] = [2]
        group["reference-frame"] = "LAB"
        group["Q-value"] = 0.0
        group.create_group("cross-products")
        group.create_group("prompt-angular-cosine-distribution").attrs[
            "type"
        ] = "isotropic"
        _spectrum(group.create_group("prompt-energy-distribution"), 0.5)
        delayed = group.create_group("delayed-neutrons")
        delayed["fractions"] = [0.25, 0.75]
        delayed["decay-rates"] = [1.0, 2.0]
        for i, fraction in enumerate([0.2, 0.8]):
            _spectrum(delayed.create_group(f"energy-distribution-{i}"), fraction)


@pytest.mark.parametrize("prompt_yield", [0.0, 0.8])
def test_delayed_group_products(
    tmp_path, monkeypatch, prepare_simulation, prompt_yield
):
    _library(tmp_path / "neutron-H2-293.6K.h5", prompt_yield)
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    cell = mcdc.Cell(fill=mcdc.Material(number_density={"H2": 1.0}))

    def configure(simulation):
        simulation.settings.neutron_transport.active = True

    container, data = prepare_simulation(cells=[cell], configure=configure)
    simulation = container[0]
    energies, directions, times = [], [], []
    for seed in range(1, 1001):
        particle = np.zeros(1, dtype=type_.particle)
        particle[0]["E"], particle[0]["w"] = 10.0, 1.0
        particle[0]["ux"], particle[0]["alive"] = 1.0, True
        particle[0]["rng_seed"] = seed
        interaction = np.zeros(1, dtype=type_.interaction_data)
        sample_fission(
            simulation["neutron_reactions"][0],
            particle,
            interaction,
            simulation["nuclides"][0],
            simulation,
            data,
        )
        assert particle[0]["alive"]
        assert interaction[0]["energy_deposition"] + particle[0]["E"] == pytest.approx(
            10.0
        )
        energies.append(particle[0]["E"])
        directions.append([particle[0][axis] for axis in ("ux", "uy", "uz")])
        times.append(particle[0]["t"])
    assert set(energies) == ({2.0, 5.0, 8.0} if prompt_yield else {2.0, 8.0})
    assert np.mean(np.array(energies) == 2.0) == pytest.approx(
        0.25 * (1 - prompt_yield), abs=0.025
    )
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.mean(directions, axis=0), 0, atol=0.08)
    delayed = np.array(energies) != 5.0
    assert np.all(np.array(times)[delayed] > 0)
    assert np.all(np.array(times)[~delayed] == 0)
