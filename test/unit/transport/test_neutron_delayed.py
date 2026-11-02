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
            ("atomic_number", 1),
            ("mass_number", 2),
            ("atomic_weight_ratio", 2.0),
            ("fissionable", True),
            ("excitation_level", 0),
        ]:
            file[name] = value
        reactions = file.create_group("neutron_reactions")
        reactions["xs_energy_grid"] = [1e-6, 20e-6]
        fission = reactions.create_group("fission")
        for name, value in [
            ("prompt_multiplicity", prompt_yield),
            ("delayed_multiplicity", 1 - prompt_yield),
        ]:
            group = fission.create_group(name)
            group.attrs["type"] = "polynomial"
            group["coefficient"] = [value]
        group = fission.create_group("MT-018")
        group.attrs["MT"] = 18
        group["xs"] = [1.0, 1.0]
        group["xs"].attrs["offset"] = 0
        group["reference_frame"] = "LAB"
        group["Q-value"] = 0.0
        group.create_group("angular_cosine_distribution").attrs["type"] = "isotropic"
        _spectrum(group.create_group("energy_spectrum-1"), 0.5)
        delayed = fission.create_group("delayed_neutron_precursors")
        delayed["fractions"] = [0.25, 0.75]
        delayed["decay_rates"] = [1.0, 2.0]
        for i, fraction in enumerate([0.2, 0.8]):
            _spectrum(delayed.create_group(f"energy_spectrum-{i+1}"), fraction)


@pytest.mark.parametrize("prompt_yield", [0.0, 0.8])
def test_delayed_group_products(
    tmp_path, monkeypatch, prepare_simulation, prompt_yield
):
    _library(tmp_path / "H2-293.6K.h5", prompt_yield)
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
