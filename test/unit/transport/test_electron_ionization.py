from pathlib import Path
import sys

import math

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from data_library_generator.common.hdf5_writer import (
    data_table as write_electron_data,
    interpolation,
    multi_table,
)

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import ELECTRON_CUTOFF_ENERGY, PARTICLE_ELECTRON
from mcdc.transport.physics.electron.native import sample_ionization
from mcdc.transport.rng import RNG_C, RNG_G, RNG_MOD


def _rng_state_for_xi(xi, draw_count):
    # Numba caches sample_ionization compiled against the production LCG, so
    # the test seeds the real generator to a known draw instead of mocking it.
    # The largest seed rounds to 1.0 when converted to a float.
    modulus = int(RNG_MOD)
    target_seed = min(int(xi * modulus), modulus - 1)
    inverse = pow(int(RNG_G), -1, modulus)
    seed = target_seed
    for _ in range(draw_count):
        seed = ((seed - int(RNG_C)) * inverse) % modulus
    state = np.zeros(1, dtype=type_.particle)
    state[0]["rng_seed"] = np.uint64(seed)
    return state, target_seed


def _write_ionization_library(path, product_grid, product_offset, value, cdf):
    # A small synthetic library exercising the real HDF5 reader and packing
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "mcdc-data"
        file.attrs["format-version"] = "0.1.0"
        file["atomic-number"] = 1
        grid = np.array([100.0, 10000.0])
        file["xs-energy-grid"] = grid
        reaction = file.create_group("reactions/ionization")
        reaction.attrs["ENDF-MT"] = 522
        reaction.attrs["category"] = "ionization"
        reaction["xs/value"] = [1.0, 1.0]
        reaction["xs/offset"] = 0
        interpolation(reaction["xs"], 2)
        reaction["reference-frame"] = "LAB"
        reaction.create_group("cross-products")
        shell = reaction.create_group("subshells/K")
        shell["designator"] = 1
        shell["binding-energy"] = 100.0
        write_electron_data(shell.create_group("xs"), grid, [1.0, 1.0], "barn")
        shell.create_group("cross-products")
        offsets = [*product_offset, len(value)]
        tables = []
        for i in range(len(product_grid)):
            start, end = offsets[i : i + 2]
            x, c = np.asarray(value[start:end]), np.asarray(cdf[start:end])
            pdf = np.diff(c) / np.diff(x)
            tables.append((x, np.r_[pdf, pdf[-1]], ["histogram"]))
        multi_table(
            shell.create_group("secondary-energy-distribution"), product_grid, tables
        )


@pytest.fixture
def ionization_model(tmp_path, monkeypatch, prepare_simulation):
    _write_ionization_library(
        tmp_path / "electron-H.h5",
        product_grid=[100.0, 1000.0, 10_000.0],
        product_offset=[0, 2, 5],
        value=[0.01, 0.1, 1.0, 4.0, 16.0, 9.0, 36.0, 576.0],
        cdf=[0.0, 1.0, 0.0, 0.25, 1.0, 0.0, 0.25, 1.0],
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    material = mcdc.Material(number_density={"H": 1.0})
    cell = mcdc.Cell(fill=material)

    def configure(simulation):
        simulation.settings.electron_transport.active = True

    simulation_container, data = prepare_simulation(cells=[cell], configure=configure)
    return simulation_container[0], data


def test_ionization_library_without_spectrum_above_binding_energy(
    tmp_path, monkeypatch, prepare_simulation
):
    # Every product table at or below the binding energy is rejected on load.
    _write_ionization_library(
        tmp_path / "electron-H.h5",
        product_grid=[50.0, 100.0],
        product_offset=[0, 2],
        value=[0.01, 0.1, 0.01, 0.1],
        cdf=[0.0, 1.0, 0.0, 1.0],
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))

    with pytest.raises(SystemExit):
        material = mcdc.Material(number_density={"H": 1.0})
        cell = mcdc.Cell(fill=material)

        def configure(simulation):
            simulation.settings.electron_transport.active = True

        prepare_simulation(cells=[cell], configure=configure)


@pytest.mark.parametrize(
    "incident_energy, expected_T_delta",
    [
        (100.001, 8.0 * (100.001 - 100.0) / 900.0),
        (math.sqrt(1000.0 * 10_000.0), math.sqrt(8.0 * 144.0)),
        (10_000.0, 144.0),
    ],
)
@pytest.mark.parametrize("prioritize", [False, True])
def test_ionization_conserves_energy_with_cutoff_and_banking(
    ionization_model, incident_energy, expected_T_delta, prioritize
):
    simulation, data = ionization_model
    simulation["settings"]["electron_transport"]["prioritize_low_energy"] = prioritize
    # The first draw chooses the only subshell; the second samples its spectrum.
    particle_container, _ = _rng_state_for_xi(0.5, draw_count=2)
    particle = particle_container[0]
    particle["particle_type"] = PARTICLE_ELECTRON
    particle["E"] = incident_energy
    particle["w"] = 2.0
    particle["uz"] = 1.0
    particle["alive"] = True
    collision_container = np.zeros(1, dtype=type_.interaction_data)

    sample_ionization(
        simulation["electron_reactions"][0],
        particle_container,
        collision_container,
        simulation["elements"][0],
        simulation,
        data,
    )

    expected_E = incident_energy - 100.0 - expected_T_delta
    expected_deposition = 100.0
    if expected_E <= ELECTRON_CUTOFF_ENERGY:
        expected_deposition += expected_E
        expected_E = 0.0
    if expected_T_delta <= ELECTRON_CUTOFF_ENERGY:
        expected_deposition += expected_T_delta

    bank = simulation["bank_active"]
    bank_size = bank["size"][0]
    assert bank_size == int(expected_T_delta > ELECTRON_CUTOFF_ENERGY)
    if bank_size and prioritize and 0.0 < expected_T_delta < expected_E:
        expected_E, expected_T_delta = expected_T_delta, expected_E
    assert particle["E"] == pytest.approx(expected_E, rel=1e-13)
    assert particle["alive"] == (expected_E > 0.0)
    assert collision_container[0]["energy_deposition"] == pytest.approx(
        expected_deposition * 2.0, rel=1e-13
    )

    outgoing_energy = particle["E"] * particle["w"]
    if bank_size:
        secondary = bank["particle_data"][0]
        assert secondary["E"] == pytest.approx(expected_T_delta, rel=1e-13)
        assert secondary["w"] == particle["w"]
        assert secondary["particle_type"] == PARTICLE_ELECTRON
        assert np.linalg.norm(
            [secondary["ux"], secondary["uy"], secondary["uz"]]
        ) == pytest.approx(1.0, rel=1e-13)
        outgoing_energy += secondary["E"] * secondary["w"]
    assert outgoing_energy + collision_container[0]["energy_deposition"] == (
        pytest.approx(incident_energy * 2.0, rel=1e-13)
    )


def test_ionization_priority_swaps_complete_particle_states(ionization_model):
    simulation, data = ionization_model
    results = []
    for prioritize in (False, True):
        simulation["settings"]["electron_transport"][
            "prioritize_low_energy"
        ] = prioritize
        simulation["bank_active"]["size"][0] = 0
        particles, _ = _rng_state_for_xi(0.5, draw_count=2)
        particle = particles[0]
        particle["particle_type"] = PARTICLE_ELECTRON
        particle["E"] = 10_000.0
        particle["w"] = 2.0
        particle["uz"] = 1.0
        particle["alive"] = True
        particle["cell_ID"] = 7
        particle["material_ID"] = 3
        collision = np.zeros(1, dtype=type_.interaction_data)
        sample_ionization(
            simulation["electron_reactions"][0],
            particles,
            collision,
            simulation["elements"][0],
            simulation,
            data,
        )
        assert simulation["bank_active"]["size"][0] == 1
        assert particle["alive"]
        assert particle["cell_ID"] == 7
        assert particle["material_ID"] == 3
        results.append(
            (
                particle.copy(),
                simulation["bank_active"]["particle_data"][0].copy(),
                collision[0]["energy_deposition"],
            )
        )

    primary, secondary, deposition = results[0]
    active, banked, priority_deposition = results[1]
    assert active["E"] < banked["E"]
    for name in type_.particle_data.names:
        assert active[name] == secondary[name]
        assert banked[name] == primary[name]
    assert priority_deposition == deposition
