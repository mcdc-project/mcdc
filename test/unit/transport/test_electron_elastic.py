from pathlib import Path
import sys

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
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
from mcdc.constant import DATA_NONE, DATA_TABLE, MU_CUTOFF
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import sample_multi_table_at
from mcdc.transport.physics.electron.native import sample_elastic_cosine
from mcdc.transport import rng

#: Mean of 1 - mu for each large-angle CDF, also transport / large-angle XS.
MEAN_DEFLECTION = [1.0, 0.1, 0.01]
ENERGIES = [100.0, 1.0e4, 1.0e6]


def _write_elastic_library(path, transport, moments):
    """Supply distinct CDFs and a total XS twice the large-angle component."""
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "mcdc-data"
        file.attrs["format-version"] = "0.1.0"
        file["atomic-number"] = 1
        grid = np.array(ENERGIES)
        file["xs-energy-grid"] = grid
        reaction = file.create_group("reactions/elastic")
        reaction.attrs["ENDF-MT"] = 526
        reaction.attrs["category"] = "elastic-scattering"
        reaction["xs/value"] = [2.0, 2.0, 2.0]
        reaction["xs/offset"] = 0
        interpolation(reaction["xs"], 3)
        reaction["reference-frame"] = "LAB"
        reaction.create_group("cross-products")
        if transport:
            write_electron_data(
                reaction.create_group("transport-xs"), grid, moments, "barn"
            )
        large = reaction.create_group("large-angle")
        write_electron_data(large.create_group("xs"), grid, [1.0, 1.0, 1.0], "barn")
        multi_table(
            large.create_group("angular-cosine-distribution"),
            ENERGIES,
            [
                (
                    [2.0 - 2.0 * mean - MU_CUTOFF, MU_CUTOFF],
                    [1.0 / (2.0 * mean - 2.0 * (1.0 - MU_CUTOFF))] * 2,
                    ["histogram"],
                )
                for mean in MEAN_DEFLECTION
            ],
            unit=None,
        )


@pytest.fixture
def elastic_model(tmp_path, monkeypatch, prepare_simulation):
    def build(transport=True, moments=MEAN_DEFLECTION, edit=None):
        _write_elastic_library(tmp_path / "electron-H.h5", transport, moments)
        if edit is not None:
            with h5py.File(tmp_path / "electron-H.h5", "r+") as file:
                edit(file)
        monkeypatch.setenv("MCDC_LIB", str(tmp_path))
        cell = mcdc.Cell(fill=mcdc.Material(number_density={"H": 1.0}))

        def configure(simulation):
            simulation.settings.electron_transport.active = True

        simulation_container, data = prepare_simulation(
            cells=[cell], configure=configure
        )
        simulation = simulation_container[0]
        element = simulation["elements"][0]
        ID = mcdc_get.element.electron_elastic_scattering_reaction_IDs(0, element, data)
        reaction = simulation["electron_reactions"][int(ID)]
        elastic = simulation["electron_elastic_scattering_reactions"][
            reaction["sub_ID"]
        ]
        return simulation, data, elastic

    return build


def test_mean_deflection_uses_the_large_angle_cross_section(elastic_model):
    """The source transport correction belongs to the large-angle component."""
    simulation, data, elastic = elastic_model()
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    assert mean_deflection["sub_type"] == DATA_TABLE
    for energy, expected in zip(ENERGIES, MEAN_DEFLECTION):
        assert evaluate_data(
            energy, mean_deflection, simulation, data
        ) == pytest.approx(expected)


@pytest.mark.parametrize("representation", ["polynomial", "log-table"])
def test_large_angle_normalization_evaluates_its_data(elastic_model, representation):
    """Normalization respects polynomial parameters and independent table grids."""

    def edit(file):
        reaction = file["reactions/elastic"]
        del reaction["large-angle/xs"]
        parameter = reaction.create_group("large-angle/xs")
        if representation == "polynomial":
            parameter.attrs["type"] = "polynomial"
            parameter["coefficients"] = [1.0, 1e-6]
            values = 1.0 + np.asarray(ENERGIES) * 1e-6
        else:
            parameter.attrs["type"] = "table"
            parameter["grid"] = [100.0, 1e6]
            parameter["value"] = [1.0, 100.0]
            interpolation(parameter, 2, laws=["log"])
            values = np.sqrt(np.asarray(ENERGIES) / 100.0)
        reaction["transport-xs/value"][...] = np.asarray(MEAN_DEFLECTION) * values
        reaction["xs/value"][...] = 2 * values

    simulation, data, elastic = elastic_model(edit=edit)
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    for energy, expected in zip(ENERGIES, MEAN_DEFLECTION):
        assert evaluate_data(
            energy, mean_deflection, simulation, data
        ) == pytest.approx(expected)


@pytest.mark.parametrize("energy", [100.0, 1e3, 1e4, 1e5, 1e6])
def test_sampled_mean_follows_the_source_large_angle_moment(elastic_model, energy):
    """Both grid points and intervening energies reproduce a consistent source."""
    simulation, data, elastic = elastic_model()
    distribution = simulation["distributions"][elastic["mu_ID"]]
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    assert mean_deflection["sub_type"] == DATA_TABLE

    state = np.zeros(1, dtype=type_.particle)
    state[0]["rng_seed"] = np.uint64(11)
    mu = np.array(
        [
            sample_elastic_cosine(
                energy, distribution, mean_deflection, state, simulation, data
            )
            for _ in range(12000)
        ]
    )
    expected = np.interp(energy, ENERGIES, MEAN_DEFLECTION)
    assert np.all((mu >= -1.0) & (mu <= MU_CUTOFF))
    assert np.mean(1.0 - mu) == pytest.approx(expected, rel=0.04)


@pytest.mark.parametrize(
    "energy, index", [(10.0, 0), (100.0, 0), (1e4, 1), (1e6, 2), (1e7, 2)]
)
def test_tabulated_energies_and_endpoints_use_the_source_cdf(
    elastic_model, energy, index
):
    """Sampling a knot or clamped endpoint preserves every source quantile."""
    simulation, data, elastic = elastic_model()
    distribution = simulation["distributions"][elastic["mu_ID"]]
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    multi = simulation["multi_table_distributions"][distribution["sub_ID"]]
    actual_state = np.zeros(1, dtype=type_.particle)
    actual_state[0]["rng_seed"] = np.uint64(11)
    source_state = actual_state.copy()
    for _ in range(20):
        actual = sample_elastic_cosine(
            energy, distribution, mean_deflection, actual_state, simulation, data
        )
        expected = sample_multi_table_at(index, rng.lcg(source_state), multi, data)
        assert actual == pytest.approx(expected, abs=1e-14)


def test_library_without_transport_keeps_table_selection(elastic_model):
    """Older libraries carry no mean deflection and keep the old sampler."""
    simulation, data, elastic = elastic_model(transport=False)
    assert simulation["data"][elastic["mean_deflection_ID"]]["sub_type"] == DATA_NONE


@pytest.mark.parametrize("near_boundary_draw", [1, 2])
def test_large_angle_sampling_stays_outside_the_forward_peak(
    elastic_model, near_boundary_draw
):
    """A quantile near the forward boundary must stay within the source support."""
    from mcdc.transport import rng

    simulation, data, elastic = elastic_model()
    distribution = simulation["distributions"][elastic["mu_ID"]]
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    state = np.zeros(1, dtype=type_.particle)
    # Invert the LCG so the selected draw is close to one in both execution modes.
    # Include near-one values for both the table-selection and CDF draws.
    modulus = int(rng.RNG_MOD)
    inverse = pow(int(rng.RNG_G), -1, modulus)
    seed = modulus - (1 << 20)
    for _ in range(near_boundary_draw):
        seed = ((seed - int(rng.RNG_C)) * inverse) % modulus
    state[0]["rng_seed"] = np.uint64(seed)
    mu = sample_elastic_cosine(
        1e5, distribution, mean_deflection, state, simulation, data
    )
    assert -1.0 <= mu <= MU_CUTOFF
    if near_boundary_draw == 2:
        assert MU_CUTOFF - mu < 1e-10


@pytest.mark.parametrize("seed", [1, 2])
def test_equal_source_moments_use_linear_energy_selection(elastic_model, seed):
    """Flat source moment data still gives a bounded mixture of the supplied CDFs."""
    from mcdc.transport import rng

    simulation, data, elastic = elastic_model(moments=[0.5] * 3)
    distribution = simulation["distributions"][elastic["mu_ID"]]
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    state = np.zeros(1, dtype=type_.particle)
    state[0]["rng_seed"] = np.uint64(seed)
    reference_state = state.copy()
    table_index = 1 if rng.lcg(reference_state) < 0.5 else 0
    quantile = rng.lcg(reference_state)
    mu = sample_elastic_cosine(
        5050.0, distribution, mean_deflection, state, simulation, data
    )
    lower_mu = 2.0 - 2.0 * MEAN_DEFLECTION[table_index] - MU_CUTOFF
    assert mu == pytest.approx(
        lower_mu + quantile * (MU_CUTOFF - lower_mu), rel=0.0, abs=1e-14
    )
