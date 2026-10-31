import h5py
import numpy as np
import pytest

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
from mcdc.constant import DATA_NONE, DATA_TABLE
from mcdc.transport.physics.electron.native import sample_scaled_cosine

#: Mean of 1 - mu at each cross-section grid energy (transport / elastic)
MEAN_DEFLECTION = [1.0, 0.1, 0.01]


def _write_elastic_library(path, transport):
    # A small synthetic library exercising the real HDF5 reader and packing.
    # The single cosine table, at 100 eV, is uniform in mu (mean 1 - mu = 1).
    with h5py.File(path, "w") as file:
        file["atomic_weight_ratio"] = 1.0
        file["atomic_number"] = 1
        electron = file.create_group("electron_reactions")
        electron["xs_energy_grid"] = [100.0, 1.0e4, 1.0e6]
        reaction = electron.create_group("elastic_scattering/MT-526")
        reaction.attrs["MT"] = 526
        reaction["xs"] = [1.0, 1.0, 1.0]
        reaction["xs"].attrs["offset"] = 0
        reaction["reference_frame"] = np.bytes_("LAB")
        if transport:
            reaction["transport"] = MEAN_DEFLECTION
        large_angle = reaction.create_group("large_angle")
        large_angle["xs_energy"] = [100.0, 1.0e4, 1.0e6]
        large_angle["xs"] = [1.0, 1.0, 1.0]
        cosine = large_angle.create_group("scattering_cosine")
        cosine["energy_grid"] = [100.0, 1.0e6]
        cosine["energy_offset"] = [0, 2]
        cosine["value"] = [-1.0, 1.0, -1.0, 1.0]
        cosine["CDF"] = [0.0, 1.0, 0.0, 1.0]


@pytest.fixture
def elastic_model(tmp_path, monkeypatch, prepare_simulation):
    def build(transport=True):
        _write_elastic_library(tmp_path / "H.h5", transport)
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


def test_cosine_is_rescaled_to_the_transport_mean_deflection(elastic_model):
    """Between sparse tables, <1 - mu> follows transport / elastic."""
    simulation, data, elastic = elastic_model()
    distribution = simulation["distributions"][elastic["mu_ID"]]
    mean_deflection = simulation["data"][elastic["mean_deflection_ID"]]
    assert mean_deflection["sub_type"] == DATA_TABLE

    state = np.zeros(1, dtype=type_.particle)
    state[0]["rng_seed"] = np.uint64(11)
    for E, expected in [
        (100.0, 1.0),
        (1.0e4, 0.1),
        (1.0e5, 0.1 - 0.09 * (1.0e5 - 1.0e4) / (1.0e6 - 1.0e4)),
    ]:
        mu = np.array(
            [
                sample_scaled_cosine(
                    E, distribution, mean_deflection, state, simulation, data
                )
                for _ in range(4000)
            ]
        )
        assert np.all((mu >= -1.0) & (mu <= 1.0))
        assert np.mean(1.0 - mu) == pytest.approx(expected, rel=0.05)


def test_library_without_transport_keeps_table_selection(elastic_model):
    """Older libraries carry no mean deflection and keep the old sampler."""
    simulation, data, elastic = elastic_model(transport=False)
    assert simulation["data"][elastic["mean_deflection_ID"]]["sub_type"] == DATA_NONE
