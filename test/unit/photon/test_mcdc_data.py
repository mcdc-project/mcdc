"""Photon mcdc-data reading and packed transport interpolation checks."""

import math

import h5py
import numpy as np
import pytest

from mcdc.constant import (
    PHOTON_REACTION_COHERENT_SCATTERING,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_TOTAL,
)
from mcdc.transport.data import evaluate_data_clamped
from mcdc.transport.physics.photon import native

from conftest import ENERGY_GRID, K_BINDING_ENERGY, make_photon


@pytest.mark.parametrize(
    "law", ["histogram", "linear", "semilog-x", "semilog-y", "log"]
)
def test_shared_grid_interpolation_laws(photon_model, law):
    """Nonlinear reaction values must be evaluated before summing categories."""

    def edit(file):
        xs = file["projectiles/photon/reactions/coherent-scattering/xs"]
        xs["value"][...] = [1.0, 16.0] + [16.0] * (len(ENERGY_GRID) - 2)
        xs["interpolation/laws"][...] = [law]

    simulation, data = photon_model(edit=edit)
    E = 200.0
    fraction = (
        math.log(E / 100.0) / math.log(10.0)
        if law in ("semilog-x", "log")
        else (E - 100.0) / 900.0
    )
    expected = (
        1.0
        if law == "histogram"
        else (
            16.0 ** fraction if law in ("semilog-y", "log") else 1.0 + 15.0 * fraction
        )
    )
    element = simulation["elements"][0]
    actual = native.total_micro_xs(
        PHOTON_REACTION_COHERENT_SCATTERING, E, element, simulation, data
    )
    assert actual == pytest.approx(expected)
    total = native.macro_xs(PHOTON_REACTION_TOTAL, make_photon(E), simulation, data)
    assert total == pytest.approx(expected + 2.0)


def test_cross_section_interpolation_regions(photon_model):
    def edit(file):
        xs = file["projectiles/photon/reactions/coherent-scattering/xs"]
        xs["value"][...] = [1.0, 16.0] + [100.0] * (len(ENERGY_GRID) - 2)
        del xs["interpolation"]
        xs["interpolation/laws"] = np.array(
            ["histogram", "linear"], dtype=h5py.string_dtype()
        )
        xs["interpolation/boundaries"] = [2, len(ENERGY_GRID)]

    simulation, data = photon_model(edit=edit)
    element = simulation["elements"][0]
    for energy, expected in [(200.0, 1.0), (44500.0, 58.0)]:
        assert native.total_micro_xs(
            PHOTON_REACTION_COHERENT_SCATTERING, energy, element, simulation, data
        ) == pytest.approx(expected)


@pytest.mark.parametrize("linear_fast_path", [True, False])
def test_absorption_edge_uses_distinct_left_and_right_limits(
    photon_model, linear_fast_path
):
    def edit(file):
        if linear_fast_path:
            file["projectiles/photon/reactions/photoelectric/xs/value"][0] = 0.0
            file["projectiles/photon/reactions/pair-production-nuclear-field/xs/value"][
                0
            ] = 0.0
        file["projectiles/photon/reactions/coherent-scattering/xs/value"][...] = [
            1.0,
            2.0,
            3.0,
            7.0,
        ] + [11.0] * (len(ENERGY_GRID) - 4)

    simulation, data = photon_model(edit=edit)
    element = simulation["elements"][0]
    assert bool(element["photon_xs_is_linear"]) == linear_fast_path
    for energy, expected in [
        (np.nextafter(K_BINDING_ENERGY, -np.inf), 3.0),
        (K_BINDING_ENERGY, 7.0),
        (np.nextafter(K_BINDING_ENERGY, np.inf), 7.0),
        (1.0, 1.0),
        (1.0e10, 11.0),
    ]:
        assert native.total_micro_xs(
            PHOTON_REACTION_COHERENT_SCATTERING, energy, element, simulation, data
        ) == pytest.approx(expected)


def test_table_subshell_edge_and_threshold(photon_model):
    """A repeated first energy must give zero below and the upper value at the edge."""

    def edit(file):
        xs = file["projectiles/photon/reactions/photoelectric/subshells/K/xs"]
        del xs["grid"], xs["value"]
        xs["grid"] = [K_BINDING_ENERGY, K_BINDING_ENERGY, 1.0e8]
        xs["value"] = [0.0, 3.0, 3.0]
        xs["interpolation/boundaries"][...] = [3]

    simulation, data = photon_model(edit=edit)
    reaction = simulation["photon_photoelectric_reactions"][0]
    import mcdc.mcdc_get as get

    table_ID = get.photon_photoelectric_reaction.subshell_x_IDs(0, reaction, data)
    table = simulation["data"][table_ID]
    assert (
        native.subshell_micro_xs(
            np.nextafter(K_BINDING_ENERGY, -np.inf), table, simulation, data
        )
        == 0.0
    )
    assert native.subshell_micro_xs(K_BINDING_ENERGY, table, simulation, data) == 3.0
    assert native.subshell_micro_xs(1.0e10, table, simulation, data) == 3.0
    assert evaluate_data_clamped(K_BINDING_ENERGY, table, simulation, data) == 3.0


# ======================================================================================
# Reaction identity and format validation
# ======================================================================================


def test_category_dispatch_and_both_pair_fields(photon_model):
    """Names and optional source identifiers do not drive reaction dispatch."""

    def edit(file):
        reactions = file["projectiles/photon/reactions"]
        file.copy(reactions["pair-production-nuclear-field"], reactions, name="triplet")
        reactions["triplet/field"][()] = "electron"
        reactions["triplet"].attrs["ENDF-MT"] = 515
        reactions.move("coherent-scattering", "arbitrary-source-label")
        del reactions["arbitrary-source-label"].attrs["ENDF-MT"]

    simulation, data = photon_model(edit=edit)
    element = simulation["elements"][0]
    assert simulation["N_photon_pair_production_reaction"] == 2
    assert set(simulation["photon_pair_production_reactions"]["electron_field"]) == {
        True,
        False,
    }
    assert (
        native.total_micro_xs(
            PHOTON_REACTION_PAIR_PRODUCTION, 1.0e7, element, simulation, data
        )
        == 16.0
    )


@pytest.mark.parametrize(
    "invalid", ["format", "category", "grid", "negative-xs", "cross-products"]
)
def test_invalid_photon_data_is_rejected(photon_model, invalid):
    def edit(file):
        photon = file["projectiles/photon"]
        coherent = photon["reactions/coherent-scattering"]
        if invalid == "format":
            file.attrs["format-version"] = "99.0.0"
        elif invalid == "category":
            coherent.attrs["category"] = "unknown"
        elif invalid == "grid":
            photon["xs-energy-grid"][0] = 1.0e10
        elif invalid == "negative-xs":
            coherent["xs/value"][0] = -1.0
        else:
            coherent.create_group("cross-products/0")

    with pytest.raises(ValueError):
        photon_model(edit=edit)
