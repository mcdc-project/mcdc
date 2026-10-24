"""Sampling checks for proton readers that build distribution objects."""

import h5py
import numpy as np
import pytest

from mcdc.object_.proton_reaction import (
    set_angular_distribution,
    set_energy_distribution,
)
from mcdc.transport.data import evaluate_data_clamped
from mcdc.transport.distribution import (
    sample_distribution,
    sample_distribution_with_scale,
)


@pytest.mark.parametrize("layout", ["flattened", "subgroups", "isotropic-subgroup"])
def test_proton_angular_tables(tmp_path, prepare_simulation, mock_rng_sequence, layout):
    with h5py.File(tmp_path / "angles.h5", "w") as group:
        group.attrs["type"] = "tabulated"
        if layout == "flattened":
            group["energy"], group["offset"] = [1.0, 3.0], [0, 2]
            group["value"], group["pdf"] = [-1.0, 0.0, 0.0, 1.0], [1.0] * 4
        else:
            group["incident_energies"] = [1.0, 3.0]
            for index, cosines in enumerate([[-1.0, 0.0], [0.0, 1.0]], start=1):
                table = group.create_group(f"E_in_{index}")
                if layout == "isotropic-subgroup" and index == 2:
                    table.attrs["type"] = "isotropic"
                else:
                    table["cosines"], table["pdf"] = cosines, [1.0, 1.0]
        distribution = set_angular_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    result = sample_distribution(
        2.0e6,
        simulation["distributions"][distribution.ID],
        mock_rng_sequence(0.3, 0.25),
        simulation,
        data,
    )
    assert result == pytest.approx(-0.5 if layout == "isotropic-subgroup" else 0.25)


def test_proton_table_spectrum(tmp_path, prepare_simulation, mock_rng_sequence):
    with h5py.File(tmp_path / "spectrum.h5", "w") as group:
        group.attrs["type"] = "tabulated"
        group["energy"], group["offset"] = [1.0, 3.0], [0, 2]
        group["value"], group["pdf"] = [0.1, 0.2, 0.3, 0.5], [10.0, 10.0, 5.0, 5.0]
        distribution = set_energy_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    # At the midpoint incident energy, the interpolated support is [0.2, 0.35] MeV.
    result = sample_distribution_with_scale(
        2.0e6,
        simulation["distributions"][distribution.ID],
        mock_rng_sequence(0.3, 0.25),
        simulation,
        data,
    )
    assert result == pytest.approx(0.2375e6)


@pytest.mark.parametrize("kind", ["evaporation", "maxwellian"])
def test_proton_spectrum_temperature_data(tmp_path, prepare_simulation, kind):
    with h5py.File(tmp_path / "spectrum.h5", "w") as group:
        group.attrs["type"] = kind
        group["temperature_energy_grid"], group["temperature"] = [1.0, 4.0], [0.1, 0.8]
        group["restriction_energy"] = 0.05
        if kind == "maxwellian":
            group["temperature_interpolation"] = np.bytes_("log")
        distribution = set_energy_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    temperature = simulation["data"][distribution.nuclear_temperature.ID]
    expected = 0.1e6 * 2.0**1.5 if kind == "maxwellian" else (0.1 + 0.7 / 3.0) * 1.0e6
    assert evaluate_data_clamped(2.0e6, temperature, simulation, data) == pytest.approx(
        expected
    )
    assert distribution.restriction_energy == 0.05e6
