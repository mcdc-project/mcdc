"""Interpolation and conditional-sampling checks for mcdc-data representations."""

import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from mcdc.object_.hdf5_reader import read_distribution
from mcdc.object_.distribution import DistributionTable, DistributionMultiTable
from mcdc.transport import distribution as sampling

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from data_library_generator.common.hdf5_writer import pdf_tables


@pytest.mark.parametrize("law", [1, 2, 3, 4, 5])
def test_density_interpolation_quantiles(law, mock_rng_sequence, prepare_simulation):
    distribution = DistributionTable([1.0, 4.0], pdf=[0.2, 1.1], interpolations=law)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    table = simulation["table_distributions"][distribution.sub_ID]

    def density(x):
        fraction = math.log(x) / math.log(4.0) if law in (3, 5) else (x - 1.0) / 3.0
        if law == 1:
            return 0.2
        if law in (4, 5):
            return 0.2 * (1.1 / 0.2) ** fraction
        return 0.2 + fraction * 0.9

    integral = quad(density, 1.0, 4.0)[0]
    for probability in (0.01, 0.25, 0.9):
        expected = brentq(
            lambda x: quad(density, 1.0, x)[0] / integral - probability, 1.0, 4.0
        )
        sampled = sampling.sample_table(
            table, mock_rng_sequence(probability), simulation, data
        )
        assert sampled == pytest.approx(expected, abs=1.0e-10)


def test_density_regions_and_repeated_coordinates(
    mock_rng_sequence, prepare_simulation
):
    distribution = DistributionTable(
        [1.0, 2.0, 2.0, 4.0],
        pdf=[1.0, 1.0, 2.0, 4.0],
        interpolations=[1, 5],
        interpolation_boundaries=[2, 4],
    )
    # The duplicate value is a density jump: its interval carries zero mass.
    np.testing.assert_allclose(
        distribution.pdf.aux[0], [0.0, 1.0 / 7.0, 1.0 / 7.0, 1.0]
    )
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    table = simulation["table_distributions"][distribution.sub_ID]
    # Integral from 2 to x is (x**2 - 4)/2; total density integral is 7.
    sampled = sampling.sample_table(table, mock_rng_sequence(0.5), simulation, data)
    assert sampled == pytest.approx(3.0)


def test_histogram_incident_energy_and_endpoints(mock_rng_sequence, prepare_simulation):
    distribution = DistributionMultiTable(
        [1.0, 3.0],
        value=[[10.0, 20.0], [100.0, 200.0]],
        pdf=[[1.0, 1.0], [1.0, 1.0]],
        interpolations=1,
    )
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    table = simulation["multi_table_distributions"][distribution.sub_ID]
    for energy, expected in [
        (0.0, 12.5),
        (1.0, 12.5),
        (2.0, 12.5),
        (3.0, 125.0),
        (4.0, 125.0),
    ]:
        # Interior histogram selection consumes the selection variate first.
        values = (0.9, 0.25) if 1.0 < energy < 3.0 else (0.25,)
        result = sampling._sample_multi_table(
            energy, mock_rng_sequence(*values), table, simulation, data, True
        )
        assert result == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["evaporation", "maxwellian"])
@pytest.mark.parametrize("temperature_kind", ["constant", "polynomial"])
def test_analytic_temperature_data(
    tmp_path, kind, temperature_kind, mock_rng_sequence, prepare_simulation
):
    with h5py.File(tmp_path / "spectrum.h5", "w") as group:
        group.attrs["type"] = kind
        group["restriction-energy"] = 0.0
        temperature = group.create_group("temperature")
        temperature.attrs["type"] = "polynomial"
        temperature["coefficients"] = (
            [1.0] if temperature_kind == "constant" else [1.0, 0.0]
        )
        distribution = read_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    sample = sampling.sample_distribution(
        10.0,
        simulation["distributions"][distribution.ID],
        mock_rng_sequence(0.5, 0.5, 0.5),
        simulation,
        data,
    )
    assert 0.0 < sample < 10.0


@pytest.mark.parametrize("kind", ["kalbach-mann", "energy-angle-table"])
def test_correlated_histogram_tables(
    tmp_path, kind, mock_rng_sequence, prepare_simulation
):
    """Preserve histogram densities, conditional angles, and parameter Data."""

    with h5py.File(tmp_path / "spectrum.h5", "w") as group:
        group.attrs["type"] = kind
        group["energy"] = [10.0, 20.0]
        tables = group.create_group("tables")
        pdf_tables(
            tables.create_group("outgoing-energy"),
            [
                (
                    np.array([1.0, 2.0, 4.0]) * f,
                    np.array([0.2, 0.4, 0.0]) / f,
                    ["histogram"],
                    None,
                )
                for f in (1, 2)
            ],
        )
        if kind == "kalbach-mann":
            for name, value in (("precompound-factor", 1.0), ("angular-slope", 0.0)):
                for i in range(2):
                    parameter = tables.create_group(f"{name}/{i}")
                    parameter.attrs["type"] = "polynomial"
                    parameter["coefficients"] = [value]
        else:
            pdf_tables(
                tables.create_group("cosine-distributions"),
                [
                    (
                        [-1.0, 0.0] if j < 2 else [0.0, 1.0],
                        [1.0, 1.0],
                        ["histogram"],
                        None,
                    )
                    for i in range(2)
                    for j in range(3)
                ],
                unit=None,
            )
        distribution = read_distribution(group)
    np.testing.assert_allclose(distribution.cdf, [0.0, 0.2, 1.0] * 2)
    np.testing.assert_allclose(distribution.pdf, [0.2, 0.4, 0.0, 0.1, 0.2, 0.0])
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    base = simulation["distributions"][distribution.ID]
    for energy, expected in [(5.0, 3.5), (10.0, 3.5), (20.0, 7.0), (25.0, 7.0)]:
        E, mu = sampling.sample_correlated_distribution_with_scale(
            energy, base, mock_rng_sequence(0.1, 0.8, 0.75, 0.75), simulation, data
        )
        assert E == pytest.approx(expected)
        assert mu == pytest.approx(0.5 if kind == "kalbach-mann" else -0.25)


@pytest.mark.parametrize("law", ["histogram", "linear"])
@pytest.mark.parametrize("incident_draw", [0.1, 0.9])
@pytest.mark.parametrize("energy_draw", [0.21, 0.55, 0.6, 0.61, 0.99])
def test_law61_conditional_angle_selection(
    tmp_path, law, incident_draw, energy_draw, mock_rng_sequence, prepare_simulation
):
    """Follow MCNP LAW 61's lower-point or nearest-CDF-point angular selection."""
    density = np.array([0.2, 0.4, 0.4] if law == "histogram" else [0.2, 0.2, 0.6])
    with h5py.File(tmp_path / "law61.h5", "w") as group:
        group.attrs["type"] = "energy-angle-table"
        group["energy"] = [10.0, 20.0]
        tables = group.create_group("tables")
        pdf_tables(
            tables.create_group("outgoing-energy"),
            [
                ([0.0, 1.0, 3.0], density, [law], None),
                ([100.0, 102.0, 106.0], density / 2, [law], None),
            ],
        )
        # Disjoint cosine supports identify which incident/outgoing point was used.
        supports = [
            [-1.0, -0.75],
            [-0.75, -0.5],
            [-0.5, -0.25],
            [0.25, 0.5],
            [0.5, 0.75],
            [0.75, 1.0],
        ]
        pdf_tables(
            tables.create_group("cosine-distributions"),
            [(support, [4.0, 4.0], ["linear"], None) for support in supports],
            unit=None,
        )
        distribution = read_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    rng = mock_rng_sequence(incident_draw, energy_draw, 0.5)
    energy, cosine = sampling.sample_correlated_distribution_with_scale(
        15.0, simulation["distributions"][distribution.ID], rng, simulation, data
    )

    # Both selected tables have CDF [0, 0.2, 1]. In their second interval,
    # u = 0.55 is nearer the lower CDF point, while the linear-PDF quantile
    # lies nearer the upper energy point. CDF midpoint 0.6 selects the lower.
    point = 2 if law == "linear" and energy_draw > 0.6 else 1
    incident = 1 if incident_draw < 0.5 else 0
    expected_cosine = sum(supports[3 * incident + point]) / 2
    assert cosine == pytest.approx(expected_cosine)

    if law == "histogram":
        raw_energy = 1 + (energy_draw - 0.2) / 0.4
    else:
        raw_energy = 1 + (math.sqrt(0.04 + 0.4 * (energy_draw - 0.2)) - 0.2) / 0.2
    # Interpolated support at incident energy 15 is [50, 54.5].
    assert energy == pytest.approx(50 + 1.5 * raw_energy)
    assert rng[0]["idx"] == 3
