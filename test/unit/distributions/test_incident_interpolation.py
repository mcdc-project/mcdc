"""Incident interpolation is independent of each conditional PDF's interpolation."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from mcdc.object_.distribution import DistributionMultiTable
from mcdc.object_.hdf5_reader import read_distribution
from mcdc.transport.distribution import (
    _sample_multi_table,
    sample_correlated_distribution_with_scale,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from data_library_generator.common.hdf5_writer import multi_table


def source_distribution(kind, grid, laws, boundaries, *, invalid_table=None):
    """Construct actual ACEtk records with distinguishable endpoint spectra."""
    ACEtk = pytest.importorskip("ACEtk")
    distributions = []
    for i, energy in enumerate(grid):
        low, high = 100.0 * i, 100.0 * i + 10.0 * (i + 1)
        values = [low / 1e6, high / 1e6]
        if i == invalid_table:
            values.reverse()
        pdf = [1e6 / (high - low)] * 2
        if kind == 4:
            table = ACEtk.continuous.TabulatedEnergyDistribution(
                energy / 1e6, 2, values, pdf, [0.0, 1.0]
            )
        elif kind == 44:
            table = ACEtk.continuous.TabulatedKalbachMannDistribution(
                energy / 1e6, 2, values, pdf, [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]
            )
        else:
            cosines = [-1.0, 0.0] if i == 0 else [0.0, 1.0]
            angular = [
                ACEtk.continuous.TabulatedAngularDistributionWithProbability(
                    value, density, cumulative, 2, cosines, [1.0, 1.0], [0.0, 1.0]
                )
                for value, density, cumulative in zip(values, pdf, [0.0, 1.0])
            ]
            table = ACEtk.continuous.TabulatedEnergyAngleDistribution(
                energy / 1e6, 2, angular
            )
        distributions.append(table)
    cls = {
        4: ACEtk.continuous.OutgoingEnergyDistributionData,
        44: ACEtk.continuous.KalbachMannDistributionData,
        61: ACEtk.continuous.EnergyAngleDistributionData,
    }[kind]
    return cls(boundaries, laws, distributions)


@pytest.mark.parametrize("kind", [4, 44, 61])
def test_invalid_outgoing_table_reports_law_path_and_incident_energy(tmp_path, kind):
    from data_library_generator.common.ace_converter import write_energy_distribution

    source = source_distribution(kind, [5.0e6, 7.4e6], [], [], invalid_table=1)
    with h5py.File(tmp_path / "invalid.h5", "w") as file:
        distribution = file.create_group("reactions/test/energy-distribution")
        with pytest.raises(ValueError) as error:
            write_energy_distribution(distribution, source)
    message = str(error.value)
    assert "/reactions/test/energy-distribution" in message
    assert f"ACE LAW {kind}" in message
    assert "outgoing-energy table 1 (E_in = 7.4 MeV)" in message
    assert "PDF coordinates must be nondecreasing" in message


def test_invalid_conditional_angle_reports_incident_and_outgoing_energy(tmp_path):
    ACEtk = pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import write_energy_distribution

    tables = []
    for incident in (5.0, 7.4):
        angles = [
            ACEtk.continuous.TabulatedAngularDistributionWithProbability(
                outgoing,
                1.0,
                cumulative,
                2,
                [1.0, -1.0] if incident == 7.4 and outgoing == 1.0 else [-1.0, 1.0],
                [0.5, 0.5],
                [0.0, 1.0],
            )
            for outgoing, cumulative in ((0.0, 0.0), (1.0, 1.0))
        ]
        tables.append(
            ACEtk.continuous.TabulatedEnergyAngleDistribution(incident, 2, angles)
        )
    source = ACEtk.continuous.EnergyAngleDistributionData([], [], tables)
    with h5py.File(tmp_path / "invalid-cosine.h5", "w") as file:
        with pytest.raises(ValueError) as error:
            write_energy_distribution(file, source)
    message = str(error.value)
    assert "ACE LAW 61" in message
    assert "cosine table 1 (E_in = 7.4 MeV, E_out = 1 MeV)" in message
    assert "PDF coordinates must be nondecreasing" in message


def convert(tmp_path, kind, grid, laws, boundaries):
    source = source_distribution(kind, grid, laws, boundaries)
    from data_library_generator.common.ace_converter import write_energy_distribution

    with h5py.File(tmp_path / f"law{kind}.h5", "w") as group:
        write_energy_distribution(group, source)
        names = {1: "histogram", 2: "linear"}
        assert group["interpolation/laws"].asstr()[()].tolist() == [
            names[law] for law in (laws or [2])
        ]
        np.testing.assert_array_equal(
            group["interpolation/boundaries"], boundaries or [len(grid)]
        )
        return read_distribution(group)


def draw(
    distribution, kind, energy, prepare_simulation, mock_rng_sequence, selector=0.4
):
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    rng = mock_rng_sequence(selector, 0.25, 0.75, 0.25)
    if kind == 4:
        table = simulation["multi_table_distributions"][distribution.sub_ID]
        return _sample_multi_table(energy, rng, table, simulation, data, True), None
    base = simulation["distributions"][distribution.ID]
    return sample_correlated_distribution_with_scale(
        energy, base, rng, simulation, data
    )


@pytest.mark.parametrize("kind", [4, 44, 61])
@pytest.mark.parametrize("law,expected", [(1, 2.5), (2, 28.125), (None, 28.125)])
def test_ace_conversion_and_incident_sampling(
    tmp_path, kind, law, expected, prepare_simulation, mock_rng_sequence
):
    model = convert(
        tmp_path,
        kind,
        [1.0, 9.0],
        [law] if law is not None else [],
        [2] if law is not None else [],
    )
    outgoing, cosine = draw(model, kind, 3.0, prepare_simulation, mock_rng_sequence)
    assert outgoing == pytest.approx(expected)
    if kind == 44:
        assert cosine == pytest.approx(-0.5)
    elif kind == 61:
        assert cosine == pytest.approx(-0.25)


@pytest.mark.parametrize("kind", [4, 44, 61])
def test_multiple_incident_regions_and_endpoint_clamping(
    tmp_path, kind, prepare_simulation, mock_rng_sequence
):
    model = convert(tmp_path, kind, [1.0, 3.0, 9.0, 81.0], [1, 2], [2, 4])
    for energy, expected in [
        (0.0, 2.5),
        (2.0, 2.5),
        (3.0, 105.0),
        (9.0, 207.5),
        (45.0, 258.75),
        (100.0, 310.0),
    ]:
        # Endpoint sampling by MultiTable consumes only the PDF draw.
        if kind == 4 and energy in (0.0, 100.0):
            container, data = prepare_simulation(objects=[model])
            record = container[0]["multi_table_distributions"][model.sub_ID]
            outgoing = _sample_multi_table(
                energy, mock_rng_sequence(0.25), record, container[0], data, True
            )
        else:
            outgoing, _ = draw(
                model, kind, energy, prepare_simulation, mock_rng_sequence
            )
        assert outgoing == pytest.approx(expected)


@pytest.mark.parametrize("kind", [4, 44, 61])
def test_repeated_incident_grid_point_uses_right_hand_spectrum(
    tmp_path, kind, prepare_simulation, mock_rng_sequence
):
    model = convert(tmp_path, kind, [1.0, 3.0, 3.0, 9.0], [1, 2], [2, 4])
    outgoing, _ = draw(model, kind, 3.0, prepare_simulation, mock_rng_sequence)
    assert outgoing == pytest.approx(207.5)


@pytest.mark.parametrize("kind", [4, 44, 61])
@pytest.mark.parametrize("law", [3, 4, 5])
def test_unsupported_incident_interpolation_is_rejected(tmp_path, kind, law):
    pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import write_energy_distribution

    source = source_distribution(kind, [1.0, 9.0], [law], [2])
    with h5py.File(tmp_path / "unsupported.h5", "w") as group:
        with pytest.raises(ValueError, match="histogram or linear"):
            write_energy_distribution(group, source)


@pytest.mark.parametrize("law", [3, 4, 5])
def test_object_and_writer_reject_unsupported_incident_rules(tmp_path, law):
    with pytest.raises(ValueError, match="histogram or linear"):
        DistributionMultiTable(
            [1.0, 9.0],
            value=[[0.0, 1.0]] * 2,
            pdf=[[1.0, 1.0]] * 2,
            interpolations=law,
        )
    name = {3: "semilog-x", 4: "semilog-y", 5: "log"}[law]
    with h5py.File(tmp_path / "unsupported.h5", "w") as group:
        with pytest.raises(ValueError, match="histogram or linear"):
            multi_table(
                group,
                [1.0, 9.0],
                [([0.0, 1.0], [1.0, 1.0], None, None)] * 2,
                laws=[name],
            )
