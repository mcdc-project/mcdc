"""Mixtures compose probability Data and existing scalar or joint Distributions."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from mcdc.constant import REFERENCE_FRAME_COM, REFERENCE_FRAME_LAB
from mcdc.object_.data import DataPolynomial, DataTable
from mcdc.object_.distribution import (
    DistributionIsotropic,
    DistributionNone,
    DistributionLevelScattering,
    DistributionMixture,
    DistributionNBody,
    validate_angle_energy,
)
from mcdc.object_.hdf5_reader import read_distribution
from mcdc.transport.data import evaluate_data_clamped
from mcdc.transport.distribution import (
    sample_distribution_with_scale,
    sample_correlated_distribution_with_scale,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))


def components():
    return [
        DistributionLevelScattering(0.0, 0.2),
        DistributionLevelScattering(0.0, 0.8),
    ]


def sample(model, energy, draws, prepare_simulation, mock_rng_sequence):
    container, data = prepare_simulation(objects=[model])
    simulation = container[0]
    state = mock_rng_sequence(*draws)
    value = sample_distribution_with_scale(
        energy, simulation["distributions"][model.ID], state, simulation, data
    )
    return value, state


@pytest.mark.parametrize(
    "energy,coefficient",
    [(0.0, 0.8), (2.0, 0.8), (3.0, 0.8), (np.sqrt(27.0), 0.2), (9.0, 0.2), (20.0, 0.2)],
)
def test_independent_probability_grids_and_multiple_interpolation_regions(
    energy, coefficient, prepare_simulation, mock_rng_sequence
):
    middle = np.sqrt(27.0)
    model = DistributionMixture(
        [
            DataTable([1, 3, 9], [0.2, 0.2, 0.8], [1, 3], [2, 3]),
            DataTable([1, 3, middle, 9], [0.8, 0.8, 0.5, 0.2], [1, 3], [2, 4]),
        ],
        components(),
    )
    value, state = sample(model, energy, [0.4], prepare_simulation, mock_rng_sequence)
    assert value == pytest.approx(coefficient * energy)
    assert state[0]["idx"] == 1


@pytest.mark.parametrize("energy,expected", [(1.0, 0.8), (5.0, 1.0)])
def test_polynomial_probability_data(
    energy, expected, prepare_simulation, mock_rng_sequence
):
    model = DistributionMixture(
        [DataPolynomial([0.25, 0.05]), DataPolynomial([0.75, -0.05])], components()
    )
    value, _ = sample(model, energy, [0.4], prepare_simulation, mock_rng_sequence)
    assert value == pytest.approx(expected)


def test_nested_mixtures_and_zero_probability_endpoints(
    prepare_simulation, mock_rng_sequence
):
    inner = DistributionMixture(
        [DataPolynomial([1.0]), DataPolynomial([0.0])], components()
    )
    outer = DistributionMixture(
        [DataPolynomial([0.0]), DataPolynomial([1.0])], [components()[1], inner]
    )
    value, state = sample(
        outer, 10.0, [1.0, 1.0], prepare_simulation, mock_rng_sequence
    )
    assert value == pytest.approx(2.0)
    assert state[0]["idx"] == 2


@pytest.mark.parametrize("frame", [REFERENCE_FRAME_COM, REFERENCE_FRAME_LAB])
def test_joint_mixture_sampling_and_reaction_frame_validation(
    frame, prepare_simulation, mock_rng_sequence
):
    model = DistributionMixture(
        [DataPolynomial([0.0]), DataPolynomial([1.0])],
        [
            DistributionNBody(3, 4.0, [0, 1], [1, 1]),
            DistributionNBody(3, 8.0, [0, 1], [1, 1]),
        ],
    )
    if frame == REFERENCE_FRAME_LAB:
        with pytest.raises(ValueError, match="COM reference frame"):
            validate_angle_energy(DistributionNone(), [model], "mixture", frame)
        return
    validate_angle_energy(DistributionNone(), [model], "mixture", frame)
    with pytest.raises(ValueError, match="correlation"):
        validate_angle_energy(DistributionIsotropic(), [model], "mixture", frame)
    container, data = prepare_simulation(objects=[model])
    simulation = container[0]
    energy, cosine = sample_correlated_distribution_with_scale(
        20.0,
        simulation["distributions"][model.ID],
        mock_rng_sequence(0.3, 0.25, 0.75),
        simulation,
        data,
        4.0,
        -2.0,
    )
    assert energy == pytest.approx(0.25 * (7 / 8) * (4 / 5 * 20.0 - 2.0))
    assert cosine == pytest.approx(0.5)


@pytest.mark.parametrize(
    "probabilities", [[0.0, 0.0], [0.2, 0.2], [-0.1, 1.1], [np.nan, 1.0], [np.inf, 0.0]]
)
def test_invalid_constant_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilit|finite"):
        DistributionMixture([DataPolynomial([p]) for p in probabilities], components())


def test_varying_probabilities_are_validated_at_the_sampled_energy(
    prepare_simulation, mock_rng_sequence
):
    model = DistributionMixture(
        [DataTable([1, 3], [0.2, 0.8], 4), DataTable([1, 3], [0.8, 0.2], 4)],
        components(),
    )
    with pytest.raises(ValueError, match="sum to one at the incident energy"):
        sample(model, 2.0, [0.4], prepare_simulation, mock_rng_sequence)


def test_negative_polynomial_probability_is_rejected_at_sampling(
    prepare_simulation, mock_rng_sequence
):
    model = DistributionMixture(
        [DataPolynomial([0.25, 0.05]), DataPolynomial([0.75, -0.05])], components()
    )
    with pytest.raises(ValueError, match="nonnegative"):
        sample(model, 20.0, [0.4], prepare_simulation, mock_rng_sequence)


def test_mixture_rejects_mixed_independent_and_joint_components():
    mixture = DistributionMixture(
        [DataPolynomial([0.5]), DataPolynomial([0.5])],
        [components()[0], DistributionNBody(3, 4.0, [0, 1], [1, 1])],
    )
    for angle in [DistributionNone(), DistributionIsotropic()]:
        with pytest.raises(ValueError, match="correlation"):
            validate_angle_energy(angle, [mixture], "mixture", REFERENCE_FRAME_COM)


def test_ace_mixture_keeps_independent_grids_and_default_linear_interpolation(
    tmp_path, prepare_simulation, mock_rng_sequence
):
    ace = pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import write_energy_distribution

    probabilities = [
        ace.continuous.DistributionProbability([1.0, 5.0], [0.2, 0.8]),
        ace.continuous.DistributionProbability([1.0, 3.0, 5.0], [0.8, 0.5, 0.2]),
    ]
    spectra = [
        ace.continuous.LevelScatteringDistribution(1.0, 5.0, 0.0, value)
        for value in [0.2, 0.8]
    ]
    source = ace.continuous.MultiDistributionData(probabilities, spectra)
    with h5py.File(tmp_path / "mixture.h5", "w") as group:
        write_energy_distribution(group, source)
        for i, grid in enumerate([[1.0, 5.0], [1.0, 3.0, 5.0]]):
            table = group[f"components/{i}/probability"]
            assert table.attrs["type"] == "table"
            assert table["grid"].attrs["unit"] == "eV"
            assert "unit" not in table["value"].attrs
            np.testing.assert_array_equal(table["grid"], np.asarray(grid) * 1e6)
            assert table["interpolation/laws"].asstr()[()].tolist() == ["linear"]
        model = read_distribution(group)
    for selector, expected in [(0.4, 0.6e6), (0.6, 2.4e6)]:
        value, _ = sample(
            model, 3.0e6, [selector], prepare_simulation, mock_rng_sequence
        )
        assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "law,expected", [(1, 0.2), (2, 0.35), (3, 0.5), (4, 0.2 * 4**0.25), (5, 0.4)]
)
def test_ace_probability_data_retains_each_interpolation_law(
    tmp_path, law, expected, prepare_simulation
):
    ace = pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import (
        _write_distribution_probability,
    )
    from mcdc.object_.hdf5_reader import read_data

    probability = ace.continuous.DistributionProbability(
        [2], [law], [1.0, 9.0], [0.2, 0.8]
    )
    with h5py.File(tmp_path / "probability.h5", "w") as group:
        _write_distribution_probability(group, probability)
        model = read_data(group)
    container, data = prepare_simulation(objects=[model])
    simulation = container[0]
    value = evaluate_data_clamped(3.0e6, simulation["data"][model.ID], simulation, data)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("probabilities", [[0.3, 0.7], [0.333333] * 3])
def test_ace_constant_probabilities_use_polynomials(tmp_path, probabilities):
    ace = pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import write_energy_distribution

    source = ace.continuous.MultiDistributionData(
        [
            ace.continuous.DistributionProbability([1.0, 3.0], [p, p])
            for p in probabilities
        ],
        [
            ace.continuous.LevelScatteringDistribution(1.0, 3.0, 0.0, 0.5)
            for _ in probabilities
        ],
    )
    with h5py.File(tmp_path / "constant.h5", "w") as group:
        write_energy_distribution(group, source)
        for i, p in enumerate(probabilities):
            probability = group[f"components/{i}/probability"]
            assert probability.attrs["type"] == "polynomial"
            np.testing.assert_array_equal(probability["coefficients"], [p])
            assert "unit" not in probability["coefficients"].attrs
        assert isinstance(read_distribution(group), DistributionMixture)


@pytest.mark.parametrize("probability", [0.333333, 0.3333333])
def test_source_rounded_probabilities_at_tolerance_boundary(
    probability, prepare_simulation, mock_rng_sequence
):
    model = DistributionMixture(
        [DataPolynomial([probability]) for _ in range(3)],
        [*components(), components()[0]],
    )
    value, _ = sample(model, 10.0, [1.0], prepare_simulation, mock_rng_sequence)
    assert value == pytest.approx(2.0)


@pytest.mark.parametrize(
    "grid,values,law",
    [([1, 1], [0.2, 0.8], 2), ([0, 1], [0.2, 0.8], 3), ([1, 2], [0, 1], 4)],
)
def test_invalid_probability_table_domains(grid, values, law):
    with pytest.raises(ValueError, match="probability|probabilities|Logarithmic"):
        DistributionMixture([DataTable(grid, values, law)], [components()[0]])
