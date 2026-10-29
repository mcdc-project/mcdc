"""LAW 4 interpolates corresponding lines separately from its continuous spectra."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from mcdc.object_.distribution import DistributionMultiTable
from mcdc.object_.hdf5_reader import read_distribution, read_pdf_collection
from mcdc.transport.distribution import _sample_multi_table

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from data_library_generator.common.hdf5_writer import multi_table


def draw(
    distribution, energy, values, prepare_simulation, mock_rng_sequence, scale=True
):
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    assert simulation["N_table_distribution"] == 0
    record = simulation["multi_table_distributions"][distribution.sub_ID]
    state = mock_rng_sequence(*values)
    result = _sample_multi_table(energy, state, record, simulation, data, scale)
    assert np.isfinite(result)
    return result, state[0]["idx"]


@pytest.mark.parametrize("xi,expected", [(0, 5), (0.399, 5), (0.4, 15), (1, 15)])
def test_corresponding_line_probabilities_and_energies(
    xi, expected, prepare_simulation, mock_rng_sequence
):
    model = DistributionMultiTable(
        [1, 3], discrete_value=[[2, 12], [8, 18]], pmf=[[0.2, 0.8], [0.6, 0.4]]
    )
    actual, draws = draw(model, 2, [xi], prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(expected)
    assert draws == 1


@pytest.mark.parametrize("xi,expected", [(0, 5), (0.5, 15), (1, 15)])
def test_zero_mass_lines_preserve_correspondence(
    xi, expected, prepare_simulation, mock_rng_sequence
):
    model = DistributionMultiTable(
        [1, 3],
        discrete_value=[[2, 12, 100], [8, 18, 100]],
        pmf=[[0, 1, 0], [1, 0, 0]],
    )
    actual, _ = draw(model, 2, [xi], prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(expected)


def mixed_model():
    # Continuous probabilities .8 and .4 imply upper-table weight 1/3 at E=2.
    return DistributionMultiTable(
        [1, 3],
        value=[[0, 10], [100, 120]],
        pdf=[[0.08, 0.08], [0, 0.04]],
        discrete_value=[[2], [8]],
        pmf=[[0.2], [0.6]],
    )


@pytest.mark.parametrize(
    "values,expected,draws",
    [([0.3], 5, 1), ([0.7, 0.4], 57.5, 2), ([0.7, 0.2], 50 + 15 / np.sqrt(2), 2)],
)
def test_mixed_sampling_conditions_continuous_selection_and_scales_only_pdf(
    values, expected, draws, prepare_simulation, mock_rng_sequence
):
    actual, used = draw(mixed_model(), 2, values, prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(expected)
    assert used == draws


@pytest.mark.parametrize("energy,expected", [(0, 2), (1, 2), (3, 8), (4, 8)])
def test_lines_at_grid_endpoints_and_outside_grid(
    energy, expected, prepare_simulation, mock_rng_sequence
):
    actual, draws = draw(
        mixed_model(), energy, [0.1], prepare_simulation, mock_rng_sequence
    )
    assert actual == expected
    assert draws == 1


def test_histogram_does_not_interpolate_lines_or_require_matching_counts(
    prepare_simulation, mock_rng_sequence
):
    model = DistributionMultiTable(
        [1, 3], discrete_value=[[2], [8, 18]], pmf=[[1], [0.5, 0.5]], interpolations=1
    )
    actual, _ = draw(model, 2, [0.75], prepare_simulation, mock_rng_sequence)
    assert actual == 2


def test_continuum_absent_at_one_endpoint(prepare_simulation, mock_rng_sequence):
    model = DistributionMultiTable(
        [1, 3],
        value=[None, [100, 120]],
        pdf=[None, [0.04, 0.04]],
        discrete_value=[[2], [8]],
        pmf=[[1], [0.2]],
    )
    actual, _ = draw(model, 2, [0.8, 0.9], prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(110)


def test_joint_normalization_and_mixed_cdf_input():
    model = DistributionMultiTable(
        [1, 3],
        value=[[0, 10], [100, 120]],
        cdf=[[0, 8], [0, 4]],
        discrete_value=[[2], [8]],
        pmf=[[2], [6]],
    )
    np.testing.assert_allclose(model.continuous_probability, [0.8, 0.4])
    np.testing.assert_allclose(model.pmf, [0.2, 0.6])
    np.testing.assert_allclose(model.cdf, [0, 0.8, 0, 0.4])


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("unit", [None, "eV"])
def test_packed_mixed_hdf5_round_trip(
    tmp_path, pure, unit, prepare_simulation, mock_rng_sequence
):
    with h5py.File(tmp_path / "mixed.h5", "w") as file:
        multi_table(
            file,
            [1, 3],
            None if pure else [None, ([100, 120], [0.04, 0.04], None, None)],
            discrete_value=[[2], [8]],
            pmf=[[1], [1 if pure else 0.2]],
            unit=unit,
        )
        tables = file["tables"]
        assert tables["discrete-offsets"].shape == (3,)
        np.testing.assert_array_equal(tables["discrete-offsets"], [0, 1, 2])
        assert ("unit" in tables["discrete-value"].attrs) == (unit is not None)
        assert "unit" not in tables["pmf"].attrs
        assert "unit" not in tables["discrete-offsets"].attrs
        if pure:
            assert set(tables) == {"discrete-offsets", "discrete-value", "pmf"}
        else:
            np.testing.assert_array_equal(tables["offsets"], [0, 0, 2])
            np.testing.assert_array_equal(tables["interpolation/offsets"], [0, 0, 1])
        with pytest.raises(ValueError, match="continuous only"):
            read_pdf_collection(tables)
        model = read_distribution(file)
    actual, _ = draw(model, 2, [0.1], prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(5)


@pytest.mark.parametrize(
    "arguments",
    [
        {"discrete_value": [[2], [8, 18]], "pmf": [[1], [0.5, 0.5]]},
        {"discrete_value": [[2], [8]], "pmf": [[1], [-1]]},
        {"discrete_value": [[2], [np.nan]], "pmf": [[1], [1]]},
        {"discrete_value": [[2], [8]], "pmf": [[1], [0]]},
        {"discrete_value": [2, 8], "pmf": [1, 1], "discrete_offsets": [0, 1, 3]},
        {"discrete_value": [2, 8], "pmf": [1, 1], "discrete_offsets": [0.0, 1.0, 2.0]},
    ],
)
def test_invalid_discrete_packed_objects(arguments):
    with pytest.raises(ValueError):
        DistributionMultiTable([1, 3], **arguments)


@pytest.mark.parametrize("masses", [[[1], [0.5, 0.5]], [[1], [-1]], [[1], [1.1]]])
def test_invalid_writer_is_atomic(tmp_path, masses):
    with h5py.File(tmp_path / "invalid.h5", "w") as file:
        with pytest.raises(ValueError):
            multi_table(
                file,
                [1, 3],
                discrete_value=[[2], [8, 18] if len(masses[1]) == 2 else [8]],
                pmf=masses,
            )
        assert len(file) == len(file.attrs) == 0


@pytest.mark.parametrize("density_law", [1, 2, 3, 4, 5])
def test_mixed_multi_pdf_uses_shared_density_inversion(
    density_law, prepare_simulation, mock_rng_sequence
):
    # p(x)=c*x on [1,3]; every interpolation law has its own analytic integral.
    from mcdc.object_.distribution import DistributionTable
    from mcdc.transport.distribution import sample_table

    standalone = DistributionTable(
        [1, 3], [0.1, 0.3], discrete_value=[-2], pmf=[0.6], interpolations=density_law
    )
    model = DistributionMultiTable(
        [1, 3],
        value=[[1, 3]] * 2,
        pdf=[[0.1, 0.3]] * 2,
        discrete_value=[[-2], [-2]],
        pmf=[[0.6], [0.6]],
        pdf_interpolations=density_law,
    )
    container, data = prepare_simulation(objects=[standalone])
    expected = sample_table(
        container[0]["table_distributions"][standalone.sub_ID],
        mock_rng_sequence(0.8),
        container[0],
        data,
    )
    actual, _ = draw(model, 2, [0.8, 0.2], prepare_simulation, mock_rng_sequence)
    assert actual == pytest.approx(expected)
