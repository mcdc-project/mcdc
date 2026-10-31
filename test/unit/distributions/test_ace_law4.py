import h5py
import numpy as np

import mcdc.transport.distribution as transport_distribution
from mcdc.constant import INTERPOLATION_HISTOGRAM, INTERPOLATION_LINEAR
from mcdc.object_.distribution import DistributionMultiTable
from mcdc.object_.proton_reaction import set_energy_distribution


def make_law4_distribution():
    return DistributionMultiTable(
        grid=[1.0],
        value=[[0.0, 2.0]],
        pdf=[[0.25, 0.0]],
        discrete_value=[[8.0, 5.0]],
        pmf=[[0.25, 0.25]],
        pdf_interpolations=INTERPOLATION_HISTOGRAM,
    )


def test_ace_law4_sampler_samples_discrete_lines(mock_rng_sequence, prepare_simulation):
    distribution = make_law4_distribution()
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    base = simulation["distributions"][distribution.ID]

    line_1 = transport_distribution.sample_distribution(
        1.0, base, mock_rng_sequence(0.1), simulation, data
    )
    line_2 = transport_distribution.sample_distribution(
        1.0, base, mock_rng_sequence(0.3), simulation, data
    )

    assert line_1 == 8.0
    assert line_2 == 5.0


def test_ace_law4_sampler_samples_continuum(mock_rng_sequence, prepare_simulation):
    distribution = make_law4_distribution()
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    base = simulation["distributions"][distribution.ID]

    sample = transport_distribution.sample_distribution(
        1.0, base, mock_rng_sequence(0.75), simulation, data
    )

    np.testing.assert_allclose(sample, 1.0, rtol=0.0, atol=1e-12)


def test_ace_law4_sampler_scales_continuum_but_not_discrete_lines(
    mock_rng_sequence, prepare_simulation
):
    distribution = DistributionMultiTable(
        grid=[1.0, 3.0],
        value=[[1.0, 3.0], [2.0, 6.0]],
        pdf=[[0.4, 0.4], [0.2, 0.2]],
        discrete_value=[[9.0], [9.0]],
        pmf=[[0.2], [0.2]],
        pdf_interpolations=INTERPOLATION_HISTOGRAM,
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    base = simulation["distributions"][distribution.ID]

    line = transport_distribution.sample_distribution_with_scale(
        2.0, base, mock_rng_sequence(0.1), simulation, data
    )
    continuum = transport_distribution.sample_distribution_with_scale(
        2.0, base, mock_rng_sequence(0.6, 0.9), simulation, data
    )

    assert line == 9.0
    np.testing.assert_allclose(continuum, 3.0, rtol=0.0, atol=1e-12)


def test_set_energy_distribution_loads_ace_law4_data(tmp_path):
    path = tmp_path / "law4.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("spectrum")
        group.attrs["type"] = "tabulated"
        group.create_dataset("energy", data=[1.0])
        group.create_dataset("offset", data=[0])
        group.create_dataset("value", data=[8.0, 5.0, 0.0, 0.002])
        group.create_dataset("pdf", data=[0.25, 0.25, 250.0, 0.0])
        group.create_dataset("cdf", data=[0.25, 0.5, 0.5, 1.0])
        group.create_dataset("n_discrete", data=[2])
        group.create_dataset("interpolation", data=[1])

        distribution = set_energy_distribution(group)

    assert isinstance(distribution, DistributionMultiTable)
    np.testing.assert_allclose(distribution.grid, [1e6])
    np.testing.assert_allclose(distribution.value, [0.0, 2000.0])
    np.testing.assert_allclose(distribution.discrete_value, [8e6, 5e6])
    np.testing.assert_allclose(distribution.pmf, [0.25, 0.25])
    np.testing.assert_allclose(distribution.pdf, [250.0 / 1e6, 0.0])


def test_proton_law4_pure_discrete_tables(
    tmp_path, prepare_simulation, mock_rng_sequence
):
    with h5py.File(tmp_path / "lines.h5", "w") as group:
        group.attrs["type"] = "tabulated"
        group["energy"], group["offset"] = [1.0, 3.0], [0, 2]
        group["value"], group["pdf"] = [8.0, 5.0, 10.0, 7.0], [0.25, 0.75] * 2
        group["n_discrete"], group["interpolation"] = [2, 2], [1, 1]
        distribution = set_energy_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    base = simulation["distributions"][distribution.ID]
    assert distribution.value.size == 0
    np.testing.assert_allclose(distribution.pmf, [0.25, 0.75] * 2)
    line = transport_distribution.sample_distribution_with_scale(
        2e6, base, mock_rng_sequence(0.1), simulation, data
    )
    assert line == 9e6


def test_proton_law4_retains_per_table_interpolation(
    tmp_path, prepare_simulation, mock_rng_sequence
):
    with h5py.File(tmp_path / "histogram.h5", "w") as group:
        group.attrs["type"] = "tabulated"
        group["energy"], group["offset"] = [1.0, 3.0], [0, 2]
        group["value"], group["pdf"] = [0.0, 2.0, 0.0, 2.0], [0.5, 0.0, 0.0, 1.0]
        group["n_discrete"], group["interpolation"] = [0, 0], [1, 2]
        distribution = set_energy_distribution(group)
    container, data = prepare_simulation(objects=[distribution])
    simulation = container[0]
    base = simulation["distributions"][distribution.ID]
    np.testing.assert_array_equal(
        distribution.pdf_interpolations, [INTERPOLATION_HISTOGRAM, INTERPOLATION_LINEAR]
    )
    histogram = transport_distribution.sample_distribution(
        1e6, base, mock_rng_sequence(0.25), simulation, data
    )
    linear = transport_distribution.sample_distribution(
        3e6, base, mock_rng_sequence(0.25), simulation, data
    )
    assert histogram == 0.5e6
    assert linear == 1e6
