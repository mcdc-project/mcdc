import h5py
import numpy as np

import mcdc.transport.distribution as transport_distribution
from mcdc.object_.distribution import DistributionACELaw4
from mcdc.object_.proton_reaction import set_energy_distribution


def make_law4_distribution():
    return DistributionACELaw4(
        energy=[1.0],
        offset=[0],
        value=[8.0, 5.0, 0.0, 2.0],
        pdf=[0.25, 0.25, 0.25, 0.0],
        cdf=[0.25, 0.5, 0.5, 1.0],
        n_discrete=[2],
        interpolation=[1],
    )


def test_ace_law4_sampler_samples_discrete_lines(mock_rng_sequence, prepare_simulation):
    distribution = make_law4_distribution()
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    law4 = simulation["ace_law4_distributions"][distribution.sub_ID]

    line_1 = transport_distribution.sample_ace_law4(
        1.0, mock_rng_sequence(0.1), law4, data
    )
    line_2 = transport_distribution.sample_ace_law4(
        1.0, mock_rng_sequence(0.3), law4, data
    )

    assert line_1 == 8.0
    assert line_2 == 5.0


def test_ace_law4_sampler_samples_continuum(mock_rng_sequence, prepare_simulation):
    distribution = make_law4_distribution()
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    law4 = simulation["ace_law4_distributions"][distribution.sub_ID]

    sample = transport_distribution.sample_ace_law4(
        1.0, mock_rng_sequence(0.75), law4, data
    )

    np.testing.assert_allclose(sample, 1.0, rtol=0.0, atol=1e-12)


def test_ace_law4_sampler_scales_continuum_but_not_discrete_lines(
    mock_rng_sequence, prepare_simulation
):
    distribution = DistributionACELaw4(
        energy=[1.0, 3.0],
        offset=[0, 3],
        value=[9.0, 1.0, 3.0, 9.0, 2.0, 6.0],
        pdf=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        cdf=[0.2, 0.2, 1.0, 0.2, 0.2, 1.0],
        n_discrete=[1, 1],
        interpolation=[1, 1],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    law4 = simulation["ace_law4_distributions"][distribution.sub_ID]

    line = transport_distribution._sample_ace_law4(
        2.0, mock_rng_sequence(0.9, 0.1), law4, data, True
    )
    continuum = transport_distribution._sample_ace_law4(
        2.0, mock_rng_sequence(0.9, 0.4), law4, data, True
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
        group.create_dataset("pdf", data=[0.25, 0.25, 500.0, 0.0])
        group.create_dataset("cdf", data=[0.25, 0.5, 0.5, 1.0])
        group.create_dataset("n_discrete", data=[2])
        group.create_dataset("interpolation", data=[1])

        distribution = set_energy_distribution(group)

    assert isinstance(distribution, DistributionACELaw4)
    np.testing.assert_allclose(distribution.energy, [1e6])
    np.testing.assert_allclose(distribution.value, [8e6, 5e6, 0.0, 2000.0])
