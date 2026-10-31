import pytest

from mcdc.object_.distribution import DistributionMultiTable
from mcdc.transport.physics.electron.native import sample_photon_energy


@pytest.mark.parametrize(
    "energy, expected",
    [
        (50.0, 10.0),
        (100.0, 20.0),
        (1000.0, 300.0),
        (10000.0, 4000.0),
        (20000.0, 8000.0),
    ],
)
def test_photon_energy_interpolates_energy_fractions(
    prepare_simulation, mock_rng_sequence, energy, expected
):
    """One quantile gives energy fractions interpolated in log incident energy."""
    spectrum = DistributionMultiTable(
        grid=[100.0, 10000.0],
        value=[[0.0, 20.0, 100.0], [0.0, 4000.0, 10000.0]],
        cdf=[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
    )
    container, data = prepare_simulation(objects=[spectrum])
    simulation = container[0]
    record = simulation["distributions"][spectrum.ID]
    state = mock_rng_sequence(0.5)

    sampled = sample_photon_energy(energy, record, state, simulation, data)

    assert sampled == pytest.approx(expected)
    assert state[0]["idx"] == 1


@pytest.mark.parametrize("xi", [0.0, 1.0])
def test_photon_energy_preserves_cdf_endpoints(
    prepare_simulation, mock_rng_sequence, xi
):
    """The endpoint quantiles remain bounded by the electron's energy."""
    spectrum = DistributionMultiTable(
        grid=[100.0, 10000.0],
        value=[[0.0, 100.0], [0.0, 10000.0]],
        cdf=[[0.0, 1.0], [0.0, 1.0]],
    )
    container, data = prepare_simulation(objects=[spectrum])
    simulation = container[0]
    record = simulation["distributions"][spectrum.ID]
    state = mock_rng_sequence(xi)

    assert sample_photon_energy(1000.0, record, state, simulation, data) == (
        xi * 1000.0
    )
