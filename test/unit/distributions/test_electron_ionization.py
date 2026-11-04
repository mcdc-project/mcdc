import math

import pytest

from mcdc.object_.distribution import DistributionMultiTable
from mcdc.transport.physics.electron.native import sample_delta_energy


def _prepare_distribution(prepare_simulation, **kwargs):
    distribution = DistributionMultiTable(**kwargs)
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    record = simulation["distributions"][distribution.ID]
    return record, simulation, data


@pytest.fixture
def knock_on_distribution(prepare_simulation):
    # Two CDF tables a decade of incident energy apart
    return _prepare_distribution(
        prepare_simulation,
        grid=[100.0, 10_000.0],
        value=[[1.0, 4.0, 16.0], [9.0, 36.0, 576.0]],
        cdf=[[0.0, 0.25, 1.0], [0.0, 0.25, 1.0]],
    )


@pytest.mark.parametrize(
    "xi, expected",
    [
        (0.0, 3.0),
        (0.125, 7.5),
        (0.25, 12.0),
        (0.5, math.sqrt(8.0 * 144.0)),
        (1.0, 96.0),
    ],
)
def test_delta_energy_shared_random_number(
    knock_on_distribution, mock_rng_sequence, xi, expected
):
    record, simulation, data = knock_on_distribution
    mock_rng = mock_rng_sequence(xi)

    # E = 1000 is the logarithmic midpoint of [100, 10000]. One xi inverts both
    # tables log-log (linearly on the segment starting at CDF = 0), and the two
    # energies combine into their geometric mean. At xi = 0.5, the inversions
    # give 8 and 144; at xi = 0.125, the zero-CDF segments give 2.5 and 22.5.
    T_delta = sample_delta_energy(1000.0, 10.0, record, mock_rng, simulation, data)

    assert T_delta == pytest.approx(expected, rel=1e-13)
    assert mock_rng[0]["idx"] == 1


@pytest.mark.parametrize(
    "incident_energy, expected", [(100.0, 8.0), (10_000.0, 144.0), (20_000.0, 144.0)]
)
def test_delta_energy_grid_ends(
    knock_on_distribution, mock_rng_sequence, incident_energy, expected
):
    # The first and last grid points use their own table; energies above the
    # grid keep the last table.
    record, simulation, data = knock_on_distribution
    mock_rng = mock_rng_sequence(0.5)

    T_delta = sample_delta_energy(
        incident_energy, 10.0, record, mock_rng, simulation, data
    )

    assert T_delta == pytest.approx(expected, rel=1e-13)
    assert mock_rng[0]["idx"] == 1


def test_delta_energy_interior_grid_point(prepare_simulation, mock_rng_sequence):
    # An incident energy on an interior grid point uses that table alone.
    record, simulation, data = _prepare_distribution(
        prepare_simulation,
        grid=[100.0, 1000.0, 10_000.0],
        value=[[1.0, 4.0, 16.0], [9.0, 36.0, 576.0], [1.0, 2.0, 4.0]],
        cdf=[[0.0, 0.25, 1.0], [0.0, 0.25, 1.0], [0.0, 0.25, 1.0]],
    )
    mock_rng = mock_rng_sequence(0.5)

    T_delta = sample_delta_energy(1000.0, 10.0, record, mock_rng, simulation, data)

    assert T_delta == pytest.approx(144.0, rel=1e-13)


@pytest.mark.parametrize("incident_energy", [10.001, 55.0, 100.0])
def test_delta_energy_skips_tables_at_binding_energy(
    prepare_simulation, mock_rng_sequence, incident_energy
):
    # EPRDATA14 tabulates an artificial positive spectrum at E = B. Below the
    # first table above B, that table's sample is scaled to reach zero at E = B.
    record, simulation, data = _prepare_distribution(
        prepare_simulation,
        grid=[1.0, 10.0, 100.0],
        value=[[0.01, 0.1], [0.01, 0.1], [1.0, 4.0, 16.0]],
        cdf=[[0.0, 1.0], [0.0, 1.0], [0.0, 0.25, 1.0]],
    )
    mock_rng = mock_rng_sequence(0.5)

    T_delta = sample_delta_energy(
        incident_energy, 10.0, record, mock_rng, simulation, data
    )

    assert T_delta == pytest.approx(8.0 * (incident_energy - 10.0) / 90.0)
    assert 0.0 <= T_delta <= (incident_energy - 10.0) / 2.0


@pytest.mark.parametrize(
    "xi, expected",
    [
        (0.0, 1.0),
        (0.125, 3.0),
        (0.25, 4.0),
        (0.5, 16.0),
        (0.75, 24.0),
        (1.0, 64.0),
    ],
)
def test_delta_energy_cdf_plateaus_and_endpoints(
    prepare_simulation, mock_rng_sequence, xi, expected
):
    # xi at the first CDF point returns the first value; xi on a repeated CDF
    # value returns the zero-width segment's upper value, as FRENSIE does.
    record, simulation, data = _prepare_distribution(
        prepare_simulation,
        grid=[1000.0],
        value=[[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]],
        cdf=[[0.0, 0.0, 0.25, 0.5, 0.5, 1.0, 1.0]],
    )
    mock_rng = mock_rng_sequence(xi)

    T_delta = sample_delta_energy(1000.0, 10.0, record, mock_rng, simulation, data)

    assert T_delta == pytest.approx(expected, rel=1e-13)


def test_delta_energy_caps_at_knock_on_limit(prepare_simulation, mock_rng_sequence):
    # Rounded tabulated energies can slightly exceed (E - B) / 2.
    record, simulation, data = _prepare_distribution(
        prepare_simulation,
        grid=[100.0],
        value=[[1.0, 45.0000001]],
        cdf=[[0.0, 1.0]],
    )
    mock_rng = mock_rng_sequence(1.0)

    T_delta = sample_delta_energy(100.0, 10.0, record, mock_rng, simulation, data)

    assert T_delta == 45.0


def test_delta_energy_pdf_input(prepare_simulation, mock_rng_sequence):
    # PDF input keeps the piecewise-linear PDF inversion. Rising triangular
    # PDFs give value[0] + (value[-1] - value[0]) * sqrt(xi): 5 and 20.
    record, simulation, data = _prepare_distribution(
        prepare_simulation,
        grid=[100.0, 10_000.0],
        value=[[1.0, 9.0], [4.0, 36.0]],
        pdf=[[0.0, 1.0], [0.0, 1.0]],
    )
    mock_rng = mock_rng_sequence(0.25)

    T_delta = sample_delta_energy(1000.0, 10.0, record, mock_rng, simulation, data)

    assert T_delta == pytest.approx(10.0, rel=1e-13)
    assert mock_rng[0]["idx"] == 1
