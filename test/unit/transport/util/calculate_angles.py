import numpy as np
import pytest

from mcdc.transport.util import calculate_angles
from mcdc.numba_types import particle_data


@pytest.mark.parametrize(
    "ux, uy, uz, expected_mu, expected_phi",
    [
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, np.pi),
        (0.0, 1.0, 0.0, 0.0, np.pi / 2.0),
        (0.0, -1.0, 0.0, 0.0, -np.pi / 2.0),
        (0.0, 0.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, -1.0, -1.0, 0.0),
    ],
)
def test_calculate_angles(ux, uy, uz, expected_mu, expected_phi):
    particle_container = np.zeros(1, particle_data)
    particle = particle_container[0]

    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz
    polar_reference = np.array([0.0, 0.0, 1.0])

    mu, phi = calculate_angles(particle_container, polar_reference)

    assert np.isclose(mu, expected_mu)
    assert np.isclose(phi, expected_phi)
