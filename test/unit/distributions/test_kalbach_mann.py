import math
import numpy as np

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist

from .test_data import make_test_kalbach_mann_data


def test_kalbach_mann_sample(mock_rng_sequence, make_distribution_record):
    # MCNP Theory & User Manual §2.4.3.5.4.11 (Law 44: Kalbach-87 Correlated Energy-angle Scattering)
    kalbach_dict, data = make_test_kalbach_mann_data()
    kalbach = make_distribution_record(type_.kalbach_mann_distribution, kalbach_dict)
    data = np.asarray(data, dtype=np.float64)

    # For E_in = 2.0 on the grid [1, 3], the interpolation fraction is r = 0.5.
    # xi_1 = 0.3 < r, so the second energy table is selected.
    xi1, xi2, xi3, xi4 = 0.3, 0.1, 0.7, 0.5
    mock_rng = mock_rng_sequence(xi1, xi2, xi3, xi4)
    sampled_E, sampled_mu = dist.sample_kalbach_mann(2.0, mock_rng, kalbach, data)

    # Law 44 uses the Law 4 energy construction, so with xi_2 = 0.1 in the first bin
    # of the selected table:
    E_min, E_max = 1.5, 4.5
    E_hat = 2.0 + (xi2 - 0.0) / 0.2
    # As in the Law 4 tests, the constant bin PDF means Eq. (2.66) collapses to the
    # simpler Eq. (2.65) interpolation for the sampled E'.
    expected_E = E_min + (E_hat - 2.0) / (6.0 - 2.0) * (E_max - E_min)

    # Eq. (2.90) and Eq. (2.91): with constant test data, the interpolation is trivial,
    # so A = 1 and R = 0 at every point.
    # Since xi_3 = 0.7 > R, Eq. (2.93) and Eq. (2.94) apply:
    #   T = (2 * xi_4 - 1) * sinh(A) = 0
    #   mu = ln(T + sqrt(T^2 + 1)) / A = 0
    expected_mu = 0.0

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sampled_mu, expected_mu, rtol=0.0, atol=1e-12)
