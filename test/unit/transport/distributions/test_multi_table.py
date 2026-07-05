import math
import numpy as np

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist

from .test_data import make_test_multi_table_data


def test_multi_table_distribution_sample(mock_rng_sequence, make_distribution_record):
    """
    # MCNP Theory & User Manual §2.4.3.5.4.4 (Law 4: Tabular Distribution)
    multi_table_dict, data = make_test_multi_table_data()
    multi_table = make_distribution_record(
        type_.multi_table_distribution, multi_table_dict
    )
    data = np.asarray(data, dtype=np.float64)
    # For E_in = 2.0 on the grid [1, 3], Eq. (2.62) gives r = 0.5.
    # xi_1 = 0.3 < r, so Eq. (2.64) selects l = i + 1, i.e. the second table.
    xi1, xi2 = 0.3, 0.2
    mock_rng = mock_rng_sequence(xi1, xi2)

    sampled_E = dist._sample_multi_table(2.0, mock_rng, multi_table, data, scale=True)

    # In the selected table, xi_2 = 0.2 falls in the first continuous bin.
    # Eq. (2.65) gives E' = E_l,k + (xi_2 - c_l,k) / p_l,k = 100 + 0.2 / 0.01 = 120.
    # The test data use constant p within the bin, so the linear-linear form in
    # Eq. (2.66) reduces to the same result.
    E_prime = 100.0 + (xi2 - 0.0) / 0.01
    # Eq. (2.67) and Eq. (2.68) give the scaled bounds:
    #   E_1 = 10 + 0.5 * (100 - 10) = 55
    #   E_K = 30 + 0.5 * (300 - 30) = 165
    # Here E_l,1 = 100 and E_l,K = 300 because the selected table is the second one.
    # Eq. (2.69) then gives
    #   E_out = E_1 + (E' - E_l,1) * (E_K - E_1) / (E_l,K - E_l,1)
    expected_E = 55.0 + (E_prime - 100.0) * (165.0 - 55.0) / (300.0 - 100.0)

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)
    """
