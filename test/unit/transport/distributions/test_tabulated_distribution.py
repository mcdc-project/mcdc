import math
import numpy as np

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist

from .test_data import make_test_tabulated_data


def test_tabulated_distribution_sample(mock_rng_sequence, make_distribution_record):
    # MCNP Theory & User Manual §2.4.3.5.4.4 (Law 4: Tabular Distribution)
    table_dict, data = make_test_tabulated_data([1.0, 3.0, 7.0], [0.0, 0.4, 1.0])
    table = make_distribution_record(type_.tabulated_distribution, table_dict)
    data = np.asarray(data, dtype=np.float64)
    xi1 = 0.2
    mock_rng = mock_rng_sequence(xi1)

    sampled_E = dist.sample_tabulated(table, mock_rng, data)
    # This is the single-table inverse-CDF interpolation used by the tabulated sampler:
    # xi_1 = 0.2 lies in the first bin, so linear interpolation between
    # (c_0, E_0) = (0.0, 1.0) and (c_1, E_1) = (0.4, 3.0) gives the expected value.
    expected_E = 1.0 + (xi1 - 0.0) * (3.0 - 1.0) / (0.4 - 0.0)

    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
