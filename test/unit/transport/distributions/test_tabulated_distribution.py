import math

import mcdc.transport.distribution as dist

from .test_data import make_test_tabulated_data


def test_tabulated_distribution_sample(rng_sequence, rng_state):
    # MCNP Theory & User Manual §2.4.3.5.4.4 (Law 4: Tabular Distribution)
    table, data = make_test_tabulated_data([1.0, 3.0, 7.0], [0.0, 0.4, 1.0])
    rng_sequence([0.2])

    sampled_E = dist.sample_tabulated(table, rng_state, data)
    # This is the single-table inverse-CDF interpolation used by the tabulated sampler:
    # xi = 0.2 lies in the first bin, so linear interpolation between
    # (c_0, E_0) = (0.0, 1.0) and (c_1, E_1) = (0.4, 3.0) gives the expected value.
    expected_E = 1.0 + (0.2 - 0.0) * (3.0 - 1.0) / (0.4 - 0.0)

    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
