import math

import mcdc.transport.distribution as dist
from mcdc.constant import DISTRIBUTION_N_BODY

from .test_data import make_test_tabulated_data


def test_nbody_sample_correlated(rng_sequence, rng_state):
    # MCNP Theory & User Manual §2.4.3.5.4.13 (Law 66: N-body Phase Space Distribution)
    table, data = make_test_tabulated_data([2.0, 4.0, 6.0], [0.0, 0.4, 1.0])
    mcdc = {"nbody_distributions": [table]}
    distribution = {"child_type": DISTRIBUTION_N_BODY, "child_ID": 0}

    # First value samples energy, second value samples isotropic cosine.
    rng_sequence([0.2, 0.75])

    sampled_E, sampled_mu = dist.sample_correlated_distribution(
        2.0,
        distribution,
        rng_state,
        mcdc,
        data,
    )

    # The current implementation samples energy from the tabulated distribution and
    # samples the cosine isotropically. This test is therefore checking the current
    # reduced implementation, not reconstructing the full Law 66 rejection sampler
    # from Eq. (2.103) through Eq. (2.106).
    # For the tabulated-energy part, xi = 0.2 gives linear interpolation in the first
    # bin. For the angular part, MCNP Eq. (2.107) gives mu = 2 * xi_10 - 1 for
    # isotropic center-of-mass sampling.
    expected_E = 2.0 + (0.2 - 0.0) * (4.0 - 2.0) / (0.4 - 0.0)
    expected_mu = 2.0 * 0.75 - 1.0

    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(sampled_mu, expected_mu, rel_tol=0.0, abs_tol=1e-12)
