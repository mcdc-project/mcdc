import math
import numpy as np

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist
from mcdc.constant import DISTRIBUTION_N_BODY

from .test_data import make_test_tabulated_data


def test_nbody_sample_correlated(mock_rng_sequence, make_distribution_record):
    # MCNP Theory & User Manual §2.4.3.5.4.13 (Law 66: N-body Phase Space Distribution)
    table_dict, data = make_test_tabulated_data([2.0, 4.0, 6.0], [0.0, 0.4, 1.0])
    nbody = make_distribution_record(type_.nbody_distribution, table_dict)
    distribution = make_distribution_record(
        type_.distribution, {"child_type": DISTRIBUTION_N_BODY, "child_ID": 0}
    )
    data = np.asarray(data, dtype=np.float64)

    # Numba compiles all correlated-branch field accesses, so this container needs
    # the three correlated distribution arrays even though this test uses N-body only.
    simulation_dtype = np.dtype(
        [
            ("kalbach_mann_distributions", type_.kalbach_mann_distribution, (1,)),
            (
                "tabulated_energy_angle_distributions",
                type_.tabulated_energy_angle_distribution,
                (1,),
            ),
            ("nbody_distributions", type_.nbody_distribution, (1,)),
        ]
    )
    simulation_container = np.zeros(1, dtype=simulation_dtype)
    simulation = simulation_container[0]
    simulation["nbody_distributions"][0] = nbody

    # First value samples energy, second value samples isotropic cosine.
    xi1, xi2 = 0.2, 0.75
    mock_rng = mock_rng_sequence(xi1, xi2)

    sampled_E, sampled_mu = dist.sample_correlated_distribution(
        2.0,
        distribution,
        mock_rng,
        simulation,
        data,
    )

    # The current implementation samples energy from the tabulated distribution and
    # samples the cosine isotropically. This test is therefore checking the current
    # reduced implementation, not reconstructing the full Law 66 rejection sampler
    # from Eq. (2.103) through Eq. (2.106).
    # For the tabulated-energy part, xi_1 = 0.2 gives linear interpolation in the first
    # bin. For the angular part, MCNP Eq. (2.107) gives mu = 2 * xi_10 - 1 for
    # isotropic center-of-mass sampling.
    expected_E = 2.0 + (xi1 - 0.0) * (4.0 - 2.0) / (0.4 - 0.0)
    expected_mu = 2.0 * xi2 - 1.0

    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(sampled_mu, expected_mu, rel_tol=0.0, abs_tol=1e-12)
