import math
import numpy as np

import mcdc.transport.distribution as dist
from mcdc.constant import PI

from .test_data import make_test_table_data_constant


def test_maxwellian_sample(rng_sequence, rng_state):
    # MCNP Theory & User Manual §2.4.3.5.4.6 (Law 7: Simple Maxwell Fission Spectrum)
    table, data = make_test_table_data_constant(1.0)
    mcdc = {"table_data": [table]}
    maxwellian = {"nuclear_temperature_ID": 0, "restriction_energy": 0.0}

    xi1, xi2, xi3 = 0.9, 0.9, 0.0
    rng_sequence([xi1, xi2, xi3])

    sampled_E = dist.sample_maxwellian(2.0, rng_state, maxwellian, mcdc, data)
    # MCNP Eq. (2.73):
    #   E_out = -T(E_in) * [xi_1^2 / (xi_1^2 + xi_2^2) * ln(xi_3) + ln(xi_4)]
    # MCDC uses the equivalent polar-form reduction
    #   xi_1^2 / (xi_1^2 + xi_2^2) = cos(theta)^2
    # with theta = (pi / 2) * xi3 and T(E_in) = 1 for this test table.
    expected_E = -(math.log(xi1) + math.log(xi2) * math.cos(0.5 * PI * xi3) ** 2)

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)
