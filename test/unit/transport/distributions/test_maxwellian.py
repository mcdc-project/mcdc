import math
import numpy as np

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist
from mcdc.constant import PI

from .test_data import make_test_table_data_constant


def test_maxwellian_sample(mock_rng_sequence, make_distribution_record):
    # MCNP Theory & User Manual §2.4.3.5.4.6 (Law 7: Simple Maxwell Fission Spectrum)
    table_dict, data = make_test_table_data_constant(1.0)
    table = make_distribution_record(type_.table_data, table_dict)
    maxwellian = make_distribution_record(
        type_.maxwellian_distribution,
        {"nuclear_temperature_ID": 0, "restriction_energy": 0.0},
    )
    data = np.asarray(data, dtype=np.float64)

    simulation_dtype = np.dtype([("table_data", type_.table_data, (1,))])
    simulation_container = np.zeros(1, dtype=simulation_dtype)
    simulation = simulation_container[0]
    simulation["table_data"][0] = table

    xi1, xi2, xi3 = 0.9, 0.9, 0.0
    mock_rng = mock_rng_sequence(xi1, xi2, xi3)

    sampled_E = dist.sample_maxwellian(2.0, mock_rng, maxwellian, simulation, data)
    # MCNP Eq. (2.73):
    #   E_out = -T(E_in) * [xi_1^2 / (xi_1^2 + xi_2^2) * ln(xi_3) + ln(xi_4)]
    # MCDC uses the equivalent polar-form reduction
    #   xi_1^2 / (xi_1^2 + xi_2^2) = cos(theta)^2
    # with theta = (pi / 2) * xi3 and T(E_in) = 1 for this test table.
    expected_E = -(math.log(xi1) + math.log(xi2) * math.cos(0.5 * PI * xi3) ** 2)

    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
