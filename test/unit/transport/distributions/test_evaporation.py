import math
import numpy as np

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist

from .test_data import make_test_table_data_constant


def test_evaporation_sample(mock_rng_sequence, make_distribution_record):
    # MCNP Theory & User Manual §2.4.3.5.4.7 (Law 9: Evaporation Spectrum)
    table_dict, data = make_test_table_data_constant(1.0)
    table = make_distribution_record(type_.table_data, table_dict)
    evaporation = make_distribution_record(
        type_.evaporation_distribution,
        {"nuclear_temperature_ID": 0, "restriction_energy": 0.0},
    )
    data = np.asarray(data, dtype=np.float64)

    simulation_dtype = np.dtype([("table_data", type_.table_data, (1,))])
    simulation_container = np.zeros(1, dtype=simulation_dtype)
    simulation = simulation_container[0]
    simulation["table_data"][0] = table

    xi1, xi2 = 0.1, 0.2
    mock_rng = mock_rng_sequence(xi1, xi2)

    sampled_E = dist.sample_evaporation(2.0, mock_rng, evaporation, simulation, data)
    # The constant temperature table gives T(E_in) = 1.0 for this test.
    # With U = 0, the MCNP notation E_in - U becomes 2.0, so
    #   w = (E_in - U) / T(E_in) = 2.0.
    w = 2.0
    # Eq. (2.75) introduces g = 1 - exp(-w) in the normalized truncated spectrum.
    g = 1.0 - math.exp(-w)
    # Eq. (2.76) then gives the sampled evaporation energy:
    #   E_out = -T(E_in) * ln[(1 - g * xi_1) (1 - g * xi_2)].
    # Since T(E_in) = 1.0 here, the prefactor is omitted in the simplified form below.
    expected_E = -math.log((1.0 - g * xi1) * (1.0 - g * xi2))

    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
