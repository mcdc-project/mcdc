import math
import numpy as np

import mcdc.transport.distribution as dist

from .test_data import make_test_tabulated_energy_angle_data


def test_tabulated_energy_angle_sample(rng_sequence, rng_state):
    # MCNP Theory & User Manual §2.4.3.5.4.12 (Law 61: Correlated Energy-angle Scattering)
    table, data = make_test_tabulated_energy_angle_data()

    xi1, xi2, xi3 = 0.3, 0.1, 0.25
    rng_sequence([xi1, xi2, xi3])

    sampled_E, sampled_mu = dist.sample_tabulated_energy_angle(
        2.0, rng_state, table, data
    )

    # Law 61 uses the Law 4 energy construction first.
    # For E_in = 2.0 on the grid [1, 3], Eq. (2.62) gives r = 0.5.
    # xi_1 = 0.3 < r, so the second outgoing-energy table is selected.
    # The test data again use a constant bin PDF, so the sampled E' is the
    # Eq. (2.65) form rather than the more general Eq. (2.66) expression.
    E_min, E_max = 1.5, 4.5
    E_hat = 2.0 + (xi2 - 0.0) / 0.2
    # E_min and E_max are the scaled lower and upper bounds from the two incident
    # energy tables, i.e. the Law 4 quantities from Eq. (2.67) through Eq. (2.69).
    expected_E = E_min + (E_hat - 2.0) / (6.0 - 2.0) * (E_max - E_min)

    # For the angular table, Law 61 says the linear-interpolation case chooses the
    # tabular angular distribution whose CDF point is closest to the sampled xi_2.
    # Here xi_2 = 0.1 is tied between the first two CDF points in the selected energy
    # bin, and the implementation keeps the lower index, so the first cosine table is
    # used. Sampling that table gives mu = -1 + xi_3 / 0.5.
    expected_mu = -1.0 + (xi3 - 0.0) / 0.5

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sampled_mu, expected_mu, rtol=0.0, atol=1e-12)
