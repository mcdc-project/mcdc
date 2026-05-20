import math

import mcdc.numba_types as type_
import mcdc.transport.distribution as dist


def test_level_scattering_sample(make_distribution_record):
    # MCNP Theory & User Manual §2.4.3.5.4.3 (Law 3: Inelastic Scattering from Nuclear Levels)
    # The implementation stores the Law 3 relation in the linear form
    #   E_out = C2 * (E_in - C1).
    # For this test, E_in = 5, C1 = 1, and C2 = 0.5, so
    #   E_out = 0.5 * (5 - 1) = 2.
    level = make_distribution_record(
        type_.level_scattering_distribution, {"C1": 1.0, "C2": 0.5}
    )
    sampled_E = dist.sample_level_scattering(5.0, level)
    expected_E = 2.0
    assert math.isclose(sampled_E, expected_E, rel_tol=0.0, abs_tol=1e-12)
