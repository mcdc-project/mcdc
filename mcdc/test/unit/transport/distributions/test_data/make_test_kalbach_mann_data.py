import numpy as np


def make_test_kalbach_mann_data():
    # Two incident-energy tables, each with 3 points.
    Ei = [1.0, 3.0, 5.0]
    L_i = [0.0, 3.0, 6.0]

    E_i_k = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    p_i_k = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    c_i_k = [0.0, 0.5, 1.0, 0.0, 0.2, 1.0]

    # Keep R = 0 and A = 1 for deterministic angular sampling.
    R = [0.0] * 6
    A = [1.0] * 6

    data = np.array(
        Ei + L_i + E_i_k + p_i_k + c_i_k + R + A,
        dtype=np.float64,
    )

    idx = 0
    grid_offset = idx
    idx += len(Ei)
    offset_offset = idx
    idx += len(L_i)
    energy_out_offset = idx
    idx += len(E_i_k)
    pdf_offset = idx
    idx += len(p_i_k)
    cdf_offset = idx
    idx += len(c_i_k)
    precompound_offset = idx
    idx += len(R)
    angular_slope_offset = idx

    kalbach = {
        "energy_offset": grid_offset,
        "energy_length": len(Ei),
        "offset_offset": offset_offset,
        "offset_length": len(L_i),
        "energy_out_offset": energy_out_offset,
        "energy_out_length": len(E_i_k),
        "pdf_offset": pdf_offset,
        "pdf_length": len(p_i_k),
        "cdf_offset": cdf_offset,
        "cdf_length": len(c_i_k),
        "precompound_factor_offset": precompound_offset,
        "precompound_factor_length": len(R),
        "angular_slope_offset": angular_slope_offset,
        "angular_slope_length": len(A),
    }
    return kalbach, data
