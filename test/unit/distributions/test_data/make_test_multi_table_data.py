import numpy as np


def make_test_multi_table_data():
    # Two incident-energy tables, each with 3 points.
    Ei = [1.0, 3.0]
    L_i = [0.0, 3.0]

    E_i_k = [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    p_i_k = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    c_i_k = [0.0, 0.5, 1.0, 0.0, 0.6, 1.0]

    data = np.array(Ei + L_i + E_i_k + p_i_k + c_i_k, dtype=np.float64)

    idx = 0
    grid_offset = idx
    idx += len(Ei)
    offset_offset = idx
    idx += len(L_i)
    value_offset = idx
    idx += len(E_i_k)
    pdf_offset = idx
    idx += len(p_i_k)
    cdf_offset = idx

    multi_table = {
        "grid_offset": grid_offset,
        "grid_length": len(Ei),
        "offset_offset": offset_offset,
        "offset_length": len(L_i),
        "value_offset": value_offset,
        "value_length": len(E_i_k),
        "pdf_offset": pdf_offset,
        "pdf_length": len(p_i_k),
        "cdf_offset": cdf_offset,
        "cdf_length": len(c_i_k),
    }
    return multi_table, data
