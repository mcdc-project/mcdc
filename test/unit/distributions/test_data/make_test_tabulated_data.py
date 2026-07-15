import numpy as np


def make_test_tabulated_data(E_i_k, c_i_k):
    E_i_k = list(E_i_k)
    c_i_k = list(c_i_k)
    data = np.array(E_i_k + c_i_k, dtype=np.float64)
    table = {
        "value_offset": 0,
        "value_length": len(E_i_k),
        "cdf_offset": len(E_i_k),
        "cdf_length": len(c_i_k),
    }
    return table, data
