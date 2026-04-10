import numpy as np

from mcdc.constant import INTERPOLATION_LINEAR


def make_test_table_data_constant(value=1.0):
    x = [0.0, 10.0]
    y = [value, value]
    data = np.array(x + y, dtype=np.float64)
    table = {
        "x_offset": 0,
        "x_length": len(x),
        "y_offset": len(x),
        "y_length": len(y),
        "interpolation": INTERPOLATION_LINEAR,
    }
    return table, data
