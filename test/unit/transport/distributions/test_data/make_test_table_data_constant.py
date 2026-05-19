import numpy as np

from mcdc.constant import INTERPOLATION_LINEAR


def make_test_table_data_constant(value=1.0):
    x = [0.0, 10.0]
    y = [value, value]
    interpolations = [INTERPOLATION_LINEAR]
    interpolation_boundaries = [len(x)]
    data = np.array(x + y + interpolations + interpolation_boundaries, dtype=np.float64)
    table = {
        "x_offset": 0,
        "x_length": len(x),
        "y_offset": len(x),
        "y_length": len(y),
        "interpolations_offset": len(x) + len(y),
        "interpolations_length": len(interpolations),
        "interpolation_boundaries_offset": len(x) + len(y) + len(interpolations),
        "interpolation_boundaries_length": len(interpolation_boundaries),
    }
    return table, data
