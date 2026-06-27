from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import (
    DATA_POLYNOMIAL,
    DATA_TABLE,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
    INTERPOLATION_SEMILOGX,
    INTERPOLATION_SEMILOGY,
    INTERPOLATION_LOG,
)
from mcdc.transport.util import (
    find_bin,
    histogram_interpolation,
    linear_interpolation,
    semilogx_interpolation,
    semilogy_interpolation,
    log_interpolation,
)


@njit
def evaluate_data(x, data_base, simulation, data):
    data_type = data_base["child_type"]
    ID = data_base["child_ID"]
    if data_type == DATA_TABLE:
        table = simulation["table_data"][ID]
        return evaluate_table(x, table, data)
    elif data_type == DATA_POLYNOMIAL:
        polynomial = simulation["polynomial_data"][ID]
        return evaluate_polynomial(x, polynomial, data)
    else:
        return 0.0


@njit
def evaluate_table(x, table, data):
    offset = table["x_offset"]
    length = table["x_length"]
    grid = data[offset : offset + length]
    # Above is equivalent to: grid = mcdc_get.table_data.x_all(table, data)

    idx = find_bin(x, grid)
    x1 = grid[idx]
    x2 = grid[idx + 1]
    y1 = mcdc_get.table_data.y(idx, table, data)
    y2 = mcdc_get.table_data.y(idx + 1, table, data)

    # Get interpolation law
    interpolation = get_table_interpolation_law(idx, table, data)

    # Perform interpolation
    if interpolation == INTERPOLATION_HISTOGRAM:
        return histogram_interpolation(x, x1, x2, y1, y2)
    elif interpolation == INTERPOLATION_LINEAR:
        return linear_interpolation(x, x1, x2, y1, y2)
    elif interpolation == INTERPOLATION_SEMILOGX:
        return semilogx_interpolation(x, x1, x2, y1, y2)
    elif interpolation == INTERPOLATION_SEMILOGY:
        return semilogy_interpolation(x, x1, x2, y1, y2)
    elif interpolation == INTERPOLATION_LOG:
        return log_interpolation(x, x1, x2, y1, y2)


@njit
def get_table_interpolation_law(idx, table, data) -> int:
    """Return the interpolation law for interval [idx, idx + 1]."""
    offset = table["interpolation_boundaries_offset"]
    length = table["interpolation_boundaries_length"]
    boundaries = data[offset : offset + length]
    # Above is equivalent to: boundaries = mcdc_get.table_data.interpolation_boundaries_all(table, data)

    offset = table["interpolations_offset"]
    length = table["interpolations_length"]
    interpolations = data[offset : offset + length]
    # Above is equivalent to: interpolations = mcdc_get.table_data.interpolations_all(table, data)

    # Boundaries are exclusive upper point indices.
    upper_point = idx + 1

    I = len(boundaries)
    for i in range(I):
        if upper_point < boundaries[i]:
            return interpolations[i]
    return interpolations[-1]


@njit
def evaluate_polynomial(x, polynomial, data):
    offset = polynomial["coefficients_offset"]
    length = polynomial["coefficients_length"]
    coeffs = data[offset : offset + length]
    # Above is equivalent to: coeffs = mcdc_get.polynomial_data.coefficients_all(polynomial, data)

    total = 0.0
    for i in range(len(coeffs)):
        total += coeffs[i] * x**i
    return total
