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
def evaluate_data(x, data_, simulation, data):
    return evaluate_data_with_clamp_option(x, data_, simulation, data, False)


@njit
def evaluate_data_clamped(x, data_, simulation, data):
    return evaluate_data_with_clamp_option(x, data_, simulation, data, True)


@njit
def evaluate_data_with_clamp_option(x, data_, simulation, data, clamp):
    """Evaluate data, optionally clamping tabulated data to its endpoint values."""
    data_type = data_["sub_type"]
    ID = data_["sub_ID"]
    if data_type == DATA_TABLE:
        table = simulation["table_data"][ID]
        return evaluate_table_with_clamp_option(x, table, data, clamp)
    elif data_type == DATA_POLYNOMIAL:
        polynomial = simulation["polynomial_data"][ID]
        return evaluate_polynomial(x, polynomial, data)
    else:
        return 0.0


@njit
def evaluate_table(x, table, data):
    return evaluate_table_with_clamp_option(x, table, data, False)


@njit
def evaluate_table_clamped(x, table, data):
    return evaluate_table_with_clamp_option(x, table, data, True)


@njit
def evaluate_table_with_clamp_option(x, table, data, clamp):
    """Evaluate a table with optional constant extrapolation outside its grid."""
    grid = mcdc_get.table_data.x_all(table, data)

    if len(grid) == 1:
        return mcdc_get.table_data.y(0, table, data)

    # Clamp beyond the grid; equality is evaluated to the right of any jump.
    if clamp:
        if x < grid[0]:
            return mcdc_get.table_data.y(0, table, data)
        if x >= grid[-1]:
            return mcdc_get.table_data.y(len(grid) - 1, table, data)

    idx = min(max(find_bin(x, grid), 0), len(grid) - 2)
    if x == grid[-1]:
        return mcdc_get.table_data.y(len(grid) - 1, table, data)
    while idx < len(grid) - 2 and grid[idx] == grid[idx + 1] and x >= grid[idx + 1]:
        idx += 1
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
    boundaries = mcdc_get.table_data.interpolation_boundaries_all(table, data)

    interpolations = mcdc_get.table_data.interpolations_all(table, data)

    # Boundaries are exclusive upper point indices.
    upper_point = idx + 1

    I = len(boundaries)
    for i in range(I):
        if upper_point < boundaries[i]:
            return interpolations[i]
    return interpolations[-1]


@njit
def evaluate_polynomial(x, polynomial, data):
    coeffs = mcdc_get.polynomial_data.coefficients_all(polynomial, data)

    total = 0.0
    for i in range(len(coeffs)):
        total += coeffs[i] * x**i
    return total
