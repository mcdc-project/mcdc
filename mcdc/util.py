import numpy as np

from numba import njit


# ======================================================================================
# Binary search
# ======================================================================================


@njit
def binary_search(val, grid):
    """
    Binary search with full length of the given grid.
    See binary_search_with_length
    """
    return binary_search_with_length(val, grid, len(grid))


@njit
def binary_search_with_length(val, grid, length):
    """
    Binary search that returns the bin index of the value `val` given grid `grid`
    Only search up to `length`-th element

    Some special cases:
        val < min(grid)  --> -1
        val > max(grid)  --> size of bins
        val = a grid point --> bin location whose upper bound is val
                               (-1 if val = min(grid))
    """
    left = 0
    right = length - 1
    mid = -1
    while left <= right:
        mid = int((left + right) / 2)
        if grid[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return int(right)


# ======================================================================================
# Data evaluators
# ======================================================================================


@njit
def evaluate_from_table(x, x_array, y_array):
    # Get bin index
    idx = binary_search(x, x_array)

    # Extrapolate if x is outside the given data
    if idx == -1:
        idx = 0
    elif idx + 1 == len(x_array):
        idx -= 1

    # Linear interpolation
    x1 = x_array[idx]
    x2 = x_array[idx + 1]
    y1 = y_array[idx]
    y2 = y_array[idx + 1]
    return linear_interpolation(x, x1, x2, y1, y2)


# ======================================================================================
# Interpolations
# ======================================================================================


@njit
def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


# ======================================================================================
# Python mode only
# ======================================================================================


def cdf_from_pdf(offset, value, pdf):
    cdf = np.zeros_like(pdf)
    for i in range(len(offset)):
        start = offset[i]
        end = offset[i + 1] if i < len(offset) - 1 else len(pdf)
        for idx in range(start, end - 2):
            cdf[idx + 1] = (
                cdf[idx]
                + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
            )
        # Ensure it ends at one
        cdf[end - 1] = 1.0

    return cdf
