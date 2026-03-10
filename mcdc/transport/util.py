import math
import numpy as np

from numba import njit
from typing import Sequence


@njit
def find_bin(
    value: float, grid: Sequence[float], epsilon: float = 0.0, go_lower: bool = True
) -> int:
    """
    Return the bin index i for which grid[i] <= value < grid[i+1], with optional
    epsilon tolerance and tie-breaking toward the lower/upper bin.

    Parameters
    ----------
    value : float
        Query point.
    grid : Sequence[float]
        Monotonically increasing bin edges of length N_grid = N_bin + 1.
    epsilon : float, optional (default: 0.0)
        Tolerance to treat values as being exactly on a grid edge if
        |value - grid[k]| <= epsilon.
    go_lower : bool, optional (default: True)
        Tie-breaking rule when value is at/within epsilon of a grid edge:
          - True  -> tie to the lower/left bin
          - False -> tie to the upper/right bin

    Edge behavior (with epsilon)
    ----------------------------
    - Interior edges (grid[k], 0<k<N_grid-1):
        * go_lower=True  -> bin k-1
        * go_lower=False -> bin k
    - First edge (grid[0]):
        * If inside or exactly at grid[0] within epsilon:
            - go_lower=True  -> -1 (treat as outside left)
            - go_lower=False -> 0  (first bin)
    - Last edge (grid[-1]):
        * If exactly at/within epsilon:
            - go_lower=True  -> last bin (N_bin-1)
            - go_lower=False -> -1 (outside right)
    - Beyond first/last edge by more than epsilon: return -1.

    Notes
    -----
    - With epsilon=0 and go_lower=True, this reduces to the standard
      left-closed/right-open binning (grid[i] <= value < grid[i+1]).
    - Scalar-only implementation (no NumPy required).
    """
    n = len(grid)

    # Fast reject beyond tolerance band
    if value < grid[0] - epsilon or value > grid[-1] + epsilon:
        return -1

    # Base binary search (strict left-closed / right-open, no epsilon)
    low, high = 0, n - 1  # search over edge indices
    if value < grid[0] or value >= grid[-1]:
        base = -1
    else:
        while high - low > 1:
            mid = (low + high) // 2
            if value < grid[mid]:
                high = mid
            else:
                low = mid
        base = low  # provisional bin: [grid[low], grid[low+1])

    # Tie-breaking near edges (epsilon band)
    if base == -1:
        # Near first edge?
        if abs(value - grid[0]) <= epsilon:
            return -1 if go_lower else 0
        # Near last edge?
        if abs(value - grid[-1]) <= epsilon:
            return (n - 2) if go_lower else -1
        return -1

    idx = base

    # Check left edge of this bin
    if abs(value - grid[idx]) <= epsilon:
        if idx == 0:
            return -1 if go_lower else 0
        return (idx - 1) if go_lower else idx

    # Check right edge of this bin
    right_edge = grid[idx + 1]
    if abs(value - right_edge) <= epsilon:
        if idx + 1 == n - 1:  # last grid point
            return (n - 2) if go_lower else -1
        return idx if go_lower else (idx + 1)

    # Strict interior
    return idx


@njit
def atomic_add(array, idx, value):
    array[idx] += value


@njit
def local_array(shape, dtype):
    return np.zeros(shape, dtype=dtype)


# ======================================================================================
# Interpolation
# ======================================================================================


@njit
def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


@njit
def log_interpolation(x, x1, x2, y1, y2):
    # Convert to logs
    lx1, lx2 = math.log(x1), math.log(x2)
    ly1, ly2 = math.log(y1), math.log(y2)

    # Slope in log–log space
    m = (ly2 - ly1) / (lx2 - lx1)

    # Interpolate log(y)
    ly = ly1 + m * (math.log(x) - lx1)

    return math.exp(ly)
