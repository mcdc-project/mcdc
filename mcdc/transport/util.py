import math
import numpy as np

from numba import njit
from typing import Sequence


@njit
def find_bin_with_rules(value, grid, epsilon, go_lower):
    """
    Return the bin index i for which grid[i] <= value < grid[i+1], with optional
    epsilon tolerance and tie-breaking toward the lower/upper bin.

    Parameters
    ----------
    value : float
        Query point.
    grid : Sequence[float]
        Monotonically increasing bin edges of length N_grid = N_bin + 1.
    epsilon : float
        Tolerance to treat values as being exactly on a grid edge if
        |value - grid[k]| <= epsilon.
    go_lower : bool
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
def find_bin(value, grid):
    tolerance = 0.0
    go_lower = True
    return find_bin_with_rules(value, grid, tolerance, go_lower)


@njit
def find_bin_with_tolerance(value, grid, tolerance):
    go_lower = True
    return find_bin_with_rules(value, grid, tolerance, go_lower)


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


# ======================================================================================
# Angle conversion utilities
# ======================================================================================


@njit
def calculate_angles(particle_container, px, py, pz):
    """
    Calculate particle mu and azimuthal angle from given reference vector

    Parameters
    ----------
    particle_container : ndarray
      Container holding the particle.
    px : float
      X-component of reference vector
    py : float
      Y-component of reference vector
    pz : float
      Z-component of reference vector

    Returns
    -------
    mu : float
      Mu of particle relative to reference vector
    azimuthal : float
      Azimuthal angle of particle relative to reference vector
    """
    particle = particle_container[0]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    mu = ux*px + uy*py + uz*pz

    azimuthal = _calculate_azimuthal(ux, uy, uz, px, py, pz)

    return mu, azimuthal


def _calculate_azimuthal(ux, uy, uz, px, py, pz):
    """
    Calculates the azimuthal angle of a particle relative to a reference vector.
    This is done by finding two orthonormal basis vectors perpendicular to the reference vector.
    The first orthonormal vector, u1, is found using the graham-schmidt procedure.
    The second, u2, is found from the cross product of the reference vector and u1.

    Parameters
    ----------
    ux : float
      X-component of particle's direction
    uy : float
      Y-component of particle's direction
    uz : float
      Z-component of particle's direction
    px : float
      X-component of reference vector
    py : float
      Y-component of reference vector
    pz : float
      Z-component of reference vector

    Returns
    -------
    azimuthal : float
      Azimuthal angle of particle relative to reference vector
    """
    # get two orthonormal basis vectors u1, u2 perpendicular to p
    # u1 done via gram-schmidt, u2 done via cross product

    # choose arbitrary guess v1, check to make sure its not accidentally parallel to p
    if px < 0.9:
      v1x, v1y, v1z = 1.0, 0.0, 0.0
    else:
        v1x, v1y, v1z = 0.0, 1.0, 0.0

    # u1 = (v1 - proj(v1, p)) / ||u1||
    v1dotp = v1x*px + v1y*py + v1z*pz
    u1x = v1x - v1dotp * px
    u1y = v1y - v1dotp * py
    u1z = v1z - v1dotp * pz
    u1norm = math.sqrt(u1x*u1x + u1y*u1y + u1z*u1z)
    u1x /= u1norm
    u1y /= u1norm
    u1z /= u1norm

    # u2 = p x u1
    u2x = py*u1z - pz*u1y
    u2y = pz*u1x - px*u1z
    u2z = px*u1y - py*u1x

    # particle vector in terms of u1 and u2 (dot products)
    rel_ux = ux*u1x + uy*u1y + uz*u1z
    rel_uy = ux*u2x + uy*u2y + uz*u2z
    
    return math.atan2(rel_uy, rel_ux)


# ======================================================================================
# Framework utilities
# ======================================================================================


@njit
def atomic_add(array, idx, value):
    array[idx] += value


@njit
def local_array(shape, dtype):
    return np.zeros(shape, dtype=dtype)


@njit
def access_simulation(program):
    return program
