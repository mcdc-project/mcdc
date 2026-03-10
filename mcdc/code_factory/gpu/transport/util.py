# import harmonize

from numba import njit


@njit
def atomic_add(array, idx, value):
    harmonize.array_atomic_add(array, idx, value)
