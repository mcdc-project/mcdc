from numba import njit


@njit
def atomic_add(array, idx, value):
    return harmonize.array_atomic_add(array, idx, value)
