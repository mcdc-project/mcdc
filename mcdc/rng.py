import numpy as np

from numba import (
    uint64
)

def wrapping_mul(a, b):
    return a * b


def wrapping_mul_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a * b


def split_seed(key, seed):
    """murmur_hash64a"""
    multiplier = uint64(0xC6A4A7935BD1E995)
    length = uint64(8)
    rotator = uint64(47)
    key = uint64(key)
    seed = uint64(seed)

    hash_value = uint64(seed) ^ wrapping_mul(length, multiplier)

    key = wrapping_mul(key, multiplier)
    key ^= key >> rotator
    key = wrapping_mul(key, multiplier)
    hash_value ^= key
    hash_value = wrapping_mul(hash_value, multiplier)

    hash_value ^= hash_value >> rotator
    hash_value = wrapping_mul(hash_value, multiplier)
    hash_value ^= hash_value >> rotator
    return hash_value
