import numba as nb
import numpy as np

from numba import uint64, njit

# ======================================================================================
# Random number generator
#   LCG with hash seed-split
# ======================================================================================

# LCG parameters
RNG_G = nb.uint64(2806196910506780709)
RNG_C = nb.uint64(1)
RNG_MOD_MASK = nb.uint64(0x7FFFFFFFFFFFFFFF)
RNG_MOD = nb.uint64(0x8000000000000000)

# Splitter seeds
SEED_SPLIT_CENSUS = nb.uint64(0x43454D654E54)
SEED_SPLIT_SOURCE = nb.uint64(0x43616D696C6C65)
SEED_SPLIT_SOURCE_PRECURSOR = nb.uint64(0x546F6464)
SEED_SPLIT_PARTICLE = nb.uint64(0)
SEED_SPLIT_UQ = nb.uint64(0x5368656261)


@njit
def wrapping_mul(a, b):
    return a * b


@njit
def wrapping_add(a, b):
    return a + b


def wrapping_mul_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a * b


def wrapping_add_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a + b


@njit
def split_seed(key, seed):
    """
    murmur_hash64a

    If called from non-jitted function, may need to recast the argument key with numba.uint64
    """
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


@njit
def lcg_(seed):
    seed = uint64(seed)
    return wrapping_add(wrapping_mul(RNG_G, seed), RNG_C) & RNG_MOD_MASK


@njit
def lcg(state_container):
    state = state_container[0]
    state["rng_seed"] = lcg_(state["rng_seed"])
    return state["rng_seed"] / RNG_MOD
