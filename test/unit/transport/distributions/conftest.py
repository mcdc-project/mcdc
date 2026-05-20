import numpy as np
import pytest
from numba import njit

import mcdc.transport.distribution as dist

MOCK_RNG_MAX_VALUES = 8
MOCK_RNG_STATE_DTYPE = np.dtype(
    [
        ("idx", np.int64),
        ("n_values", np.int64),
        ("values", np.float64, (MOCK_RNG_MAX_VALUES,)),
    ]
)


@njit
def mock_lcg_known_sequence(rng_state):
    # Force lcg to pick mock rng values in sequence
    state = rng_state[0]
    i = state["idx"]
    n = state["n_values"]

    if i < n:
        value = state["values"][i]
    else:
        value = np.nan

    state["idx"] = i + 1
    return value


def mock_rng(*values):
    if len(values) == 1 and isinstance(values[0], (list, tuple, np.ndarray)):
        values = tuple(values[0])

    state = np.zeros(1, dtype=MOCK_RNG_STATE_DTYPE)
    state[0]["idx"] = 0
    state[0]["n_values"] = len(values)
    if values:
        state[0]["values"][: len(values)] = np.asarray(values, dtype=np.float64)
    return state


@pytest.fixture
def make_distribution_record():
    # Build one typed record from a dict of field values.
    def _make(record_dtype, values_dict):
        container = np.zeros(1, record_dtype)
        record = container[0]
        for key, value in values_dict.items():
            record[key] = value
        return record

    return _make


@pytest.fixture
def mock_rng_sequence():
    # Replace mcdc lcg with the known sequence
    original_lcg = dist.rng.lcg
    dist.rng.lcg = mock_lcg_known_sequence
    # Yield the mock_rng sequence
    yield mock_rng
    # Reset the lcg. If the test fails, the mock_rng could still be installed and cause later tests to fail
    dist.rng.lcg = original_lcg
