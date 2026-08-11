import os
import pytest

# Force pure-Python execution for numba in tests.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mcdc.transport.distribution as dist


class MockRNG:
    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def lcg(self, _state_container):
        # MCDC uses dist.rng.lcg(state) as its RNG hook. We override it with this
        # deterministic sequence so tests can be analytic and reproducible.
        if self._i >= len(self._values):
            raise IndexError("MockRNG depleted")
        value = self._values[self._i]
        self._i += 1
        return value


@pytest.fixture
def rng_state():
    """
    Return the minimal RNG state container expected by MCDC.

    MCDC passes a list of per-thread state dicts; tests only need one.
    """
    return [dict(rng_seed=0)]


@pytest.fixture
def rng_sequence():
    """
    Temporarily replace dist.rng.lcg with a deterministic sequence.

    Usage:
        rng_sequence([0.1, 0.2, 0.3])
        ... code under test that calls dist.rng.lcg(...)
    """
    original_lcg = dist.rng.lcg

    def _apply(values):
        # Swap in a mock LCG implementation that returns the provided values.
        rng = MockRNG(values)
        dist.rng.lcg = rng.lcg
        return rng

    # Yield a callable to the test, then restore the real RNG after the test ends.
    yield _apply
    dist.rng.lcg = original_lcg
