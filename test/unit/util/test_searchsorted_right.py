import numpy as np
import pytest

####

from mcdc.transport.util import searchsorted_right

GRIDS = [
    np.array([0.0, 1.0, 2.0, 5.0, 10.0]),
    np.array([0.0, 1.0, 1.0, 1.0, 3.0]),  # repeated values
    np.array([2.0]),
    np.array([], dtype=np.float64),
    np.array([1, 4, 4, 9]),  # integers, as for interpolation-law boundaries
]


@pytest.mark.parametrize("grid", GRIDS, ids=range(len(GRIDS)))
def test_matches_numpy(grid):
    # Every grid value (ties), points between them, and points outside
    values = np.concatenate([grid, grid + 0.5, grid - 0.5, [-100.0, 100.0]])
    for value in values:
        expected = np.searchsorted(grid, value, side="right")
        assert searchsorted_right(grid, value) == expected, value


def test_random_sorted_grids():
    rng = np.random.default_rng(1)
    for _ in range(200):
        grid = np.sort(rng.integers(0, 10, size=rng.integers(1, 20))).astype(float)
        for value in rng.uniform(-1.0, 11.0, size=20):
            expected = np.searchsorted(grid, value, side="right")
            assert searchsorted_right(grid, value) == expected
