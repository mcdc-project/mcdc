import numpy as np
import pytest

####

from mcdc.transport.util import find_bin


@pytest.fixture
def grid():
    return np.array([0.0, 1.0, 2.0, 5.0, 10.0])


@pytest.fixture
def eps():
    return 1e-5


def test_inside_bins(grid):
    assert find_bin(0.5, grid) == 0
    assert find_bin(1.5, grid) == 1
    assert find_bin(4.9, grid) == 2
    assert find_bin(9.9, grid) == 3


def test_exact_interior_edges(grid):
    assert find_bin(1.0, grid, 0.0, True) == 0
    assert find_bin(1.0, grid, 0.0, False) == 1
    assert find_bin(5.0, grid, 0.0, True) == 2
    assert find_bin(5.0, grid, 0.0, False) == 3


def test_first_edge(grid):
    assert find_bin(0.0, grid, 0.0, True) == -1
    assert find_bin(0.0, grid, 0.0, False) == 0


def test_last_edge(grid):
    assert find_bin(10.0, grid, 0.0, True) == 3
    assert find_bin(10.0, grid, 0.0, False) == -1


def test_near_interior_edges_with_epsilon(grid, eps):
    assert find_bin(1.0 - 1e-6, grid, eps, True) == 0
    assert find_bin(1.0 - 1e-6, grid, eps, False) == 1
    assert find_bin(1.0 + 1e-6, grid, eps, True) == 0
    assert find_bin(1.0 + 1e-6, grid, eps, False) == 1


def test_near_first_edge_with_epsilon(grid, eps):
    assert find_bin(0.0 + 1e-6, grid, eps, True) == -1
    assert find_bin(0.0 + 1e-6, grid, eps, False) == 0


def test_near_last_edge_with_epsilon(grid, eps):
    assert find_bin(10.0 - 1e-6, grid, eps, True) == 3
    assert find_bin(10.0 - 1e-6, grid, eps, False) == -1


def test_out_of_range(grid):
    assert find_bin(-1.0, grid) == -1
    assert find_bin(11.0, grid) == -1
