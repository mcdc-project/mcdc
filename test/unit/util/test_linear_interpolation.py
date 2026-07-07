from mcdc.transport.util import linear_interpolation


def test_exact_endpoints():
    assert linear_interpolation(0, 0, 10, 0, 100) == 0
    assert linear_interpolation(10, 0, 10, 0, 100) == 100


def test_midpoint():
    assert linear_interpolation(5, 0, 10, 0, 100) == 50
    assert linear_interpolation(2.5, 0, 10, 0, 100) == 25


def test_negative_slopes():
    assert linear_interpolation(5, 0, 10, 100, 0) == 50
    assert linear_interpolation(2, 0, 4, 4, 0) == 2


def test_non_uniform_interval():
    # Interval [2, 4] mapped to [10, 30]
    assert linear_interpolation(3, 2, 4, 10, 30) == 20


def test_floats():
    result = linear_interpolation(0.5, 0, 1, 0.0, 1.0)
    assert abs(result - 0.5) < 1e-12


def test_extrapolation():
    # x before x1
    assert linear_interpolation(-5, 0, 10, 0, 100) == -50
    # x beyond x2
    assert linear_interpolation(20, 0, 10, 0, 100) == 200


def test_x1_equals_x2_raises_zero_division():
    try:
        linear_interpolation(1, 2, 2, 0, 10)
    except ZeroDivisionError:
        assert True
    else:
        assert False, "Expected ZeroDivisionError when x1 == x2"
