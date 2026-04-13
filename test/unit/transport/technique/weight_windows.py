import mcdc
import numpy as np
import pytest
import os

from mcdc.main import preparation
import mcdc.numba_types as type_
from mcdc.transport.technique import (
    roulette_from_weight_window,
    split_from_weight_window,
    query_weight_window,
    weight_windows,
    particle_bank_module,
)
from mcdc.transport.mesh.interface import get_indices
from mcdc.constant import TINY

# =========================================================================== #
# Helper method for creating a dummy model
# =========================================================================== #


def make_mesh():
    # constants
    pitch = 2.0
    height = 10.0
    N = 3

    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-pitch / 2, pitch / N, N),
        y=(-pitch / 2, pitch / N, N),
        z=(0.0, height / N, N),
    )

    return mesh, N


def make_ww_model_params(lower=0.1, target=1.0, upper=1.0, mess_up_size=False):
    import mcdc

    mesh, N = make_mesh()

    if mess_up_size:
        ww_array = np.ones((N, N, 4, 3))
    else:
        ww_array = np.ones((N, N, N, 3))
        ww_array[..., 0] = lower
        ww_array[..., 1] = target
        ww_array[..., 2] = upper

    mcdc.simulation.weight_windows(mesh, ww_array)

    mcdc_container, data = preparation()
    return mcdc_container[0], data


def make_ww_model_distinct():
    import mcdc

    mesh, N = make_mesh()

    ww_array = np.empty((N, N, N, 3))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                val = 100 * i + 10 * j + k + 1
                ww_array[i, j, k, 0] = val
                ww_array[i, j, k, 1] = 1000 + val
                ww_array[i, j, k, 2] = 2000 + val

    mcdc.simulation.weight_windows(mesh, ww_array)

    mcdc_container, data = preparation()
    return mcdc_container[0], data


# =========================================================================== #
# Test error throwing in object creation
# =========================================================================== #


@pytest.mark.parametrize(
    "kwargs, expected_msg",
    [
        # incorrect size
        (
            {"mess_up_size": True},
            "Weight window array has shape (3, 3, 4, 3), but expected (3, 3, 3, 3)",
        ),
        # negative lower
        (
            {"lower": -1.0},
            "Lower bound weights must be strictly positive",
        ),
        # lower > target
        (
            {"lower": 1.0, "target": 0.5},
            "Lower bound weight can not be greater than the target weight",
        ),
        # target > upper
        (
            {"target": 1.5},
            "Target weight can not be greater than the upper bound weight",
        ),
    ],
)
def test_error_throw(capsys, kwargs, expected_msg):
    with pytest.raises(SystemExit):
        make_ww_model_params(**kwargs)

    out = capsys.readouterr().out
    assert expected_msg in out


# =========================================================================== #
# Tests for helper methods
# =========================================================================== #


def test_roullete_from_weight_window():
    # because of rng, want to loop over to hit both branches
    for i in range(10):
        particles = np.zeros(1, type_.particle_data)
        particles[0]["w"] = 0.1
        target = 0.2
        threshold = 0.1 + TINY

        roulette_from_weight_window(particles, threshold, target)
        p = particles[0]
        assert p["w"] == target or not p["alive"]


def test_split_from_weight_window():
    particles = np.zeros(1, type_.particle_data)
    init_weight = 2.0 + TINY
    particles[0]["w"] = init_weight
    threshold = 1.0
    mcdc_obj, data = make_ww_model_distinct()

    # get bank and init size
    bank = mcdc_obj["bank_active"]
    init_bank_size = particle_bank_module.get_bank_size(bank)

    # split
    split_from_weight_window(particles, threshold, mcdc_obj)

    p1 = particles[0]
    num_split = np.ceil(init_weight / threshold)
    num_new = num_split - 1

    # check weight of original particle
    assert p1["w"] == init_weight / num_split
    assert particle_bank_module.get_bank_size(bank) == init_bank_size + num_new
    for i in range(2):
        pnew = bank["particle_data"][init_bank_size + i]
        assert pnew["w"] == p1["w"]


def test_query_weight_window():
    p = np.zeros(1, type_.particle_data)

    mcdc_obj, data = make_ww_model_distinct()

    # hardcode mesh params
    pitch, height, N = 2.0, 10.0, 3
    nx, ny, nz = N, N, N
    xmin, ymin, zmin = -pitch / 2, -pitch / 2, 0.0
    dx, dy, dz = pitch / N, pitch / N, height / N

    # loop over all bins, check query against expected ww
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                p[0]["x"] = xmin + dx * (ix + 0.5)
                p[0]["y"] = ymin + dy * (iy + 0.5)
                p[0]["z"] = zmin + dz * (iz + 0.5)

                lower, target, upper = query_weight_window(p, mcdc_obj, data)
                exp_lower = 100 * ix + 10 * iy + iz + 1
                exp_target = 1000 + exp_lower
                exp_upper = 2000 + exp_lower

                assert lower == exp_lower
                assert target == exp_target
                assert upper == exp_upper
