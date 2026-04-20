import mcdc
import numpy as np
import pytest
import os

from mcdc.main import preparation
import mcdc.numba_types as type_
from mcdc.transport.technique import (
    weight_roulette,
    split_from_weight_window,
    query_weight_window,
    weight_windows,
    bank_split_particles,
    particle_bank_module,
)
from mcdc.transport.mesh.interface import get_indices
from mcdc.constant import TINY
import mcdc.transport.util as util

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
    Ne = 1

    if mess_up_size:
        ww_array = np.ones((Ne, N, N, 4, 3))
    else:
        # global assign for simplicity
        ww_array = np.ones((Ne, N, N, N, 3))
        ww_array[..., 0] = lower
        ww_array[..., 1] = target
        ww_array[..., 2] = upper

    mcdc.simulation.weight_windows(ww_array, mesh=mesh)

    mcdc_container, data = preparation()
    return mcdc_container[0], data


def make_ww_model_distinct():
    import mcdc

    mesh, N = make_mesh()
    energy = np.linspace(0.0, 6.0, 7)
    Ne = 6

    ww_array = np.empty((Ne, N, N, N, 3))

    # value at index is related to index, easy to predict during later test
    for e in range(Ne):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    val = 1000 * e + 100 * i + 10 * j + k + 1
                    ww_array[e, i, j, k, 0] = val
                    ww_array[e, i, j, k, 1] = 10000 + val
                    ww_array[e, i, j, k, 2] = 20000 + val

    mcdc.simulation.weight_windows(ww_array, mesh=mesh, energy=energy)

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
            "Weight window array has shape (1, 3, 3, 4, 3), but expected (1, 3, 3, 3, 3)",
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


def test_roulette_from_weight_bounds():
    # because of rng, want to loop over to hit both branches
    for _ in range(10):
        particles = np.zeros(1, type_.particle_data)
        particles[0]["w"] = 0.1
        target = 0.2
        threshold = 0.1 + TINY

        weight_roulette(particles, threshold, target)
        p = particles[0]
        assert p["w"] == target or not p["alive"]


def test_split_from_weight_window():
    particles = np.zeros(1, type_.particle_data)
    init_weight = 2.0 + TINY
    particles[0]["w"] = init_weight
    threshold = 1.0
    program, data = make_ww_model_distinct()
    simulation = util.access_simulation(program)

    # get bank and init size
    bank = simulation["bank_active"]
    init_bank_size = particle_bank_module.get_bank_size(bank)

    # split
    num_bank = split_from_weight_window(particles, threshold)

    p1 = particles[0]
    num_split = np.ceil(init_weight / threshold)
    num_new = num_split - 1

    # check weight of original particle
    assert p1["w"] == init_weight / num_split
    # check correct number of particles to bank
    assert num_bank ==  num_new


def test_bank_split_particles():
    particle_container = np.zeros(1, type_.particle_data)
    particle = particle_container[0]

    # weight params for both tests 
    particle["w"] = 0.1
    target = 1.0
    
    program, data = make_ww_model_distinct()
    simulation = util.access_simulation(program)

    # get bank and init size
    bank = simulation["bank_active"]
    init_bank_size = particle_bank_module.get_bank_size(bank)

    # define the num to bank
    num_bank = 3

    # no banks rouletted
    threshold = 0.0
    bank_split_particles(particle_container, num_bank, threshold, target, program)

    assert init_bank_size + num_bank == particle_bank_module.get_bank_size(bank)
    for i in range(num_bank):
        pnew = bank["particle_data"][init_bank_size + i]
        assert pnew["w"] == particle["w"]

    # update init bank size
    init_bank_size += num_bank

    # banks rouletted
    num_trials = 10 # need multiple to ensure rng hits a roulette
    threshold = 2 * particle["w"] # arbitrarily greater
    for _ in range(num_trials):
        bank_split_particles(particle_container, num_bank, threshold, target, program)
    num_added = particle_bank_module.get_bank_size(bank) - init_bank_size
    assert num_added < num_trials * num_bank
    for i in range(num_added):
        pnew = bank["particle_data"][init_bank_size + i]
        assert pnew["w"] == target


def test_query_weight_window():
    p = np.zeros(1, type_.particle_data)

    program, data = make_ww_model_distinct()
    simulation = util.access_simulation(program)
    simulation["settings"]["neutron_multigroup_mode"] = False
    # hardcode mesh params
    pitch, height, N = 2.0, 10.0, 3
    nx, ny, nz = N, N, N
    xmin, ymin, zmin = -pitch / 2, -pitch / 2, 0.0
    dx, dy, dz = pitch / N, pitch / N, height / N

    # hardcode energy params
    energies = np.linspace(0.5, 5.5, 6)

    # loop over all bins, check query against expected ww
    for ne, energy in enumerate(energies):
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # put particle in center of current mesh bin
                    p[0]["x"] = xmin + dx * (ix + 0.5)
                    p[0]["y"] = ymin + dy * (iy + 0.5)
                    p[0]["z"] = zmin + dz * (iz + 0.5)
                    # assign energy to be in center of bins
                    p[0]["E"] = energy

                    # query and predict
                    lower, target, upper = query_weight_window(p, simulation, data)
                    exp_lower = 1000 * ne + 100 * ix + 10 * iy + iz + 1
                    exp_target = 10000 + exp_lower
                    exp_upper = 20000 + exp_lower

                    assert lower == exp_lower
                    assert target == exp_target
                    assert upper == exp_upper
