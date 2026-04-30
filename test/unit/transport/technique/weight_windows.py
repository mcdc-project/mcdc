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
    Nt = 1
    Nmu = 1
    Na = 1

    if mess_up_size:
        ww_array = np.ones((Nt, Ne, Nmu, Na, N, N, 4, 3))
    else:
        # global assign for simplicity
        ww_array = np.ones((Nt, Ne, Nmu, Na, N, N, N, 3))
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
    time = np.linspace(0.0, 4.0, 5)
    Nt = 4
    mu = np.linspace(-1.0, 1.0, 5)
    Nmu = 4
    azimuthal = np.linspace(-np.pi, np.pi, 5)
    Na = 4

    ww_array = np.empty((Nt, Ne, Nmu, Na, N, N, N, 3))

    # value at index is related to index, easy to predict during later test
    for t in range(Nt):
      for e in range(Ne):
          for m in range(Nmu):
              for a in range(Na):
                for i in range(N):
                    for j in range(N):
                        for k in range(N):
                            val = 1_000_000*t + 100_000*e + 10_000*m + 1_000*a + 100 * i + 10 * j + k + 1
                            ww_array[t, e, m, a, i, j, k, 0] = val
                            ww_array[t, e, m, a, i, j, k, 1] = 10_000_000 + val
                            ww_array[t, e, m, a, i, j, k, 2] = 20_000_000 + val

    mcdc.simulation.weight_windows(ww_array, mesh=mesh, energy=energy, time=time, mu=mu, azimuthal=azimuthal)

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
            "Weight window array has shape (1, 1, 1, 1, 3, 3, 4, 3), but expected (1, 1, 1, 1, 3, 3, 3, 3)",
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
    program, data = make_ww_model_distinct()
    simulation = util.access_simulation(program)
    bank = simulation["bank_active"]

    def run_split(initial_weight, w_upper=1.0, w_target=0.5, w_lower=0.0):
        particles = np.zeros(1, type_.particle)
        particles[0]["w"] = initial_weight
        particles[0]["alive"] = True
        init_bank_size = particle_bank_module.get_bank_size(bank)

        split_from_weight_window(
            particles,
            w_upper=w_upper,
            w_target=w_target,
            w_lower=w_lower,
            program=program,
        )

        final_bank_size = particle_bank_module.get_bank_size(bank)
        banked_particles = bank["particle_data"][init_bank_size:final_bank_size]
        return particles[0], banked_particles

    # No split occurs when particle weight is at the upper bound.
    particle, banked_particles = run_split(initial_weight=1.0)
    assert particle["w"] == 1.0
    assert len(banked_particles) == 0

    # Integer split, all should be at target weight
    particle, banked_particles = run_split(initial_weight=2.0)
    assert particle["w"] == 0.5
    assert len(banked_particles) == 3
    for banked_particle in banked_particles:
        assert banked_particle["w"] == 0.5

    # Non-integer split, all but last should be at target, last should be residual
    particle, banked_particles = run_split(initial_weight=2.1)
    assert particle["w"] == 0.5
    assert len(banked_particles) == 4
    for banked_particle in banked_particles[:-1]:
        assert banked_particle["w"] == 0.5
    assert banked_particles[-1]["w"] == pytest.approx(0.1)

    # Non-integer split, all but last should be at target, last should be residual
    total_banked = 0
    maximum_bank = 10 * 4
    for _ in range(10):
        particle, banked_particles = run_split(initial_weight=2.1, w_lower=0.2)
        assert particle["w"] == 0.5
        total_banked += len(banked_particles)
        for banked_particle in banked_particles:
            assert banked_particle["w"] == 0.5
    assert total_banked < maximum_bank


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
    times = np.linspace(0.5, 3.5, 4)
    energies = np.linspace(0.5, 5.5, 6)
    mus = np.linspace(-0.75, 0.75, 4)
    azimuthals = np.linspace(-3.0 * np.pi / 4.0, 3.0 * np.pi / 4.0, 4)

    # loop over all bins, check query against expected ww
    for it, time in enumerate(times):
      for ie, energy in enumerate(energies):
          for imu, mu in enumerate(mus):
              for ia, azimuthal in enumerate(azimuthals):
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            # put particle in center of current mesh bin
                            p[0]["x"] = xmin + dx * (ix + 0.5)
                            p[0]["y"] = ymin + dy * (iy + 0.5)
                            p[0]["z"] = zmin + dz * (iz + 0.5)
                            # assign params to be in center of bins
                            p[0]["t"] = time
                            p[0]["E"] = energy
                            p[0]["ux"] = np.sqrt(1.0 - mu**2) * np.cos(azimuthal)
                            p[0]["uy"] = np.sqrt(1.0 - mu**2) * np.sin(azimuthal)
                            p[0]["uz"] = mu

                            # query and predict
                            lower, target, upper = query_weight_window(p, simulation, data)
                            exp_lower = 1_000_000*it + 100_000*ie + 10_000*imu + 1_000*ia + 100 * ix + 10 * iy + iz + 1
                            exp_target = 10_000_000 + exp_lower
                            exp_upper = 20_000_000 + exp_lower

                            assert lower == exp_lower
                            assert target == exp_target
                            assert upper == exp_upper
