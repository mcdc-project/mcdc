import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_

from mcdc.constant import INF
from mcdc.transport.geometry.surface import interface, torus, torus_z

# ======================================================================================
# Setup
# ======================================================================================

A = 0.0
B = 0.0
C = 0.0
R = 1.0
r = 0.5

axis_aligned_surface = None
reference_surface = None
oblique_surface = None
data = None
particle_container = None
particle = None


@pytest.fixture(autouse=True)
def setup_geometry_case():
    global axis_aligned_surface, reference_surface, oblique_surface
    global data, particle_container, particle

    axis_aligned_surface_obj = mcdc.Surface.Torus(
        center=[A, B, C],
        axis=[0.0, 0.0, 1.0],
        R=R,
        r=r,
    )
    reference_surface_obj = mcdc.Surface.TorusZ(A=A, B=B, C=C, R=R, r=r)

    # The x-axis is perpendicular to the axis [0, 1, 1], so points on the x-axis
    # give a simple analytic slice through the arbitrary-axis torus.
    oblique_surface_obj = mcdc.Surface.Torus(
        center=[A, B, C],
        axis=[0.0, 1.0, 1.0],
        R=R,
        r=r,
    )

    from mcdc.main import preparation

    structure_container, data = preparation()
    structure = structure_container[0]
    axis_aligned_surface = structure["surfaces"][axis_aligned_surface_obj.ID]
    reference_surface = structure["surfaces"][reference_surface_obj.ID]
    oblique_surface = structure["surfaces"][oblique_surface_obj.ID]
    particle_container = np.zeros(1, type_.particle_data)
    particle = particle_container[0]


# ======================================================================================
# Surface creation
# ======================================================================================


def test_zero_axis_error():
    with pytest.raises(SystemExit):
        mcdc.Surface.Torus(axis=[0.0, 0.0, 0.0], R=1.0, r=0.5)


# ======================================================================================
# General torus core functions
# ======================================================================================


def test_evaluate_matches_torus_z():
    def run(x, y, z):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z

        general_result = torus.evaluate(particle_container, axis_aligned_surface)
        reference_result = torus_z.evaluate(particle_container, reference_surface)

        np.testing.assert_allclose(
            general_result, reference_result, rtol=1e-5, atol=1e-8
        )

    run(x=1.0, y=0.0, z=0.0)
    run(x=1.5, y=0.0, z=0.0)
    run(x=0.0, y=1.0, z=0.0)
    run(x=0.0, y=0.0, z=2.0)


def test_oblique_evaluate():
    def run(x, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["z"] = 0.0

        result = torus.evaluate(particle_container, oblique_surface)
        np.testing.assert_allclose(result, answer, rtol=1e-5, atol=1e-8)

    # Inside
    run(x=R, answer=-0.9375)

    # On surface
    run(x=R + r, answer=0.0)

    # Outside
    run(x=R + 2.0 * r, answer=6.5625)


def test_oblique_get_normal_component():
    def run(x, ux, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["z"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        particle["uz"] = 0.0

        result = torus.get_normal_component(particle_container, oblique_surface)
        np.testing.assert_allclose(result, answer, rtol=1e-5, atol=1e-8)

    # On the outer surface, the outward normal is +x
    run(x=R + r, ux=-1.0, answer=-1.0)
    run(x=R + r, ux=1.0, answer=1.0)


def test_oblique_get_distance():
    def run(x, ux, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["z"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        particle["uz"] = 0.0

        result = torus.get_distance(particle_container, oblique_surface)
        np.testing.assert_allclose(result, answer, rtol=1e-5, atol=1e-8)

    # Outside, moving inward toward the outer surface
    run(x=R + 2.0 * r, ux=-1.0, answer=r)

    # Outside, moving away
    run(x=R + 2.0 * r, ux=1.0, answer=INF)


# ======================================================================================
# General torus integrated transport interface
# ======================================================================================


def test_interface_evaluate_oblique():
    particle["x"] = R
    particle["y"] = 0.0
    particle["z"] = 0.0

    result = interface.evaluate(particle_container, oblique_surface, data)
    np.testing.assert_allclose(result, -0.9375, rtol=1e-5, atol=1e-8)


def test_interface_get_distance_oblique():
    particle["x"] = R + 2.0 * r
    particle["y"] = 0.0
    particle["z"] = 0.0
    particle["ux"] = -1.0
    particle["uy"] = 0.0
    particle["uz"] = 0.0

    speed = 1.0
    result = interface.get_distance(particle_container, speed, oblique_surface, data)
    np.testing.assert_allclose(result, r, rtol=1e-5, atol=1e-8)
