import mcdc
import numpy as np

####

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)
from mcdc.main import preparation

# ======================================================================================
# Setup
# ======================================================================================

# Reference surface description
# General cylinder parallel to z-axis, centered at origin, radius R = 5.0
# f(x, y, z) = x^2 + y^2 - R^2
R = 5.0
durations = np.array([5.0, 5.0, 5.0])
velocities = np.zeros((3, 3))
velocities[:, 0] = np.array([-1.0, 2.0, -3.0])

# Test object: static surface
static_surface = mcdc.Surface.Cylinder(A=1.0, B=1.0, J=-(R**2))

# Test object: moving surface
moving_surface = mcdc.Surface.Cylinder(A=1.0, B=1.0, J=-(R**2))
moving_surface.move(velocities, durations)

# Test object: static surface with general coefficients
general_surface = mcdc.Surface.Cylinder(
    A=1.5, B=0.5, C=2.0, G=-1.0, H=3.0, I=-0.5, J=-4.0
)

# Create the dummy simulation structure and data
structure_container, data = preparation()
structure = structure_container[0]

# Get the "compiled" test objects
static_surface = structure["surfaces"][0]
moving_surface = structure["surfaces"][1]
general_surface = structure["surfaces"][2]

# Particle object for testing
import mcdc.numba_types as type_

particle_container = np.zeros(1, type_.particle_data)
particle = particle_container[0]

# Miscellanies
# For quadratic surfaces, position offset delta gives f ~ 2*R*delta,
# so delta < COINCIDENCE_TOLERANCE / (2*R) is needed for coincidence.
TINY = COINCIDENCE_TOLERANCE / (2.0 * R) * 0.8

# Load modules to be tested
from mcdc.transport.geometry.surface import (
    interface,
    cylinder,
)

# =====================================================================================
# Cylinder core functions
# =====================================================================================


def test_evaluate():
    def run(x, y, answer):
        particle["x"] = x
        particle["y"] = y
        result = cylinder.evaluate(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive side (outside)
    run(x=8.0, y=0.0, answer=39.0)
    # Negative side (inside)
    run(x=3.0, y=0.0, answer=-16.0)


def test_evaluate_general_coefficients():
    def run(x, y, z, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        result = cylinder.evaluate(particle_container, general_surface)
        assert np.isclose(result, answer)

    run(x=1.0, y=-2.0, z=0.5, answer=-7.25)
    run(x=2.0, y=1.0, z=-1.0, answer=6.0)


def test_reflect():
    def run(ux, answer):
        particle["x"] = R
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        cylinder.reflect(particle_container, static_surface)
        assert np.isclose(particle["ux"], answer)

    # From positive direction
    run(ux=0.2, answer=-0.2)
    # From negative direction
    run(ux=-0.1, answer=0.1)


def test_get_normal_component():
    def run(ux, answer):
        particle["x"] = R
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        result = cylinder.get_normal_component(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive direction
    run(ux=0.4, answer=0.4)
    # Negative direction
    run(ux=-0.2, answer=-0.2)
    # Parallel
    run(ux=0.0, answer=0.0)


def test_get_normal_component_general_coefficients():
    particle["x"] = 1.0
    particle["y"] = -2.0
    particle["z"] = 0.5
    particle["ux"] = 0.4
    particle["uy"] = -0.2
    particle["uz"] = 0.1

    result = cylinder.get_normal_component(particle_container, general_surface)
    answer = (2.0 * 0.4 + 1.0 * (-0.2) + 1.5 * 0.1) / (2.0**2 + 1.0**2 + 1.5**2) ** 0.5
    assert np.isclose(result, answer)


def test_get_distance():
    def run(x, ux, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        result = cylinder.get_distance(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive side (outside)
    x = 8.0
    ## Moving closer
    run(x, ux=-0.4, answer=7.5)
    ## Moving away
    run(x, ux=0.3, answer=INF)
    ## Parallel
    run(x, ux=0.0, answer=INF)

    # Negative side (inside)
    x = 3.0
    ## Moving outward (toward near surface)
    run(x, ux=0.4, answer=5.0)
    ## Moving inward (toward far surface)
    run(x, ux=-0.3, answer=80.0 / 3.0)
    ## Parallel
    run(x, ux=0.0, answer=INF)

    # At surface, within tolerance, on the positive side
    x = R + TINY
    ## Moving away
    run(x, ux=0.4, answer=INF)
    ## Moving closer (crosses to far side)
    run(x, ux=-0.4, answer=2.0 * R / 0.4)
    ## Parallel
    run(x, ux=0.0, answer=INF)

    # At surface, within tolerance, on the negative side
    x = R - TINY
    ## Moving away (toward center, crosses to far side)
    run(x, ux=-0.4, answer=2.0 * R / 0.4)
    ## Moving closer
    run(x, ux=0.4, answer=INF)
    ## Parallel
    run(x, ux=0.0, answer=INF)


# =====================================================================================
# Cylinder integrated transport interface
# =====================================================================================


def test_interface_reflect():
    def run(ux, answer):
        particle["x"] = R
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        interface.reflect(particle_container, static_surface)
        assert np.isclose(particle["ux"], answer)

    # From positive direction
    run(ux=0.2, answer=-0.2)
    # From negative direction
    run(ux=-0.1, answer=0.1)


def test_interface_evaluate():
    def run_static(x, y, answer):
        particle["x"] = x
        particle["y"] = y
        result = interface.evaluate(particle_container, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, y, t, answer):
        particle["x"] = x
        particle["y"] = y
        particle["t"] = t
        result = interface.evaluate(particle_container, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side (outside)
    run_static(x=8.0, y=0.0, answer=39.0)
    # Negative side (inside)
    run_static(x=3.0, y=0.0, answer=-16.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin: center x = -3.0
    t = 3.0
    ## Positive side (outside)
    run_moving(x=4.0, y=0.0, t=t, answer=24.0)
    ## Negative side (inside)
    run_moving(x=-4.0, y=0.0, t=t, answer=-24.0)

    # First bin, at grid: center x = -5.0
    t = 5.0
    ## Positive side (outside)
    run_moving(x=4.0, y=0.0, t=t, answer=56.0)
    ## Negative side (inside)
    run_moving(x=-4.0, y=0.0, t=t, answer=-24.0)

    # Interior bin: center x = -1.0
    t = 12.0
    ## Positive side (outside)
    run_moving(x=6.0, y=0.0, t=t, answer=24.0)
    ## Negative side (inside)
    run_moving(x=-4.0, y=0.0, t=t, answer=-16.0)

    # Final bin: center x = -10.0
    t = 100.0
    ## Positive side (outside)
    run_moving(x=0.0, y=0.0, t=t, answer=75.0)
    ## Negative side (inside)
    run_moving(x=-8.0, y=0.0, t=t, answer=-21.0)


def test_interface_get_normal_component():
    def run_static(ux, answer):
        particle["x"] = R
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        speed = 2.0  # Arbitrary
        result = interface.get_normal_component(
            particle_container, speed, static_surface, data
        )
        assert np.isclose(result, answer)

    def run_moving(x, ux, t, speed, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        particle["t"] = t
        result = interface.get_normal_component(
            particle_container, speed, moving_surface, data
        )
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive direction
    run_static(ux=0.4, answer=0.4)
    # Negative direction
    run_static(ux=-0.2, answer=-0.2)
    # Parallel
    run_static(ux=0.0, answer=0.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin: center x = -3.0, velocity = -1.0
    t = 3.0
    x = -3.0 + R
    run_moving(x, ux=0.4, t=t, speed=2.0, answer=0.9)
    run_moving(x, ux=-0.6, t=t, speed=2.0, answer=-0.1)
    run_moving(x, ux=-0.5, t=t, speed=2.0, answer=0.0)

    # Interior bin: center x = 1.0, velocity = 2.0
    t = 8.0
    x = 1.0 + R
    run_moving(x, ux=0.4, t=t, speed=2.0, answer=-0.6)
    run_moving(x, ux=1.0, t=t, speed=2.0, answer=0.0)
    run_moving(x, ux=0.0, t=t, speed=4.0, answer=-0.5)

    # Interior bin: center x = -1.0, velocity = -3.0
    t = 12.0
    x = -1.0 + R
    run_moving(x, ux=-0.2, t=t, speed=10.0, answer=0.1)


def test_interface_check_sense():
    def run_static(x, y, ux, answer):
        particle["x"] = x
        particle["y"] = y
        particle["ux"] = ux
        particle["uy"] = 0.0
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, y, ux, t, speed, answer):
        particle["x"] = x
        particle["y"] = y
        particle["ux"] = ux
        particle["uy"] = 0.0
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    ux = 0.3  # Arbitrary
    ## Positive side (outside)
    run_static(x=8.0, y=0.0, ux=ux, answer=True)
    ## Negative side (inside)
    run_static(x=3.0, y=0.0, ux=ux, answer=False)

    # At surface, positive side
    x = R + TINY
    ## Positive direction (outward)
    run_static(x, y=0.0, ux=0.4, answer=True)
    ## Negative direction (inward)
    run_static(x, y=0.0, ux=-0.4, answer=False)

    # At surface, negative side
    x = R - TINY
    ## Positive direction (outward)
    run_static(x, y=0.0, ux=0.2, answer=True)
    ## Negative direction (inward)
    run_static(x, y=0.0, ux=-0.2, answer=False)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin: center x = -3.0
    t = 3.0
    speed = 2.0
    ## Not at surface
    run_moving(x=4.0, y=0.0, ux=0.2, t=t, speed=speed, answer=True)
    run_moving(x=-4.0, y=0.0, ux=0.2, t=t, speed=speed, answer=False)
    ## At surface
    x = -3.0 + R
    run_moving(x, y=0.0, ux=0.4, t=t, speed=speed, answer=True)
    run_moving(x, y=0.0, ux=-0.6, t=t, speed=speed, answer=False)
    run_moving(x, y=0.0, ux=-0.5, t=t, speed=speed, answer=False)

    # Interior bin: center x = 1.0
    t = 8.0
    speed = 4.0
    x = 1.0 + R
    run_moving(x, y=0.0, ux=0.8, t=t, speed=speed, answer=True)
    run_moving(x, y=0.0, ux=0.5, t=t, speed=speed, answer=False)


def test_interface_get_distance():
    def run_static(x, ux, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, ux, t, speed, answer):
        particle["x"] = x
        particle["y"] = 0.0
        particle["ux"] = ux
        particle["uy"] = 0.0
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side (outside)
    x = 8.0
    ## Moving closer
    run_static(x, ux=-0.4, answer=7.5)
    ## Moving away
    run_static(x, ux=0.3, answer=INF)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # Negative side (inside)
    x = 3.0
    ## Moving outward (toward near surface)
    run_static(x, ux=0.4, answer=5.0)
    ## Moving inward (toward far surface)
    run_static(x, ux=-0.3, answer=80.0 / 3.0)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # At surface, on the positive side
    x = R + TINY
    ## Moving away
    run_static(x, ux=0.4, answer=INF)
    ## Moving closer (crosses to far side)
    run_static(x, ux=-0.4, answer=2.0 * R / 0.4)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # At surface, on the negative side
    x = R - TINY
    ## Moving away (toward center, crosses to far side)
    run_static(x, ux=-0.4, answer=2.0 * R / 0.4)
    ## Moving closer
    run_static(x, ux=0.4, answer=INF)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin intersection
    run_moving(x=6.0, ux=-1.0, t=1.0, speed=2.0, answer=4.0)

    # Crossing after entering the second bin
    run_moving(x=10.0, ux=-1.0, t=2.0, speed=2.0, answer=8.0)

    # Moving away from the surface
    run_moving(x=10.0, ux=1.0, t=6.0, speed=2.0, answer=INF)

    # Starting inside and moving outward
    run_moving(x=-2.0, ux=1.0, t=2.0, speed=2.0, answer=10.0 / 3.0)
