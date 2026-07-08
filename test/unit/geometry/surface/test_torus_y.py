import mcdc
import math
import numpy as np
import pytest

####

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)
import mcdc.numba_types as type_

# ======================================================================================
# Setup
# ======================================================================================

# Reference surface description
A = 0.0
B = 0.0
C = 0.0
R = 1.0
r = 0.5
durations = np.array([5.0, 5.0, 5.0])
velocities = np.zeros((3, 3))
velocities[:, 0] = np.array([-1.0, 2.0, -3.0])

static_surface = None
moving_surface = None
data = None
particle_container = None
particle = None

# Miscellanies
TINY = COINCIDENCE_TOLERANCE * 0.1  # Tiny value within coincidence tolerance

# Load modules to be tested
from mcdc.transport.geometry.surface import (
    interface,
    torus_y,
)


@pytest.fixture(autouse=True)
def setup_geometry_case(compile_surfaces):
    global static_surface, moving_surface, data, particle_container, particle

    # Test object: static surface
    static_surface_obj = mcdc.Surface.TorusY(A=A, B=B, C=C, R=R, r=r)

    # Test object: moving surface
    moving_surface_obj = mcdc.Surface.TorusY(A=A, B=B, C=C, R=R, r=r)
    moving_surface_obj.move(velocities, durations)

    # Create the dummy simulation structure and get the compiled test objects.
    static_surface, moving_surface, data = compile_surfaces(
        static_surface_obj, moving_surface_obj
    )

    # Particle object for testing
    particle_container = np.zeros(1, type_.particle_data)
    particle = particle_container[0]


# =====================================================================================
# Torus-Y core functions
# =====================================================================================


def test_evaluate():
    def run(x, z, y, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        result = torus_y.evaluate(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Inside
    run(x=1.0, z=0.0, y=0.0, answer=-0.9375)
    # Outside
    run(x=0.0, z=-1.0, y=5.0, answer=711.5625)


# Answers parameter is a numpy array of the correct [ux, uz, uy] values of the reflected particle
def test_reflect():
    def run(x, z, y, ux, uz, uy, answers):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        torus_y.reflect(particle_container, static_surface)
        directions = np.array([particle["ux"], particle["uz"], particle["uy"]])
        assert np.allclose(directions, answers)

    # Particle traveling in through the Top of the torus
    run(x=R, z=0.0, y=(r + TINY), ux=0.0, uz=0.0, uy=-1.0, answers=np.array([0, 0, 1]))
    # Particle traveling out through the Top of the torus
    run(x=R, z=0.0, y=(r - TINY), ux=0.0, uz=0.0, uy=1.0, answers=np.array([0, 0, -1]))

    # Particle traveling in through the Bottom of the torus
    run(x=0.0, z=R, y=-(r + TINY), ux=0.0, uz=0.0, uy=1.0, answers=np.array([0, 0, -1]))
    # Particle traveling out through the Bottom of the torus
    run(x=0.0, z=R, y=-(r - TINY), ux=0.0, uz=0.0, uy=-1.0, answers=np.array([0, 0, 1]))

    root = math.sqrt(2) / 2  # 45 degree X-Y lengths on the unit circle
    d = (
        R + r
    ) * root  # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the x-axis
    run(
        x=(d + TINY),
        z=(d + TINY),
        y=0.0,
        ux=-root,
        uz=-root,
        uy=0.0,
        answers=np.array([root, root, 0]),
    )
    # Particle traveling out through the Side of the torus as above
    run(
        x=(d - TINY),
        z=(d - TINY),
        y=0.0,
        ux=root,
        uz=root,
        uy=0.0,
        answers=np.array([-root, -root, 0]),
    )


def test_get_normal_component():
    def run(x, z, y, ux, uz, uy, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        result = torus_y.get_normal_component(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Particle traveling in through the Top of the torus
    run(x=R, z=0.0, y=(r + TINY), ux=0.0, uz=0.0, uy=-1.0, answer=-1)

    # Particle traveling out through the Top of the torus
    run(x=0.0, z=R, y=(r - TINY), ux=0.0, uz=0.0, uy=1.0, answer=1)

    # Particle moving parallel to the torus
    run(x=0.0, z=R, y=-(0.5 + TINY), ux=0.0, uz=1, uy=0.0, answer=0)


def test_get_distance():
    def run(x, z, y, ux, uz, uy, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        result = torus_y.get_distance(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Outside Torus
    x = R
    z = 0.0
    y = r + 1
    ## Moving closer
    run(x, z, y, ux=0.0, uz=0.0, uy=-1.0, answer=1.0)
    ## Moving away
    run(x, z, y, ux=0.0, uz=0.0, uy=1.0, answer=INF)
    ## Parallel
    run(x, z, y, ux=1.0, uz=0.0, uy=0.0, answer=INF)

    # Inside Torus
    x = 0.0
    z = R
    y = r / 2
    ## Moving Up
    run(x, z, y, ux=0.0, uz=0.0, uy=1, answer=(r / 2))
    ## Moving Down
    run(x, z, y, ux=0.0, uz=0.0, uy=-1.0, answer=(3 * (r / 2)))

    # At surface, within tolerance, on the outside
    x = R
    z = 0.0
    y = r + TINY
    ## Moving away
    run(x, z, y, ux=0.0, uz=0.0, uy=1.0, answer=INF)
    ## Moving closer
    run(x, z, y, ux=0.0, uz=0.0, uy=-1.0, answer=(2 * r))
    ## Parallel
    run(x, z, y, ux=1.0, uz=0.0, uy=0.0, answer=INF)

    # At surface, within tolerance, on the inside
    x = 0.0
    z = R
    y = r - TINY
    ## Moving up
    run(x, z, y, ux=0.0, uz=0.0, uy=1.0, answer=INF)
    ## Moving down
    run(x, z, y, ux=0.0, uz=0.0, uy=-1.0, answer=(2 * r))


# =====================================================================================
# Torus-Y integrated transport interface
# =====================================================================================


def test_interface_reflect():
    def run(x, z, y, ux, uz, uy, answers):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        interface.reflect(particle_container, static_surface)
        directions = np.array([particle["ux"], particle["uz"], particle["uy"]])
        assert np.allclose(directions, answers)

    run(x=R, z=0.0, y=(r + TINY), ux=0.0, uz=0.0, uy=-1.0, answers=np.array([0, 0, 1]))
    run(x=R, z=0.0, y=(r - TINY), ux=0.0, uz=0.0, uy=1.0, answers=np.array([0, 0, -1]))
    run(x=0.0, z=R, y=-(r + TINY), ux=0.0, uz=0.0, uy=1.0, answers=np.array([0, 0, -1]))
    run(x=0.0, z=R, y=-(r - TINY), ux=0.0, uz=0.0, uy=-1.0, answers=np.array([0, 0, 1]))

    root = math.sqrt(2) / 2  # 45 degree X-Y lengths on the unit circle
    d = (
        R + r
    ) * root  # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the x-axis
    run(
        x=(d + TINY),
        z=(d + TINY),
        y=0.0,
        ux=-root,
        uz=-root,
        uy=0.0,
        answers=np.array([root, root, 0]),
    )
    # Particle traveling out through the Side of the torus as above
    run(
        x=(d - TINY),
        z=(d - TINY),
        y=0.0,
        ux=root,
        uz=root,
        uy=0.0,
        answers=np.array([-root, -root, 0]),
    )


def test_interface_evaluate():

    def run_static(x, z, y, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        result = interface.evaluate(particle_container, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, z, y, ux, uz, uy, t, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        particle["t"] = t
        result = interface.evaluate(particle_container, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Inside
    run_static(x=1.0, z=0.0, y=0.0, answer=-0.9375)
    # Outside
    run_static(x=0.0, z=-1.0, y=5.0, answer=711.5625)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin
    t = 3.0  # Torus center x-position = -3.0 as the torus has a velocity of -1 and started in the center
    ux = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-3.0, z=1.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-3.0, z=-1.0, y=5.0, ux=ux, uz=uz, uy=uy, t=t, answer=711.5625)

    # First bin, at grid
    t = 5.0  # Torus center x-position = -5.0
    ux = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    uy = -0.3  # Arbitrary
    ## Inside side
    run_moving(x=-5.0, z=1.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-5.0, z=-1.0, y=5.0, ux=ux, uz=uz, uy=uy, t=t, answer=711.5625)

    # Interior bin
    t = 12.0  # Torus center x-position = -1.0 due to velocity and duration values
    ux = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-1.0, z=1.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-1.0, z=-1.0, y=5.0, ux=ux, uz=uz, uy=uy, t=t, answer=711.5625)

    # Interior bin, at grid
    t = 15.0  # Surface x-position = -10.0
    ux = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    uy = -0.3  # Arbitrary
    ## Inside side
    run_moving(x=-10.0, z=1.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-10.0, z=-1.0, y=5.0, ux=ux, uz=uz, uy=uy, t=t, answer=711.5625)

    # Final bin
    t = 100.0  # Surface x-position = -10.0
    ux = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-10.0, z=1.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-10.0, z=-1.0, y=5.0, ux=ux, uz=uz, uy=uy, t=t, answer=711.5625)


def test_interface_get_normal_component():
    def run_static(x, z, y, ux, uz, uy, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        speed = 2.0  # Arbitrary
        result = interface.get_normal_component(
            particle_container, speed, static_surface, data
        )
        assert np.isclose(result, answer)

    def run_moving(x, z, y, ux, uz, uy, t, speed, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        particle["t"] = t
        result = interface.get_normal_component(
            particle_container, speed, moving_surface, data
        )
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Particle traveling in through the Top of the torus
    run_static(x=R, z=0.0, y=(r + TINY), ux=0.0, uz=0.0, uy=-1.0, answer=-1)
    # Particle traveling out through the Top of the torus
    run_static(x=0.0, z=R, y=(r - TINY), ux=0.0, uz=0.0, uy=1.0, answer=1)
    # Particle moving parallel to the torus
    run_static(x=0.0, z=R, y=-(0.5 + TINY), ux=0.0, uz=1, uy=0.0, answer=0)

    # =================================================================================
    # Moving
    # =================================================================================

    # Surface moving in the positive direction
    t = 8.0  # Surface x-velocity = 2.0, center of torus x position at 1
    #
    ## Positive direction
    ux = 0.4
    uz = 0.0
    uy = 0.0
    ### Faster (normal component values on the very centerline of the torus should match an x-plane)
    run_moving(
        x=(1 + R + r),
        z=0.0,
        y=0.0,
        ux=ux,
        uz=uz,
        uy=uy,
        t=t,
        speed=6.0,
        answer=0.4 / 6.0,
    )
    ### Slower (change sign)
    run_moving(
        x=(1 + R + r),
        z=0.0,
        y=0.0,
        ux=ux,
        uz=uz,
        uy=uy,
        t=t,
        speed=2.0,
        answer=-1.2 / 2.0,
    )
    ### Same speed (cancel out)
    run_moving(
        x=(1 + R + r), z=0.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, speed=5.0, answer=0.0
    )
    #
    ## Negative direction
    run_moving(
        x=(1 + R + r),
        z=0.0,
        y=0.0,
        ux=-0.4,
        uz=uz,
        uy=uy,
        t=t,
        speed=6.0,
        answer=-4.4 / 6.0,
    )
    ## Parallel
    run_moving(
        x=(1 + R + r),
        z=0.0,
        y=0.0,
        ux=-0.0,
        uz=uz,
        uy=uy,
        t=t,
        speed=6.0,
        answer=-2.0 / 6.0,
    )

    # Surface moving in the negative direction
    t = 10.0  # Surface x-velocity = -3.0, center of torus x position at 5
    #
    ## Negative direction
    ux = -0.4
    uz = 0.0
    uy = 0.0
    ### Faster
    run_moving(
        x=(5 + R + r),
        z=0.0,
        y=0.0,
        ux=ux,
        uz=uz,
        uy=uy,
        t=t,
        speed=8.0,
        answer=-0.2 / 8.0,
    )
    ### Slower (change sign)
    run_moving(
        x=(5 + R + r),
        z=0.0,
        y=0.0,
        ux=ux,
        uz=uz,
        uy=uy,
        t=t,
        speed=2.0,
        answer=2.2 / 2.0,
    )
    ### Same speed (cancel out)
    run_moving(
        x=(5 + R + r), z=0.0, y=0.0, ux=ux, uz=uz, uy=uy, t=t, speed=7.5, answer=0.0
    )
    #
    ## Positive direction
    run_moving(
        x=(5 + R + r),
        z=0.0,
        y=0.0,
        ux=0.4,
        uz=uz,
        uy=uy,
        t=t,
        speed=8.0,
        answer=6.2 / 8.0,
    )
    ## Parallel
    run_moving(
        x=(5 + R + r),
        z=0.0,
        y=0.0,
        ux=0.0,
        uz=uz,
        uy=uy,
        t=t,
        speed=8.0,
        answer=3.0 / 8.0,
    )


def test_interface_check_sense():  # Returns true if the particle is on the outside of the torus, and false if it's on the inside (particle direction and speed tiebreak)
    def run_static(x, z, y, ux, uz, uy, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, z, y, ux, uz, uy, t, speed, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    z = 0.0
    y = 0.0
    ux = 0.3  # Arbitrary
    uz = 0.0
    uy = 0.0
    ## Positive side
    run_static(x=3.0, z=z, y=y, ux=ux, uz=uz, uy=uy, answer=True)
    ## Negative side
    run_static(x=-4.0, z=z, y=y, ux=ux, uz=uz, uy=uy, answer=True)

    # At surface, outside
    x = R + r + TINY
    z = 0.0
    y = 0.0
    uz = 0.0
    uy = 0.0
    ## Outward direction
    run_static(x=x, z=z, y=y, ux=0.4, uz=uz, uy=uy, answer=True)
    ## Inward direction
    run_static(x=x, z=z, y=y, ux=-0.4, uz=uz, uy=uy, answer=False)

    # At surface, inside
    x = R + r - TINY
    z = 0.0
    y = 0.0
    uz = 0.0
    uy = 0.0
    ## Outward direction
    run_static(x=x, z=z, y=y, ux=0.2, uz=uz, uy=uy, answer=True)
    ## Inward direction
    run_static(x=x, z=z, y=y, ux=-0.2, uz=uz, uy=uy, answer=False)

    # =================================================================================
    # Moving: Surface moving in the positive X direction
    # =================================================================================
    t = 8.5  # Surface x-center = 2.0; surface x-velocity = 2.0

    # Not at surface
    z = 0.0
    y = 0.0
    ux = 0.3  # Arbitrary
    uz = 0.0
    uy = 0.0
    speed = 3.0  # Arbitrary
    ## Outside
    run_moving(x=13.0, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=speed, answer=True)
    ## Inside
    run_moving(x=3.0, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=speed, answer=False)

    # At surface, outside
    x = 3.5 + TINY
    z = 0.0
    y = 0.0
    ## Positive direction (same direction)
    ux = 0.4
    uz = 0.0
    uy = 0.0
    ### Faster
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(x=x, z=z, y=y, ux=-0.4, uz=uz, uy=uy, t=t, speed=6.0, answer=False)

    # At surface, inside
    x = 3.5 - TINY
    z = 0.0
    y = 0.0
    ## Positive direction (same direction)
    ux = 0.4
    uz = 0.0
    uy = 0.0
    ### Faster
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(x=x, z=z, y=y, ux=-0.4, uz=uz, uy=uy, t=t, speed=6.0, answer=False)

    # =================================================================================
    # Moving: Surface moving in the negative direction
    # =================================================================================
    t = 13.0  # Surface x-center = -4.0; surface x-velocity = -3.0

    # Not at surface
    z = 0.0
    y = 0.0
    ux = 0.3  # Arbitrary
    uz = 0.0
    uy = 0.0
    speed = 3.0  # Arbitrary
    ## Outside
    run_moving(x=0.0, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=speed, answer=True)
    ## Inside
    run_moving(x=-5.0, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=speed, answer=False)

    # At surface, outside
    x = -2.5 + TINY
    z = 0.0
    y = 0.0
    ## Positive direction (opposite direction)
    uz = 0.0
    uy = 0.0
    run_moving(x=x, z=z, y=y, ux=0.6, uz=uz, uy=uy, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    ux = -0.6
    ### Faster
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=5.0, answer=False)

    # At surface, inside
    x = -2.5 - TINY
    z = 0.0
    y = 0.0
    ## Positive direction (opposite direction)
    uz = 0.0
    uy = 0.0
    run_moving(x=x, z=z, y=y, ux=0.6, uz=uz, uy=uy, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    ux = -0.6
    ### Faster
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=5.0, answer=False)


def test_interface_get_distance():
    def run_static(x, z, y, ux, uz, uy, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, z, y, ux, uz, uy, t, speed, answer):
        particle["x"] = x
        particle["z"] = z
        particle["y"] = y
        particle["ux"] = ux
        particle["uz"] = uz
        particle["uy"] = uy
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    x = 6.5  # Arbitrary
    z = 0.0
    y = 0.0
    uz = 0.0
    uy = 0.0
    ## Positive direction (moving away)
    run_static(x=x, z=z, y=y, ux=0.3, uz=uz, uy=uy, answer=INF)
    ## Negative direction (moving closer)
    run_static(x=x, z=z, y=y, ux=-0.3, uz=uz, uy=uy, answer=5.0 / 0.3)
    ## Parallel
    run_static(x=x, z=z, y=y, ux=0.0, uz=1.0, uy=uy, answer=INF)

    # Negative side
    x = -5.5  # Arbitrary
    z = 0.0
    y = 0.0
    uz = 0.0
    uy = 0.0
    ## Positive direction (moving closer)
    run_static(x=x, z=z, y=y, ux=0.3, uz=uz, uy=uy, answer=4.0 / 0.3)
    ## Negative direction (moving away)
    run_static(x=x, z=z, y=y, ux=-0.3, uz=uz, uy=uy, answer=INF)
    ## Parallel
    run_static(x=x, z=z, y=y, ux=0.0, uz=1.0, uy=uy, answer=INF)

    # At surface, on the outside
    x = 1.5 + TINY
    z = 0.0
    y = 0.0
    uz = 0.0
    uy = 0.0
    ## Positive direction
    run_static(x=x, z=z, y=y, ux=0.4, uz=uz, uy=uy, answer=INF)
    ## Negative direction
    run_static(x=x, z=z, y=y, ux=-0.4, uz=uz, uy=uy, answer=1.0 / 0.4)
    ## Parallel
    run_static(x=x, z=z, y=y, ux=0.0, uz=1.0, uy=uy, answer=INF)

    # At surface, on the inside
    x = 1.5 - TINY
    z = 0.0
    y = 0.0
    uz = 0.0
    uy = 0.0
    ## Positive direction
    run_static(x=x, z=z, y=y, ux=0.4, uz=uz, uy=uy, answer=INF)
    ## Negative direction
    run_static(x=x, z=z, y=y, ux=-0.4, uz=uz, uy=uy, answer=1.0 / 0.4)
    ## Parallel
    run_static(x=x, z=z, y=y, ux=0.0, uz=1.0, uy=uy, answer=INF)

    # TODO: Numerical Risk - Add off-midplane and near-tangent moving-distance cases.
    # ============================================================================================
    # Moving (Only testing a hit on the midline of the torus in positive and negative x directions)
    # ============================================================================================
    # Bin 0: t = [ 0.0,  5.0]; surface_x_center = [-5.0,    0.0]; surface_speed = -1.0
    # Bin 1: t = [ 5.0, 10.0]; surface_x_center = [-5.0,    5.0]; surface_speed =  2.0
    # Bin 2: t = [10.0, 15.0]; surface_x_center = [-10.0,   5.0]; surface_speed = -3.0
    # Bin 3: t = [15.0,  INF]; surface_x_center = [-10.0, -10.0]; surface_speed =  0.0

    # surface_x_center (0.0   --->  -5.0)
    # surface_x_center (-5.0  --->   5.0)
    # surface_x_center (5.0,  ---> -10.0)
    # surface_x_center (-10.0 ---> -10.0)

    # This is the distance from the particle to the center of the torus, not counting the inner or outer radii
    def center_distance(x, ux, speed, bin_idx):
        # Time when the surface enters the final bin to be evaluated (0 when evaluating bin 0, and 5 when evaluating bin 1)
        t0 = 0.0 + np.sum(durations[:bin_idx])

        # Starting position of the surface at the beginning of the last bin to be evaluated (0 for bin 0, and -5 for bin 1)
        surface_x = A + np.sum(durations[:bin_idx] * velocities[:bin_idx, 0])

        # Speed of the surface in the final bin to be evaluated (-1 for bin 0)
        surface_speed = velocities[bin_idx, 0]

        return ((surface_speed * -t0) - (x - surface_x)) / (ux - surface_speed / speed)

    def outer_surface_distance(x, ux, speed, bin_idx):
        surface_speed = velocities[bin_idx, 0]
        relative_ux = ux - surface_speed / speed
        return center_distance(x, ux, speed, bin_idx) - (R + r) / abs(relative_ux)

    # Start from the beginning
    t = 0.0

    # Positive x side of the torus
    x = 2.0
    z = 0.0
    y = 0.0
    #
    ## Positive direction (moving away)
    ux = 0.4
    uz = 0.0
    uy = 0.0
    ### Surface catches up after reversing direction in bin 1
    answer = outer_surface_distance(x, ux, speed=1.0, bin_idx=1)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=1.0, answer=answer)
    ### Hit (rear-ended by the surface)
    answer = outer_surface_distance(
        x, ux, speed=0.9, bin_idx=1
    )  # Collision in bin 1 where the surface catches up to the particle
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=0.9, answer=answer)
    #
    ## Negative direction (moving closer)
    ux = -0.4
    uz = 0.0
    uy = 0.0
    ### Hit (rear-end the surface)
    answer = outer_surface_distance(
        x, ux, speed=3.0, bin_idx=0
    )  # Collision in bin 0 where the surface is running away from particle
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=3.0, answer=answer)
    ### Hit (head-on opposite directions)
    answer = outer_surface_distance(x, ux, speed=0.1, bin_idx=1)  # Collision in bin 1
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=0.1, answer=answer)

    # Negative x side of the torus
    x = -2.0
    z = 0.0
    y = 0.0
    #
    ## Negative direction (moving away)
    ux = -0.4
    uz = 0.0
    uy = 0.0
    ### Surface catches up in bin 0
    answer = outer_surface_distance(x, ux, speed=2.0, bin_idx=0)
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=2.0, answer=answer)
    ### Hit (rear-ended by the surface)
    answer = outer_surface_distance(x, ux, speed=0.4, bin_idx=0)  # Collision in bin 0
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=0.4, answer=answer)
    #
    ## Positive direction (moving closer)
    ux = 0.4
    uz = 0.0
    uy = 0.0
    ### Hit (head-on)
    answer = outer_surface_distance(x, ux, speed=0.1, bin_idx=0)  # Collision in bin 0
    run_moving(x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=0.1, answer=answer)
    ### Hit (rear-end the surface)
    x = -20.0
    answer = outer_surface_distance(
        x, ux, speed=(20.0 / 3.0), bin_idx=1
    )  # Collision in bin 1
    run_moving(
        x=x, z=z, y=y, ux=ux, uz=uz, uy=uy, t=t, speed=(20.0 / 3.0), answer=answer
    )
