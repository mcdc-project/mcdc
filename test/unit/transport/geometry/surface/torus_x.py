import mcdc
import math
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
A = 0.0
B = 0.0
C = 0.0
R = 1.0
r = 0.5
durations = np.array([5.0, 5.0, 5.0])
velocities = np.zeros((3, 3))
velocities[:, 1] = np.array([-1.0, 2.0, -3.0])

# Test object: static surface
static_surface = mcdc.Surface.TorusX(A=A, B=B, C=C, R=R, r=r)

# Test object: moving surface
moving_surface = mcdc.Surface.TorusX(A=A, B=B, C=C, R=R, r=r)
moving_surface.move(velocities, durations)

# Create the dummy simulation structure and data
structure_container, data = preparation()
structure = structure_container[0]

# Get the "compiled" test objects
static_surface = structure["surfaces"][0]
moving_surface = structure["surfaces"][1]

# Particle object for testing
import mcdc.numba_types as type_

particle_container = np.zeros(1, type_.particle_data)
particle = particle_container[0]

# Miscellanies
TINY = COINCIDENCE_TOLERANCE * 0.1  # Tiny value within coincidence tolerance

# Load modules to be tested
from mcdc.transport.geometry.surface import (
    interface,
    torus_x,
)

# =====================================================================================
# Torus-X core functions
# =====================================================================================


def test_evaluate():
    def run(y, z, x, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        result = torus_x.evaluate(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Inside
    run(y=1.0, z=0.0, x=0.0, answer=-0.9375)
    # Outside
    run(y=0.0, z=-1.0, x=5.0, answer=711.5625)


# Answers parameter is a numpy array of the correct [uy, uz, ux] values of the reflected particle
def test_reflect():
    def run(y, z, x, uy, uz, ux, answers):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        torus_x.reflect(particle_container, static_surface)
        directions = np.array([particle["uy"], particle["uz"], particle["ux"]])
        assert np.allclose(directions, answers)

    # Particle traveling in through the Top of the torus
    run(y=R, z=0.0, x=(r + TINY), uy=0.0, uz=0.0, ux=-1.0, answers=np.array([0, 0, 1]))
    # Particle traveling out through the Top of the torus
    run(y=R, z=0.0, x=(r - TINY), uy=0.0, uz=0.0, ux=1.0, answers=np.array([0, 0, -1]))

    # Particle traveling in through the Bottom of the torus
    run(y=0.0, z=R, x=-(r + TINY), uy=0.0, uz=0.0, ux=1.0, answers=np.array([0, 0, -1]))
    # Particle traveling out through the Bottom of the torus
    run(y=0.0, z=R, x=-(r - TINY), uy=0.0, uz=0.0, ux=-1.0, answers=np.array([0, 0, 1]))

    root = math.sqrt(2) / 2  # 45 degree X-Y lengths on the unit circle
    d = (
        R + r
    ) * root  # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the y-axis
    run(
        y=(d + TINY),
        z=(d + TINY),
        x=0.0,
        uy=-root,
        uz=-root,
        ux=0.0,
        answers=np.array([root, root, 0]),
    )
    # Particle traveling out through the Side of the torus as above
    run(
        y=(d - TINY),
        z=(d - TINY),
        x=0.0,
        uy=root,
        uz=root,
        ux=0.0,
        answers=np.array([-root, -root, 0]),
    )


def test_get_normal_component():
    def run(y, z, x, uy, uz, ux, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        result = torus_x.get_normal_component(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Particle traveling in through the Top of the torus
    run(y=R, z=0.0, x=(r + TINY), uy=0.0, uz=0.0, ux=-1.0, answer=-1)

    # Particle traveling out through the Top of the torus
    run(y=0.0, z=R, x=(r - TINY), uy=0.0, uz=0.0, ux=1.0, answer=1)

    # Particle moving parallel to the torus
    run(y=0.0, z=R, x=-(0.5 + TINY), uy=0.0, uz=1, ux=0.0, answer=0)


def test_get_distance():
    def run(y, z, x, uy, uz, ux, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        result = torus_x.get_distance(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Outside Tours
    y = R
    z = 0.0
    x = r + 1
    ## Moving closer
    run(y, z, x, uy=0.0, uz=0.0, ux=-1.0, answer=1.0)
    ## Moving away
    run(y, z, x, uy=0.0, uz=0.0, ux=1.0, answer=INF)
    ## Parallel
    run(y, z, x, uy=1.0, uz=0.0, ux=0.0, answer=INF)

    # Inside Torus
    y = 0.0
    z = R
    x = r / 2
    ## Moving Up
    run(y, z, x, uy=0.0, uz=0.0, ux=1, answer=(r / 2))
    ## Moving Down
    run(y, z, x, uy=0.0, uz=0.0, ux=-1.0, answer=(3 * (r / 2)))

    # At surface, within tolerance, on the outside
    y = R
    z = 0.0
    x = r + TINY
    ## Moving away
    run(y, z, x, uy=0.0, uz=0.0, ux=1.0, answer=INF)
    ## Moving closer
    run(y, z, x, uy=0.0, uz=0.0, ux=-1.0, answer=(2 * r))
    ## Parallel
    run(y, z, x, uy=1.0, uz=0.0, ux=0.0, answer=INF)

    # At surface, within tolerance, on the inside
    y = 0.0
    z = R
    x = r - TINY
    ## Moving up
    run(y, z, x, uy=0.0, uz=0.0, ux=1.0, answer=INF)
    ## Moving down
    run(y, z, x, uy=0.0, uz=0.0, ux=-1.0, answer=(2 * r))


# =====================================================================================
# Torus-X integrated transport interface
# =====================================================================================


def test_interface_reflect():
    def run(y, z, x, uy, uz, ux, answers):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        interface.reflect(particle_container, static_surface)
        directions = np.array([particle["uy"], particle["uz"], particle["ux"]])
        assert np.allclose(directions, answers)

    run(y=R, z=0.0, x=(r + TINY), uy=0.0, uz=0.0, ux=-1.0, answers=np.array([0, 0, 1]))
    run(y=R, z=0.0, x=(r - TINY), uy=0.0, uz=0.0, ux=1.0, answers=np.array([0, 0, -1]))
    run(y=0.0, z=R, x=-(r + TINY), uy=0.0, uz=0.0, ux=1.0, answers=np.array([0, 0, -1]))
    run(y=0.0, z=R, x=-(r - TINY), uy=0.0, uz=0.0, ux=-1.0, answers=np.array([0, 0, 1]))

    root = math.sqrt(2) / 2  # 45 degree X-Y lengths on the unit circle
    d = (
        R + r
    ) * root  # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the y-axis
    run(
        y=(d + TINY),
        z=(d + TINY),
        x=0.0,
        uy=-root,
        uz=-root,
        ux=0.0,
        answers=np.array([root, root, 0]),
    )
    # Particle traveling out through the Side of the torus as above
    run(
        y=(d - TINY),
        z=(d - TINY),
        x=0.0,
        uy=root,
        uz=root,
        ux=0.0,
        answers=np.array([-root, -root, 0]),
    )


def test_interface_evaluate():

    def run_static(y, z, x, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        result = interface.evaluate(particle_container, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(y, z, x, uy, uz, ux, t, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        particle["t"] = t
        result = interface.evaluate(particle_container, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Inside
    run_static(y=1.0, z=0.0, x=0.0, answer=-0.9375)
    # Outside
    run_static(y=0.0, z=-1.0, x=5.0, answer=711.5625)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin
    t = 3.0  # Torus center y-position = -3.0 as the torus has a velocity of -1 and started in the center
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ux = 0.3  # Arbitrary
    ## Inside side
    run_moving(y=-3.0, z=1.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, answer=-0.9375)
    ## Outside side
    run_moving(y=-3.0, z=-1.0, x=5.0, uy=uy, uz=uz, ux=ux, t=t, answer=711.5625)

    # First bin, at grid
    t = 5.0  # Torus center y-position = -5.0
    uy = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    ux = -0.3  # Arbitrary
    ## Inside side
    run_moving(y=-5.0, z=1.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, answer=-0.9375)
    ## Outside side
    run_moving(y=-5.0, z=-1.0, x=5.0, uy=uy, uz=uz, ux=ux, t=t, answer=711.5625)

    # Interior bin
    t = 12.0  # Torus center y-position = -1.0 due to velocity and duration values
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ux = 0.3  # Arbitrary
    ## Inside side
    run_moving(y=-1.0, z=1.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, answer=-0.9375)
    ## Outside side
    run_moving(y=-1.0, z=-1.0, x=5.0, uy=uy, uz=uz, ux=ux, t=t, answer=711.5625)

    # Interior bin, at grid
    t = 15.0  # Surface y-position = -10.0
    uy = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    ux = -0.3  # Arbitrary
    ## Inside side
    run_moving(y=-10.0, z=1.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, answer=-0.9375)
    ## Outside side
    run_moving(y=-10.0, z=-1.0, x=5.0, uy=uy, uz=uz, ux=ux, t=t, answer=711.5625)

    # Final bin
    t = 100.0  # Surface y-position = -10.0
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ux = 0.3  # Arbitrary
    ## Inside side
    run_moving(y=-10.0, z=1.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, answer=-0.9375)
    ## Outside side
    run_moving(y=-10.0, z=-1.0, x=5.0, uy=uy, uz=uz, ux=ux, t=t, answer=711.5625)


def test_interface_get_normal_component():
    def run_static(y, z, x, uy, uz, ux, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        speed = 2.0  # Arbitrary
        result = interface.get_normal_component(
            particle_container, speed, static_surface, data
        )
        assert np.isclose(result, answer)

    def run_moving(y, z, x, uy, uz, ux, t, speed, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        particle["t"] = t
        result = interface.get_normal_component(
            particle_container, speed, moving_surface, data
        )
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Particle traveling in through the Top of the torus
    run_static(y=R, z=0.0, x=(r + TINY), uy=0.0, uz=0.0, ux=-1.0, answer=-1)
    # Particle traveling out through the Top of the torus
    run_static(y=0.0, z=R, x=(r - TINY), uy=0.0, uz=0.0, ux=1.0, answer=1)
    # Particle moving parallel to the torus
    run_static(y=0.0, z=R, x=-(0.5 + TINY), uy=0.0, uz=1, ux=0.0, answer=0)

    # =================================================================================
    # Moving
    # =================================================================================

    # Surface moving in the positive direction
    t = 8.0  # Surface y-velocity = 2.0, center of torus y position at 1
    #
    ## Positive direction
    uy = 0.4
    uz = 0.0
    ux = 0.0
    ### Faster (normal component values on the very centerline of the torus should match an y-plane)
    run_moving(
        y=(1 + R + r),
        z=0.0,
        x=0.0,
        uy=uy,
        uz=uz,
        ux=ux,
        t=t,
        speed=6.0,
        answer=0.4 / 6.0,
    )
    ### Slower (change sign)
    run_moving(
        y=(1 + R + r),
        z=0.0,
        x=0.0,
        uy=uy,
        uz=uz,
        ux=ux,
        t=t,
        speed=2.0,
        answer=-1.2 / 2.0,
    )
    ### Same speed (cancel out)
    run_moving(
        y=(1 + R + r), z=0.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, speed=5.0, answer=0.0
    )
    #
    ## Negative direction
    run_moving(
        y=(1 + R + r),
        z=0.0,
        x=0.0,
        uy=-0.4,
        uz=uz,
        ux=ux,
        t=t,
        speed=6.0,
        answer=-4.4 / 6.0,
    )
    ## Parallel
    run_moving(
        y=(1 + R + r),
        z=0.0,
        x=0.0,
        uy=-0.0,
        uz=uz,
        ux=ux,
        t=t,
        speed=6.0,
        answer=-2.0 / 6.0,
    )

    # Surface moving in the negative direction
    t = 10.0  # Surface y-velocity = -3.0, center of torus y position at 5
    #
    ## Negative direction
    uy = -0.4
    uz = 0.0
    ux = 0.0
    ### Faster
    run_moving(
        y=(5 + R + r),
        z=0.0,
        x=0.0,
        uy=uy,
        uz=uz,
        ux=ux,
        t=t,
        speed=8.0,
        answer=-0.2 / 8.0,
    )
    ### Slower (change sign)
    run_moving(
        y=(5 + R + r),
        z=0.0,
        x=0.0,
        uy=uy,
        uz=uz,
        ux=ux,
        t=t,
        speed=2.0,
        answer=2.2 / 2.0,
    )
    ### Same speed (cancel out)
    run_moving(
        y=(5 + R + r), z=0.0, x=0.0, uy=uy, uz=uz, ux=ux, t=t, speed=7.5, answer=0.0
    )
    #
    ## Positive direction
    run_moving(
        y=(5 + R + r),
        z=0.0,
        x=0.0,
        uy=0.4,
        uz=uz,
        ux=ux,
        t=t,
        speed=8.0,
        answer=6.2 / 8.0,
    )
    ## Parallel
    run_moving(
        y=(5 + R + r),
        z=0.0,
        x=0.0,
        uy=0.0,
        uz=uz,
        ux=ux,
        t=t,
        speed=8.0,
        answer=3.0 / 8.0,
    )


def test_interface_check_sense():  # Returns true if the particle is on the outside of the torus, and false if it's on the inside (particle direction and speed tiebreak)
    def run_static(y, z, x, uy, uz, ux, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(y, z, x, uy, uz, ux, t, speed, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    z = 0.0
    x = 0.0
    uy = 0.3  # Arbitrary
    uz = 0.0
    ux = 0.0
    ## Positive side
    run_static(y=3.0, z=z, x=x, uy=uy, uz=uz, ux=ux, answer=True)
    ## Negative side
    run_static(y=-4.0, z=z, x=x, uy=uy, uz=uz, ux=ux, answer=True)

    # At surface, outside
    y = R + r + TINY
    z = 0.0
    x = 0.0
    uz = 0.0
    ux = 0.0
    ## Outward direction
    run_static(y=y, z=z, x=x, uy=0.4, uz=uz, ux=ux, answer=True)
    ## Inward direction
    run_static(y=y, z=z, x=x, uy=-0.4, uz=uz, ux=ux, answer=False)

    # At surface, inside
    y = R + r - TINY
    z = 0.0
    x = 0.0
    uz = 0.0
    ux = 0.0
    ## Outward direction
    run_static(y=y, z=z, x=x, uy=0.2, uz=uz, ux=ux, answer=True)
    ## Inward direction
    run_static(y=y, z=z, x=x, uy=-0.2, uz=uz, ux=ux, answer=False)

    # =================================================================================
    # Moving: Surface moving in the positive X direction
    # =================================================================================
    t = 8.5  # Surface y-center = 2.0; surface y-velocity = 2.0

    # Not at surface
    z = 0.0
    x = 0.0
    uy = 0.3  # Arbitrary
    uz = 0.0
    ux = 0.0
    speed = 3.0  # Arbitrary
    ## Outside
    run_moving(y=13.0, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=speed, answer=True)
    ## Inside
    run_moving(y=3.0, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=speed, answer=False)

    # At surface, outside
    y = 3.5 + TINY
    z = 0.0
    x = 0.0
    ## Positive direction (same direction)
    uy = 0.4
    uz = 0.0
    ux = 0.0
    ### Faster
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(y=y, z=z, x=x, uy=-0.4, uz=uz, ux=ux, t=t, speed=6.0, answer=False)

    # At surface, inside
    y = 3.5 - TINY
    z = 0.0
    x = 0.0
    ## Positive direction (same direction)
    uy = 0.4
    uz = 0.0
    ux = 0.0
    ### Faster
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(y=y, z=z, x=x, uy=-0.4, uz=uz, ux=ux, t=t, speed=6.0, answer=False)

    # =================================================================================
    # Moving: Surface moving in the negative direction
    # =================================================================================
    t = 13.0  # Surface y-center = -4.0; surface y-velocity = -3.0

    # Not at surface
    z = 0.0
    x = 0.0
    uy = 0.3  # Arbitrary
    uz = 0.0
    ux = 0.0
    speed = 3.0  # Arbitrary
    ## Outside
    run_moving(y=0.0, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=speed, answer=True)
    ## Inside
    run_moving(y=-5.0, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=speed, answer=False)

    # At surface, outside
    y = -2.5 + TINY
    z = 0.0
    x = 0.0
    ## Positive direction (opposite direction)
    uz = 0.0
    ux = 0.0
    run_moving(y=y, z=z, x=x, uy=0.6, uz=uz, ux=ux, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    uy = -0.6
    ### Faster
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=5.0, answer=False)

    # At surface, inside
    y = -2.5 - TINY
    z = 0.0
    x = 0.0
    ## Positive direction (opposite direction)
    uz = 0.0
    ux = 0.0
    run_moving(y=y, z=z, x=x, uy=0.6, uz=uz, ux=ux, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    uy = -0.6
    ### Faster
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=5.0, answer=False)


def test_interface_get_distance():
    def run_static(y, z, x, uy, uz, ux, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(y, z, x, uy, uz, ux, t, speed, answer):
        particle["y"] = y
        particle["z"] = z
        particle["x"] = x
        particle["uy"] = uy
        particle["uz"] = uz
        particle["ux"] = ux
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    y = 6.5  # Arbitrary
    z = 0.0
    x = 0.0
    uz = 0.0
    ux = 0.0
    ## Positive direction (moving away)
    run_static(y=y, z=z, x=x, uy=0.3, uz=uz, ux=ux, answer=INF)
    ## Negative direction (moving closer)
    run_static(y=y, z=z, x=x, uy=-0.3, uz=uz, ux=ux, answer=5.0 / 0.3)
    ## Parallel
    run_static(y=y, z=z, x=x, uy=0.0, uz=1.0, ux=ux, answer=INF)

    # Negative side
    y = -5.5  # Arbitrary
    z = 0.0
    x = 0.0
    uz = 0.0
    ux = 0.0
    ## Positive direction (moving closer)
    run_static(y=y, z=z, x=x, uy=0.3, uz=uz, ux=ux, answer=4.0 / 0.3)
    ## Negative direction (moving away)
    run_static(y=y, z=z, x=x, uy=-0.3, uz=uz, ux=ux, answer=INF)
    ## Parallel
    run_static(y=y, z=z, x=x, uy=0.0, uz=1.0, ux=ux, answer=INF)

    # At surface, on the outside
    y = 1.5 + TINY
    z = 0.0
    x = 0.0
    uz = 0.0
    ux = 0.0
    ## Positive direction
    run_static(y=y, z=z, x=x, uy=0.4, uz=uz, ux=ux, answer=INF)
    ## Negative direction
    run_static(y=y, z=z, x=x, uy=-0.4, uz=uz, ux=ux, answer=1.0 / 0.4)
    ## Parallel
    run_static(y=y, z=z, x=x, uy=0.0, uz=1.0, ux=ux, answer=INF)

    # At surface, on the inside
    y = 1.5 - TINY
    z = 0.0
    x = 0.0
    uz = 0.0
    ux = 0.0
    ## Positive direction
    run_static(y=y, z=z, x=x, uy=0.4, uz=uz, ux=ux, answer=INF)
    ## Negative direction
    run_static(y=y, z=z, x=x, uy=-0.4, uz=uz, ux=ux, answer=1.0 / 0.4)
    ## Parallel
    run_static(y=y, z=z, x=x, uy=0.0, uz=1.0, ux=ux, answer=INF)

    # TODO: Numerical Risk - Add off-midplane and near-tangent moving-distance cases.
    # ============================================================================================
    # Moving (Only testing a hit on the midline of the torus in positive and negative y directions)
    # ============================================================================================
    # Bin 0: t = [ 0.0,  5.0]; surface_y_center = [-5.0,    0.0]; surface_speed = -1.0
    # Bin 1: t = [ 5.0, 10.0]; surface_y_center = [-5.0,    5.0]; surface_speed =  2.0
    # Bin 2: t = [10.0, 15.0]; surface_y_center = [-10.0,   5.0]; surface_speed = -3.0
    # Bin 3: t = [15.0,  INF]; surface_y_center = [-10.0, -10.0]; surface_speed =  0.0

    # surface_y_center (0.0   --->  -5.0)
    # surface_y_center (-5.0  --->   5.0)
    # surface_y_center (5.0,  ---> -10.0)
    # surface_y_center (-10.0 ---> -10.0)

    # This is the distance from the particle to the center of the torus, not counting the inner or outer radii
    def center_distance(y, uy, speed, bin_idx):
        # Time when the surface enters the final bin to be evaluated (0 when evaluating bin 0, and 5 when evaluating bin 1)
        t0 = 0.0 + np.sum(durations[:bin_idx])

        # Starting position of the surface at the beginning of the last bin to be evaluated (0 for bin 0, and -5 for bin 1)
        surface_y = B + np.sum(durations[:bin_idx] * velocities[:bin_idx, 1])

        # Speed of the surface in the final bin to be evaluated (-1 for bin 0)
        surface_speed = velocities[bin_idx, 1]

        return ((surface_speed * -t0) - (y - surface_y)) / (uy - surface_speed / speed)

    def outer_surface_distance(y, uy, speed, bin_idx):
        surface_speed = velocities[bin_idx, 1]
        relative_uy = uy - surface_speed / speed
        return center_distance(y, uy, speed, bin_idx) - (R + r) / abs(relative_uy)

    # Start from the beginning
    t = 0.0

    # Positive y side of the torus
    y = 2.0
    z = 0.0
    x = 0.0
    #
    ## Positive direction (moving away)
    uy = 0.4
    uz = 0.0
    ux = 0.0
    ### Surface catches up after reversing direction in bin 1
    answer = outer_surface_distance(y, uy, speed=1.0, bin_idx=1)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=1.0, answer=answer)
    ### Hit (rear-ended by the surface)
    answer = outer_surface_distance(
        y, uy, speed=0.9, bin_idx=1
    )  # Collision in bin 1 where the surface catches up to the particle
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=0.9, answer=answer)
    #
    ## Negative direction (moving closer)
    uy = -0.4
    uz = 0.0
    ux = 0.0
    ### Hit (rear-end the surface)
    answer = outer_surface_distance(
        y, uy, speed=3.0, bin_idx=0
    )  # Collision in bin 0 where the surface is running away from particle
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=3.0, answer=answer)
    ### Hit (head-on opposite directions)
    answer = outer_surface_distance(y, uy, speed=0.1, bin_idx=1)  # Collision in bin 1
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=0.1, answer=answer)

    # Negative y side of the torus
    y = -2.0
    z = 0.0
    x = 0.0
    #
    ## Negative direction (moving away)
    uy = -0.4
    uz = 0.0
    ux = 0.0
    ### Surface catches up in bin 0
    answer = outer_surface_distance(y, uy, speed=2.0, bin_idx=0)
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=2.0, answer=answer)
    ### Hit (rear-ended by the surface)
    answer = outer_surface_distance(y, uy, speed=0.4, bin_idx=0)  # Collision in bin 0
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=0.4, answer=answer)
    #
    ## Positive direction (moving closer)
    uy = 0.4
    uz = 0.0
    ux = 0.0
    ### Hit (head-on)
    answer = outer_surface_distance(y, uy, speed=0.1, bin_idx=0)  # Collision in bin 0
    run_moving(y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=0.1, answer=answer)
    ### Hit (rear-end the surface)
    y = -20.0
    answer = outer_surface_distance(
        y, uy, speed=(20.0 / 3.0), bin_idx=1
    )  # Collision in bin 1
    run_moving(
        y=y, z=z, x=x, uy=uy, uz=uz, ux=ux, t=t, speed=(20.0 / 3.0), answer=answer
    )
