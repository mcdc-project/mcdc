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
velocities[:, 0] = np.array([-1.0, 2.0, -3.0])

# Test object: static surface
static_surface = mcdc.Surface.TorusZ(A=A, B=B, C=C, R=R, r=r)

# Test object: moving surface
moving_surface = mcdc.Surface.TorusZ(A=A, B=B, C=C, R=R, r=r)
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
    torus_z,
)

# =====================================================================================
# Torus-Z core functions
# =====================================================================================


def test_evaluate():
    def run(x, y, z, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        result = torus_z.evaluate(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Inside
    run(x=1.0, y=0.0, z=0.0, answer=-0.9375)
    # Outside
    run(x=0.0, y=-1.0, z=5.0, answer=711.5625)


# Answers parameter is a numpy array of the correct [ux, uy, uz] values of the reflected particle
def test_reflect():
    def run(x, y, z, ux, uy, uz, answers):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        torus_z.reflect(particle_container, static_surface)
        directions = np.array([particle["ux"], particle["uy"], particle["uz"]])
        assert np.allclose(directions, answers)

    # Particle traveling in through the Top of the torus
    run(x=R, y=0.0, z=(r + TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0, 0, 1]))
    # Particle traveling out through the Top of the torus
    run(x=R, y=0.0, z=(r - TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0, 0, -1]))

    # Particle traveling in through the Bottom of the torus
    run(x=0.0, y=R, z=-(r + TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0, 0, -1]))
    # Particle traveling out through the Bottom of the torus
    run(x=0.0, y=R, z=-(r - TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0, 0, 1]))

    root = math.sqrt(2) / 2  # 45 degree X-Y lengths on the unit circle
    d = (
        R + r
    ) * root  # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the x-axis
    run(
        x=(d + TINY),
        y=(d + TINY),
        z=0.0,
        ux=-root,
        uy=-root,
        uz=0.0,
        answers=np.array([root, root, 0]),
    )
    # Particle traveling out through the Side of the torus as above
    run(
        x=(d - TINY),
        y=(d - TINY),
        z=0.0,
        ux=root,
        uy=root,
        uz=0.0,
        answers=np.array([-root, -root, 0]),
    )


def test_get_normal_component():
    def run(x, y, z, ux, uy, uz, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        result = torus_z.get_normal_component(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Particle traveling in through the Top of the torus
    run(x=R, y=0.0, z=(r + TINY), ux=0.0, uy=0.0, uz=-1.0, answer=-1)

    # Particle traveling out through the Top of the torus
    run(x=0.0, y=R, z=(r - TINY), ux=0.0, uy=0.0, uz=1.0, answer=1)

    # Particle moving parallel to the torus
    run(x=0.0, y=R, z=-(0.5 + TINY), ux=0.0, uy=1, uz=0.0, answer=0)


def test_get_distance():
    def run(x, y, z, ux, uy, uz, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        result = torus_z.get_distance(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Outside Tours
    x = R
    y = 0.0
    z = r + 1
    ## Moving closer
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=1.0)
    ## Moving away
    run(x, y, z, ux=0.0, uy=0.0, uz=1.0, answer=INF)
    ## Parallel
    run(x, y, z, ux=1.0, uy=0.0, uz=0.0, answer=INF)

    # Inside Torus
    x = 0.0
    y = R
    z = r / 2
    ## Moving Up
    run(x, y, z, ux=0.0, uy=0.0, uz=1, answer=(r / 2))
    ## Moving Down
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=(3 * (r / 2)))

    # At surface, within tolerance, on the outside
    x = R
    y = 0.0
    z = r + TINY
    ## Moving away
    run(x, y, z, ux=0.0, uy=0.0, uz=1.0, answer=INF)
    ## Moving closer
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=(2 * r))
    ## Parallel
    run(x, y, z, ux=1.0, uy=0.0, uz=0.0, answer=INF)

    # At surface, within tolerance, on the inside
    x = 0.0
    y = R
    z = r - TINY
    ## Moving up
    run(x, y, z, ux=0.0, uy=0.0, uz=1.0, answer=INF)
    ## Moving down
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=(2 * r))


# =====================================================================================
# Torus-Z integrated transport interface
# =====================================================================================


def test_interface_reflect():
    def run(x, y, z, ux, uy, uz, answers):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        interface.reflect(particle_container, static_surface)
        directions = np.array([particle["ux"], particle["uy"], particle["uz"]])
        assert np.allclose(directions, answers)

    run(x=R, y=0.0, z=(r + TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0, 0, 1]))
    run(x=R, y=0.0, z=(r - TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0, 0, -1]))
    run(x=0.0, y=R, z=-(r + TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0, 0, -1]))
    run(x=0.0, y=R, z=-(r - TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0, 0, 1]))

    root = math.sqrt(2) / 2  # 45 degree X-Y lengths on the unit circle
    d = (
        R + r
    ) * root  # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the x-axis
    run(
        x=(d + TINY),
        y=(d + TINY),
        z=0.0,
        ux=-root,
        uy=-root,
        uz=0.0,
        answers=np.array([root, root, 0]),
    )
    # Particle traveling out through the Side of the torus as above
    run(
        x=(d - TINY),
        y=(d - TINY),
        z=0.0,
        ux=root,
        uy=root,
        uz=0.0,
        answers=np.array([-root, -root, 0]),
    )


def test_interface_evaluate():

    def run_static(x, y, z, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        result = interface.evaluate(particle_container, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, y, z, ux, uy, uz, t, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        particle["t"] = t
        result = interface.evaluate(particle_container, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Inside
    run_static(x=1.0, y=0.0, z=0.0, answer=-0.9375)
    # Outside
    run_static(x=0.0, y=-1.0, z=5.0, answer=711.5625)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin
    t = 3.0  # Torus center x-position = -3.0 as the torus has a velocity of -1 and started in the center
    ux = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-3.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-3.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.5625)

    # First bin, at grid
    t = 5.0  # Torus center x-position = -5.0
    ux = -0.3  # Arbitrary
    uy = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    ## Inside side
    run_moving(x=-5.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-5.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.5625)

    # Interior bin
    t = 12.0  # Torus center x-position = -1.0 due to velocity and duration values
    ux = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-1.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-1.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.5625)

    # Interior bin, at grid
    t = 15.0  # Surface x-position = -10.0
    ux = -0.3  # Arbitrary
    uy = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    ## Inside side
    run_moving(x=-10.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-10.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.5625)

    # Final bin
    t = 100.0  # Surface x-position = -10.0
    ux = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-10.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-10.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.5625)


def test_interface_get_normal_component():
    def run_static(x, y, z, ux, uy, uz, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        speed = 2.0  # Arbitrary
        result = interface.get_normal_component(
            particle_container, speed, static_surface, data
        )
        assert np.isclose(result, answer)

    def run_moving(x, y, z, ux, uy, uz, t, speed, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        particle["t"] = t
        result = interface.get_normal_component(
            particle_container, speed, moving_surface, data
        )
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Particle traveling in through the Top of the torus
    run_static(x=R, y=0.0, z=(r + TINY), ux=0.0, uy=0.0, uz=-1.0, answer=-1)
    # Particle traveling out through the Top of the torus
    run_static(x=0.0, y=R, z=(r - TINY), ux=0.0, uy=0.0, uz=1.0, answer=1)
    # Particle moving parallel to the torus
    run_static(x=0.0, y=R, z=-(0.5 + TINY), ux=0.0, uy=1, uz=0.0, answer=0)

    # =================================================================================
    # Moving
    # =================================================================================

    # Surface moving in the positive direction
    t = 8.0  # Surface x-velocity = 2.0, center of torus x position at 1
    #
    ## Positive direction
    ux = 0.4
    uy = 0.0
    uz = 0.0
    ### Faster (normal component values on the very centerline of the torus should match an x-plane)
    run_moving(
        x=(1 + R + r),
        y=0.0,
        z=0.0,
        ux=ux,
        uy=uy,
        uz=uz,
        t=t,
        speed=6.0,
        answer=0.4 / 6.0,
    )
    ### Slower (change sign)
    run_moving(
        x=(1 + R + r),
        y=0.0,
        z=0.0,
        ux=ux,
        uy=uy,
        uz=uz,
        t=t,
        speed=2.0,
        answer=-1.2 / 2.0,
    )
    ### Same speed (cancel out)
    run_moving(
        x=(1 + R + r), y=0.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, speed=5.0, answer=0.0
    )
    #
    ## Negative direction
    run_moving(
        x=(1 + R + r),
        y=0.0,
        z=0.0,
        ux=-0.4,
        uy=uy,
        uz=uz,
        t=t,
        speed=6.0,
        answer=-4.4 / 6.0,
    )
    ## Parallel
    run_moving(
        x=(1 + R + r),
        y=0.0,
        z=0.0,
        ux=-0.0,
        uy=uy,
        uz=uz,
        t=t,
        speed=6.0,
        answer=-2.0 / 6.0,
    )

    # Surface moving in the negative direction
    t = 10.0  # Surface x-velocity = -3.0, center of torus x position at 5
    #
    ## Negative direction
    ux = -0.4
    uy = 0.0
    uz = 0.0
    ### Faster
    run_moving(
        x=(5 + R + r),
        y=0.0,
        z=0.0,
        ux=ux,
        uy=uy,
        uz=uz,
        t=t,
        speed=8.0,
        answer=-0.2 / 8.0,
    )
    ### Slower (change sign)
    run_moving(
        x=(5 + R + r),
        y=0.0,
        z=0.0,
        ux=ux,
        uy=uy,
        uz=uz,
        t=t,
        speed=2.0,
        answer=2.2 / 2.0,
    )
    ### Same speed (cancel out)
    run_moving(
        x=(5 + R + r), y=0.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, speed=7.5, answer=0.0
    )
    #
    ## Positive direction
    run_moving(
        x=(5 + R + r),
        y=0.0,
        z=0.0,
        ux=0.4,
        uy=uy,
        uz=uz,
        t=t,
        speed=8.0,
        answer=6.2 / 8.0,
    )
    ## Parallel
    run_moving(
        x=(5 + R + r),
        y=0.0,
        z=0.0,
        ux=0.0,
        uy=uy,
        uz=uz,
        t=t,
        speed=8.0,
        answer=3.0 / 8.0,
    )


def test_interface_check_sense():  # Returns true if the particle is on the outside of the torus, and false if it's on the inside (particle direction and speed tiebreak)
    def run_static(x, y, z, ux, uy, uz, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, y, z, ux, uy, uz, t, speed, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    y = 0.0
    z = 0.0
    ux = 0.3  # Arbitrary
    uy = 0.0
    uz = 0.0
    ## Positive side
    run_static(x=3.0, y=y, z=z, ux=ux, uy=uy, uz=uz, answer=True)
    ## Negative side
    run_static(x=-4.0, y=y, z=z, ux=ux, uy=uy, uz=uz, answer=True)

    # At surface, outside
    x = R + r + TINY
    y = 0.0
    z = 0.0
    uy = 0.0
    uz = 0.0
    ## Outward direction
    run_static(x=x, y=y, z=z, ux=0.4, uy=uy, uz=uz, answer=True)
    ## Inward direction
    run_static(x=x, y=y, z=z, ux=-0.4, uy=uy, uz=uz, answer=False)

    # At surface, inside
    x = R + r - TINY
    y = 0.0
    z = 0.0
    uy = 0.0
    uz = 0.0
    ## Outward direction
    run_static(x=x, y=y, z=z, ux=0.2, uy=uy, uz=uz, answer=True)
    ## Inward direction
    run_static(x=x, y=y, z=z, ux=-0.2, uy=uy, uz=uz, answer=False)

    # =================================================================================
    # Moving: Surface moving in the positive X direction
    # =================================================================================
    t = 8.5  # Surface x-center = 2.0; surface x-velocity = 2.0

    # Not at surface
    y = 0.0
    z = 0.0
    ux = 0.3  # Arbitrary
    uy = 0.0
    uz = 0.0
    speed = 3.0  # Arbitrary
    ## Outside
    run_moving(x=13.0, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=speed, answer=True)
    ## Inside
    run_moving(x=3.0, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=speed, answer=False)

    # At surface, outside
    x = 3.5 + TINY
    y = 0.0
    z = 0.0
    ## Positive direction (same direction)
    ux = 0.4
    uy = 0.0
    uz = 0.0
    ### Faster
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(x=x, y=y, z=z, ux=-0.4, uy=uy, uz=uz, t=t, speed=6.0, answer=False)

    # At surface, inside
    x = 3.5 - TINY
    y = 0.0
    z = 0.0
    ## Positive direction (same direction)
    ux = 0.4
    uy = 0.0
    uz = 0.0
    ### Faster
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(x=x, y=y, z=z, ux=-0.4, uy=uy, uz=uz, t=t, speed=6.0, answer=False)

    # =================================================================================
    # Moving: Surface moving in the negative direction
    # =================================================================================
    t = 13.0  # Surface x-center = -4.0; surface x-velocity = -3.0

    # Not at surface
    y = 0.0
    z = 0.0
    ux = 0.3  # Arbitrary
    uy = 0.0
    uz = 0.0
    speed = 3.0  # Arbitrary
    ## Outside
    run_moving(x=0.0, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=speed, answer=True)
    ## Inside
    run_moving(x=-5.0, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=speed, answer=False)

    # At surface, outside
    x = -2.5 + TINY
    y = 0.0
    z = 0.0
    ## Positive direction (opposite direction)
    uy = 0.0
    uz = 0.0
    run_moving(x=x, y=y, z=z, ux=0.6, uy=uy, uz=uz, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    ux = -0.6
    ### Faster
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=5.0, answer=False)

    # At surface, inside
    x = -2.5 - TINY
    y = 0.0
    z = 0.0
    ## Positive direction (opposite direction)
    uy = 0.0
    uz = 0.0
    run_moving(x=x, y=y, z=z, ux=0.6, uy=uy, uz=uz, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    ux = -0.6
    ### Faster
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=5.0, answer=False)


def test_interface_get_distance():
    def run_static(x, y, z, ux, uy, uz, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, y, z, ux, uy, uz, t, speed, answer):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    x = 6.5  # Arbitrary
    y = 0.0
    z = 0.0
    uy = 0.0
    uz = 0.0
    ## Positive direction (moving away)
    run_static(x=x, y=y, z=z, ux=0.3, uy=uy, uz=uz, answer=INF)
    ## Negative direction (moving closer)
    run_static(x=x, y=y, z=z, ux=-0.3, uy=uy, uz=uz, answer=5.0 / 0.3)
    ## Parallel
    run_static(x=x, y=y, z=z, ux=0.0, uy=1.0, uz=uz, answer=INF)

    # Negative side
    x = -5.5  # Arbitrary
    y = 0.0
    z = 0.0
    uy = 0.0
    uz = 0.0
    ## Positive direction (moving closer)
    run_static(x=x, y=y, z=z, ux=0.3, uy=uy, uz=uz, answer=4.0 / 0.3)
    ## Negative direction (moving away)
    run_static(x=x, y=y, z=z, ux=-0.3, uy=uy, uz=uz, answer=INF)
    ## Parallel
    run_static(x=x, y=y, z=z, ux=0.0, uy=1.0, uz=uz, answer=INF)

    # At surface, on the outside
    x = 1.5 + TINY
    y = 0.0
    z = 0.0
    uy = 0.0
    uz = 0.0
    ## Positive direction
    run_static(x=x, y=y, z=z, ux=0.4, uy=uy, uz=uz, answer=INF)
    ## Negative direction
    run_static(x=x, y=y, z=z, ux=-0.4, uy=uy, uz=uz, answer=1.0 / 0.4)
    ## Parallel
    run_static(x=x, y=y, z=z, ux=0.0, uy=1.0, uz=uz, answer=INF)

    # At surface, on the inside
    x = 1.5 - TINY
    y = 0.0
    z = 0.0
    uy = 0.0
    uz = 0.0
    ## Positive direction
    run_static(x=x, y=y, z=z, ux=0.4, uy=uy, uz=uz, answer=INF)
    ## Negative direction
    run_static(x=x, y=y, z=z, ux=-0.4, uy=uy, uz=uz, answer=1.0 / 0.4)
    ## Parallel
    run_static(x=x, y=y, z=z, ux=0.0, uy=1.0, uz=uz, answer=INF)

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
    y = 0.0
    z = 0.0
    #
    ## Positive direction (moving away)
    ux = 0.4
    uy = 0.0
    uz = 0.0
    ### Surface catches up after reversing direction in bin 1
    answer = outer_surface_distance(x, ux, speed=1.0, bin_idx=1)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=1.0, answer=answer)
    ### Hit (rear-ended by the surface)
    answer = outer_surface_distance(
        x, ux, speed=0.9, bin_idx=1
    )  # Collision in bin 1 where the surface catches up to the particle
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=0.9, answer=answer)
    #
    ## Negative direction (moving closer)
    ux = -0.4
    uy = 0.0
    uz = 0.0
    ### Hit (rear-end the surface)
    answer = outer_surface_distance(
        x, ux, speed=3.0, bin_idx=0
    )  # Collision in bin 0 where the surface is running away from particle
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=3.0, answer=answer)
    ### Hit (head-on opposite directions)
    answer = outer_surface_distance(x, ux, speed=0.1, bin_idx=1)  # Collision in bin 1
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=0.1, answer=answer)

    # Negative x side of the torus
    x = -2.0
    y = 0.0
    z = 0.0
    #
    ## Negative direction (moving away)
    ux = -0.4
    uy = 0.0
    uz = 0.0
    ### Surface catches up in bin 0
    answer = outer_surface_distance(x, ux, speed=2.0, bin_idx=0)
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=2.0, answer=answer)
    ### Hit (rear-ended by the surface)
    answer = outer_surface_distance(x, ux, speed=0.4, bin_idx=0)  # Collision in bin 0
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=0.4, answer=answer)
    #
    ## Positive direction (moving closer)
    ux = 0.4
    uy = 0.0
    uz = 0.0
    ### Hit (head-on)
    answer = outer_surface_distance(x, ux, speed=0.1, bin_idx=0)  # Collision in bin 0
    run_moving(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=0.1, answer=answer)
    ### Hit (rear-end the surface)
    x = -20.0
    answer = outer_surface_distance(
        x, ux, speed=(20.0 / 3.0), bin_idx=1
    )  # Collision in bin 1
    run_moving(
        x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, t=t, speed=(20.0 / 3.0), answer=answer
    )
