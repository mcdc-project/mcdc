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
R = 1
r = 0.5
durations = np.array([5.0, 5.0, 5.0]) # The X-plane starts at x=10 and has a velocity of -1 for 5 seconds, then 2 for 5 seconds, then -3 for 5 seconds
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
TINY = COINCIDENCE_TOLERANCE * 0.8  # Tiny value within coincidence tolerance

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
    run(x=0.0, y=-1.0, z=5.0, answer=711.6)


# Answers parameter is a numpy array of the correct [ux, uy, uz] values of the reflected particle
def test_reflect():
    def run(x, y, z, ux, uy, uz, answers):
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z
        particle["ux"] = ux
        particle["uy"] = uy
        particle["uz"] = uz
        directions = np.array([particle["ux"], particle["uy"], particle["uz"]])
        torus_z.reflect(particle_container, static_surface)
        assert np.allclose(directions, answers)

    # Particle traveling in through the Top of the torus
    run(x=R, y=0.0, z=(r+TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0,0,1]))
    # Particle traveling out through the Top of the torus
    run(x=R, y=0.0, z=(r-TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0,0,-1]))

    # Particle traveling in through the Bottom of the torus
    run(x=0.0, y=R, z=-(r+TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0,0,1]))
    # Particle traveling out through the Bottom of the torus
    run(x=0.0, y=R, z=-(r-TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0,0,-1]))

    root = (math.sqrt(2) / 2) # 45 degree X-Y lengths on the unit circle
    d = (R + r) * root # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the x-axis
    run(x=(d+TINY), y=(d+TINY), z=0.0, ux=-root, uy=-root, uz=0.0, answers=np.array([root,root,0]))
    # Particle traveling out through the Side of the torus as above
    run(x=(d-TINY), y=(d-TINY), z=0.0, ux=root, uy=root, uz=0.0, answers=np.array([-root,-root,0]))


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
    run(x=R, y=0.0, z=(r+TINY), ux=0.0, uy=0.0, uz=-1.0, answer=-1)
    
    # Particle traveling out through the Top of the torus
    run(x=0.0, y=R, z=(r-TINY), ux=0.0, uy=0.0, uz=1.0, answer=1)

    # Particle moving parallel to the torus
    run(x=0.0, y=R, z=-(0.5+TINY), ux=0.0, uy=1, uz=0.0, answer=0)


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

    # Outisde Tours
    x = R
    y = 0.0
    z = (r + 1)
    ## Moving closer
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=0.0)
    ## Moving away
    run(x, y, z, ux=0.0, uy=0.0, uz=1.0, answer=INF)
    ## Parallel
    run(x, y, z, ux=1.0, uy=0.0, uz=0.0, answer=INF)

    # Inside Torus
    x = 0.0
    y = R
    z = (r / 2)
    ## Moving Up
    run(x, y, z, ux=0.0, uy=0.0, uz=1, answer=(r/2))
    ## Moving Down
    run(x, y, z, ux=0.0, uy=0.0, uz=0.0, answer=(3 * (r/2)))

    # At surface, within tolerance, on the outside
    x = R
    y = 0.0
    z = r + TINY
    ## Moving away
    run(x, y, z, ux=0.0, uy=0.0, uz=1.0, answer=INF)
    ## Moving closer
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=INF)
    ## Parallel
    run(x, y, z, ux=1.0, uy=0.0, uz=0.0, answer=INF)

    # At surface, within tolerance, on the inside
    x = 0.0
    y = R
    z = r - TINY
    ## Moving up
    run(x, y, z, ux=0.0, uy=0.0, uz=1.0, answer=INF)
    ## Moving down
    run(x, y, z, ux=0.0, uy=0.0, uz=-1.0, answer=INF)


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
        directions = np.array([particle["ux"], particle["uy"], particle["uz"]])
        interface.reflect(particle_container, static_surface)
        assert np.allclose(directions, answers)

    # Particle traveling in through the Top of the torus
    run(x=R, y=0.0, z=(r+TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0,0,1]))
    # Particle traveling out through the Top of the torus
    run(x=R, y=0.0, z=(r-TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0,0,-1]))

    # Particle traveling in through the Bottom of the torus
    run(x=0.0, y=R, z=-(r+TINY), ux=0.0, uy=0.0, uz=1.0, answers=np.array([0,0,1]))
    # Particle traveling out through the Bottom of the torus
    run(x=0.0, y=R, z=-(r-TINY), ux=0.0, uy=0.0, uz=-1.0, answers=np.array([0,0,-1]))

    root = (math.sqrt(2) / 2) # 45 degree X-Y lengths on the unit circle
    d = (R + r) * root # X-Y values for a particle at 45 degrees on a torus of given dimensions

    # Particle traveling in head on through the Side of the torus at the 45 degrees from the x-axis
    run(x=(d+TINY), y=(d+TINY), z=0.0, ux=-root, uy=-root, uz=0.0, answers=np.array([root,root,0]))
    # Particle traveling out through the Side of the torus as above
    run(x=(d-TINY), y=(d-TINY), z=0.0, ux=root, uy=root, uz=0.0, answers=np.array([-root,-root,0]))


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
    run_static(x=0.0, y=-1.0, z=5.0, answer=711.6)

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
    run_moving(x=-3.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.6)

    # First bin, at grid
    t = 5.0  # Torus center x-position = -5.0
    ux = -0.3  # Arbitrary
    uy = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    ## Inside side
    run_moving(x=-5.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-5.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.6)

    # Interior bin
    t = 12.0  # Torus center x-position = -1.0 due to velocity and duration values
    ux = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-1.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-1.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.6)

    # Interior bin, at grid
    t = 15.0  # Surface x-position = -10.0
    ux = -0.3  # Arbitrary
    uy = -0.3  # Arbitrary
    uz = -0.3  # Arbitrary
    ## Inside side
    run_moving(x=-10.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-10.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.6)

    # Final bin
    t = 100.0  # Surface x-position = -10.0
    ux = 0.3  # Arbitrary
    uy = 0.3  # Arbitrary
    uz = 0.3  # Arbitrary
    ## Inside side
    run_moving(x=-10.0, y=1.0, z=0.0, ux=ux, uy=uy, uz=uz, t=t, answer=-0.9375)
    ## Outside side
    run_moving(x=-10.0, y=-1.0, z=5.0, ux=ux, uy=uy, uz=uz, t=t, answer=711.6)


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
    run_static(x=R, y=0.0, z=(r+TINY), ux=0.0, uy=0.0, uz=-1.0, answer=-1)
    # Particle traveling out through the Top of the torus
    run_static(x=0.0, y=R, z=(r-TINY), ux=0.0, uy=0.0, uz=1.0, answer=1)
    # Particle moving parallel to the torus
    run_static(x=0.0, y=R, z=-(0.5+TINY), ux=0.0, uy=1, uz=0.0, answer=0)

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
    run_moving(x=(1+R+r), y=0.0, z=0.0, ux=ux, uy=0.0, uz=0.0, t=t, speed=6.0, answer=0.4 / 6.0)
    ### Slower (change sign)
    run_moving(x=(1+R+r), y=0.0, z=0.0, ux=ux, uy=0.0, uz=0.0, t=t, speed=2.0, answer=-1.2 / 2.0)
    ### Same speed (cancel out)
    run_moving(x=(1+R+r), y=0.0, z=0.0, ux=ux, uy=0.0, uz=0.0, t=t, speed=5.0, answer=0.0)
    #
    ## Negative direction
    run_moving(x=(1+R+r), y=0.0, z=0.0, ux=-0.4, uy=0.0, uz=0.0, t=t, speed=6.0, answer=-4.4 / 6.0)
    ## Parallel
    run_moving(x=(1+R+r), y=0.0, z=0.0, ux=-0.0, uy=0.0, uz=0.0, t=t, speed=6.0, answer=-2.0 / 6.0)

    # Surface moving in the negative direction
    t = 10.0  # Surface x-velocity = -3.0, center of torus x position at 5
    #
    ## Negative direction
    ux = -0.4
    ### Faster
    run_moving(x=(5+R+r), y=0.0, z=0.0, ux=ux, uy=0.0, uz=0.0, t=t, speed=8.0, answer=-0.2 / 8.0)
    ### Slower (change sign)
    run_moving(x=(5+R+r), y=0.0, z=0.0, ux=ux, uy=0.0, uz=0.0, t=t, speed=2.0, answer=2.2 / 2.0)
    ### Same speed (cancel out)
    run_moving(x=(5+R+r), y=0.0, z=0.0, ux=ux, uy=0.0, uz=0.0, t=t, speed=7.5, answer=0.0)
    #
    ## Positive direction
    run_moving(x=(5+R+r), y=0.0, z=0.0, ux=0.4, uy=0.0, uz=0.0, t=t, speed=8.0, answer=6.2 / 8.0)
    ## Parallel
    run_moving(x=(5+R+r), y=0.0, z=0.0, ux=0.0, uy=0.0, uz=0.0, t=t, speed=8.0, answer=3.0 / 8.0)


def test_interface_check_sense():
    def run_static(x, ux, answer):
        particle["x"] = x
        particle["ux"] = ux
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, ux, t, speed, answer):
        particle["x"] = x
        particle["ux"] = ux
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    ux = 0.3  # Arbitrary
    ## Positive side
    run_static(x=12.0, ux=ux, answer=True)
    ## Negative side
    run_static(x=4.0, ux=ux, answer=False)

    # At surface, positive side
    x = 10.0 + TINY
    ## Positive direction
    run_static(x, ux=0.4, answer=True)
    ## Negative direction
    run_static(x, ux=-0.4, answer=False)

    # At surface, negative side
    x = 10.0 - TINY
    ## Positive direction
    run_static(x, ux=0.2, answer=True)
    ## Negative direction
    run_static(x, ux=-0.2, answer=False)

    # =================================================================================
    # Moving: Surface moving in the positive direction
    # =================================================================================
    t = 8.5  # Surface x-position = 12.0; surface x-velocity = 2.0

    # Not at surface
    ux = 0.3  # Arbitrary
    speed = 3.0  # Arbitrary
    ## Positive side
    run_moving(x=13.0, ux=ux, t=t, speed=speed, answer=True)
    ## Negative side
    run_moving(x=5.0, ux=ux, t=t, speed=speed, answer=False)

    # At surface, positive side
    x = 12.0 + TINY
    #
    ## Positive direction (same direction)
    ux = 0.4
    ### Faster
    run_moving(x, ux, t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(x, ux, t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(x, ux, t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(x, ux=-0.4, t=t, speed=6.0, answer=False)

    # At surface, negative side
    x = 12.0 - TINY
    ## Positive direction (same direction)
    ux = 0.4
    ### Faster
    run_moving(x, ux, t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(x, ux, t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(x, ux, t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(x, ux=-0.4, t=t, speed=6.0, answer=False)

    # =================================================================================
    # Moving: Surface moving in the negative direction
    # =================================================================================
    t = 13.0  # Surface x-position = 6.0; surface x-velocity = -3.0

    # Not at surface
    ux = 0.3  # Arbitrary
    speed = 3.0  # Arbitrary
    ## Positive side
    run_moving(x=13.0, ux=ux, t=t, speed=speed, answer=True)
    ## Negative side
    run_moving(x=5.0, ux=ux, t=t, speed=speed, answer=False)

    # At surface, positive side
    x = 6.0 + TINY
    #
    ## Positive direction (opposite direction)
    run_moving(x, ux=0.6, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    ux = -0.6
    ### Faster
    run_moving(x, ux, t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(x, ux, t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(x, ux, t, speed=5.0, answer=False)

    # At surface, negative side
    x = 6.0 - TINY
    #
    ## Positive direction (opposite direction)
    run_moving(x, ux=0.6, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    ux = -0.6
    ### Faster
    run_moving(x, ux, t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(x, ux, t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(x, ux, t, speed=5.0, answer=False)


def test_interface_get_distance():
    def run_static(x, ux, answer):
        particle["x"] = x
        particle["ux"] = ux
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(x, ux, t, speed, answer):
        particle["x"] = x
        particle["ux"] = ux
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    x = 12.0
    ## Positive direction (moving away)
    run_static(x, ux=0.3, answer=INF)
    ## Negative direction (moving closer)
    run_static(x, ux=-0.4, answer=5.0)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # Negative side
    x = 6.0
    ## Positive direction (moving closer)
    run_static(x, ux=0.4, answer=10.0)
    ## Negative direction (moving away)
    run_static(x, ux=-0.3, answer=INF)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # At surface, on the positive side
    x = 10.0 + TINY
    ## Positive direction
    run_static(x, ux=0.4, answer=INF)
    ## Negative direction
    run_static(x, ux=-0.4, answer=INF)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # At surface, on the negative side
    x = 10.0 - TINY
    ## Positive direction
    run_static(x, ux=0.4, answer=INF)
    ## Negative direction
    run_static(x, ux=-0.4, answer=INF)
    ## Parallel
    run_static(x, ux=0.0, answer=INF)

    # =================================================================================
    # Moving
    # =================================================================================
    # Bin 0: t = [ 0.0,  5.0]; surface_x = [5.0, 10.0]; surface_speed = -1.0
    # Bin 1: t = [ 5.0, 10.0]; surface_x = [5.0, 15.0]; surface_speed =  2.0
    # Bin 2: t = [10.0, 15.0]; surface_x = [0.0, 15.0]; surface_speed = -3.0
    # Bin 3: t = [15.0,  INF]; surface_x = [0.0,  0.0]; surface_speed =  0.0

    def distance(x, ux, speed, bin_idx):
        t0 = 0.0 + np.sum(durations[:bin_idx])
        surface_x = X + np.sum(durations[:bin_idx] * velocities[:bin_idx, 0])
        surface_speed = velocities[bin_idx, 0]
        return ((surface_speed * -t0) - (x - surface_x)) / (ux - surface_speed / speed)

    # Start from the beginning
    t = 0.0

    # Positive side
    x = 11.0
    #
    ## Positive direction (moving away)
    ux = 0.4
    ### No hit
    run_moving(x, ux, t, speed=1.0, answer=INF)
    ### Hit (rear-ended by the surface)
    answer = distance(x, ux, speed=0.9, bin_idx=1)
    run_moving(x, ux, t, speed=0.9, answer=answer)
    #
    ## Negative direction (moving closer)
    ux = -0.4
    ### Hit (rear-end the surface)
    answer = distance(x, ux, speed=3.0, bin_idx=0)
    run_moving(x, ux, t, speed=3.0, answer=answer)
    ### Hit (head-on)
    answer = distance(x, ux, speed=0.1, bin_idx=1)
    run_moving(x, ux, t, speed=0.1, answer=answer)

    # Negative side
    x = 7.0
    #
    ## Negative direction (moving away)
    ux = -0.4
    ### No hit
    run_moving(x, ux, t, speed=2.0, answer=INF)
    ### Hit (rear-ended by the surface)
    answer = distance(x, ux, speed=0.4, bin_idx=0)
    run_moving(x, ux, t, speed=0.4, answer=answer)
    #
    ## Positive direction (moving closer)
    ux = 0.4
    ### Hit (head-on)
    answer = distance(x, ux, speed=0.1, bin_idx=0)
    run_moving(x, ux, t, speed=0.1, answer=answer)
    ### Hit (rear-end the surface)
    x = -10.0
    answer = distance(x, ux, speed=20.0 / 3.0, bin_idx=1)
    run_moving(x, ux, t, speed=20.0 / 3.0, answer=answer)
