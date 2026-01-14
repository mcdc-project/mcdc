import importlib
import mcdc
import numpy as np

mcdc = importlib.reload(mcdc)

####

import mcdc.numba_types as type_

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)
from mcdc.main import preparation
from mcdc.transport.geometry.surface import (
    interface,
    plane_y,
)


# ======================================================================================
# Setup
# ======================================================================================

# Reference surface description
Y = 10.0
durations = np.array([5.0, 5.0, 5.0])
velocities = np.zeros((3, 3))
velocities[:, 1] = np.array([-1.0, 2.0, -3.0])

# Test object: static surface
static_surface = mcdc.Surface.PlaneY(y=Y)

# Test object: moving surface
moving_surface = mcdc.Surface.PlaneY(y=Y)
moving_surface.move(velocities, durations)

# Create the dummy simulation structure and data
structure_container, data = preparation()
structure = structure_container[0]

# Get the "compiled" test objects
static_surface = structure["surfaces"][0]
moving_surface = structure["surfaces"][1]

# Particle object for testing
particle_container = np.zeros(1, type_.particle_data)
particle = particle_container[0]

# Miscellanies
TINY = COINCIDENCE_TOLERANCE * 0.8  # Tiny value within coincidence tolerance

# =====================================================================================
# Plane-Y core functions
# =====================================================================================


def test_evaluate():
    def run(y, answer):
        particle["y"] = y
        result = plane_y.evaluate(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive side
    run(y=13.0, answer=3.0)
    # Negative side
    run(y=6.0, answer=-4.0)


def test_reflect():
    def run(uy, answer):
        particle["uy"] = uy
        plane_y.reflect(particle_container, static_surface)
        assert np.isclose(particle["uy"], answer)

    # From positive direction
    run(uy=0.2, answer=-0.2)
    # From negative direction
    run(uy=-0.1, answer=0.1)


def test_get_normal_component():
    def run(uy, answer):
        particle["uy"] = uy
        result = plane_y.get_normal_component(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive direction
    run(uy=0.4, answer=0.4)
    # Negative direction
    run(uy=-0.2, answer=-0.2)
    # Parallel
    run(uy=0.0, answer=0.0)


def test_get_distance():
    def run(y, uy, answer):
        particle["y"] = y
        particle["uy"] = uy
        result = plane_y.get_distance(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive side
    y = 12.0
    ## Moving closer
    run(y, uy=-0.4, answer=5.0)
    ## Moving away
    run(y, uy=0.3, answer=INF)
    ## Parallel
    run(y, uy=0.0, answer=INF)

    # Negative side
    y = 6.0
    ## Moving closer
    run(y, uy=0.4, answer=10.0)
    ## Moving away
    run(y, uy=-0.3, answer=INF)
    ## Parallel
    run(y, uy=0.0, answer=INF)

    # At surface, within tolerance, on the positive side
    y = 10.0 + TINY
    ## Moving away
    run(y, uy=0.4, answer=INF)
    ## Moving closer
    run(y, uy=-0.4, answer=INF)
    ## Parallel
    run(y, uy=0.0, answer=INF)

    # At surface, within tolerance, on the negative side
    y = 10.0 - TINY
    ## Moving away
    run(y, uy=-0.4, answer=INF)
    ## Moving closer
    run(y, uy=0.4, answer=INF)
    ## Parallel
    run(y, uy=0.0, answer=INF)


# =====================================================================================
# Plane-y integrated transport interface
# =====================================================================================


def test_interface_reflect():
    def run(uy, answer):
        particle["uy"] = uy
        interface.reflect(particle_container, static_surface)
        assert np.isclose(particle["uy"], answer)

    # From positive direction
    run(uy=0.2, answer=-0.2)
    # From negative direction
    run(uy=-0.1, answer=0.1)


def test_interface_evaluate():
    def run_static(y, answer):
        particle["y"] = y
        result = interface.evaluate(particle_container, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(y, uy, t, answer):
        particle["y"] = y
        particle["uy"] = uy
        particle["t"] = t
        result = interface.evaluate(particle_container, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    run_static(y=13.0, answer=3.0)
    # Negative side
    run_static(y=6.0, answer=-4.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin
    t = 3.0  # Surface y-position = 7.0
    uy = 0.3  # Arbitrary
    ## Positive side
    run_moving(y=10.0, uy=uy, t=t, answer=3.0)
    ## Negative side
    run_moving(y=1.0, uy=uy, t=t, answer=-6.0)

    # First bin, at grid
    t = 5.0  # Surface y-position = 5.0
    uy = -0.3  # Arbitrary
    ## Positive side
    run_moving(y=10.0, uy=uy, t=t, answer=5.0)
    ## Negative side
    run_moving(y=1.0, uy=uy, t=t, answer=-4.0)

    # Interior bin
    t = 12.0  # Surface y-position = 9.0
    uy = 0.3  # Arbitrary
    ## Positive side
    run_moving(y=10.0, uy=uy, t=t, answer=1.0)
    ## Negative side
    run_moving(y=1.0, uy=uy, t=t, answer=-8.0)

    # Interior bin, at grid
    t = 15.0  # Surface y-position = 0.0
    uy = -0.3  # Arbitrary
    ## Positive side
    run_moving(y=10.0, uy=uy, t=t, answer=10.0)
    ## Negative side
    run_moving(y=-5.0, uy=uy, t=t, answer=-5.0)

    # Final bin
    t = 100.0  # Surface y-position = 0.0
    uy = 0.3  # Arbitrary
    ## Positive side
    run_moving(y=10.0, uy=uy, t=t, answer=10.0)
    ## Negative side
    run_moving(y=-5.0, uy=uy, t=t, answer=-5.0)


def test_interface_get_normal_component():
    def run_static(uy, answer):
        particle["uy"] = uy
        speed = 2.0  # Arbitrary
        result = interface.get_normal_component(
            particle_container, speed, static_surface, data
        )
        assert np.isclose(result, answer)

    def run_moving(uy, t, speed, answer):
        particle["uy"] = uy
        particle["t"] = t
        result = interface.get_normal_component(
            particle_container, speed, moving_surface, data
        )
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive direction
    run_static(uy=0.4, answer=0.4)
    # Negative direction
    run_static(uy=-0.2, answer=-0.2)
    # Parallel
    run_static(uy=0.0, answer=0.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # Surface moving in the positive direction
    t = 8.5  # Surface y-velocity = 2.0
    #
    ## Positive direction
    uy = 0.4
    ### Faster
    run_moving(uy, t, speed=6.0, answer=0.4 / 6.0)
    ### Slower (change sign)
    run_moving(uy, t, speed=2.0, answer=-1.2 / 2.0)
    ### Same speed (cancel out)
    run_moving(uy, t, speed=5.0, answer=0.0)
    #
    ## Negative direction
    run_moving(uy=-0.4, t=t, speed=6.0, answer=-4.4 / 6.0)
    ## Parallel
    run_moving(uy=0.0, t=t, speed=6.0, answer=-2.0 / 6.0)

    # Surface moving in the negative direction
    t = 10.0  # Surface y-velocity = -3.0
    #
    ## Negative direction
    uy = -0.4
    ### Faster
    run_moving(uy, t, speed=8.0, answer=-0.2 / 8.0)
    ### Slower (change sign)
    run_moving(uy, t, speed=2.0, answer=2.2 / 2.0)
    ### Same speed (cancel out)
    run_moving(uy, t, speed=7.5, answer=0.0)
    #
    ## Positive direction
    run_moving(uy=0.4, t=t, speed=8.0, answer=6.2 / 8.0)
    ## Parallel
    run_moving(uy=0.0, t=t, speed=8.0, answer=3.0 / 8.0)


def test_interface_check_sense():
    def run_static(y, uy, answer):
        particle["y"] = y
        particle["uy"] = uy
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(y, uy, t, speed, answer):
        particle["y"] = y
        particle["uy"] = uy
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    uy = 0.3  # Arbitrary
    ## Positive side
    run_static(y=12.0, uy=uy, answer=True)
    ## Negative side
    run_static(y=4.0, uy=uy, answer=False)

    # At surface, positive side
    y = 10.0 + TINY
    ## Positive direction
    run_static(y, uy=0.4, answer=True)
    ## Negative direction
    run_static(y, uy=-0.4, answer=False)

    # At surface, negative side
    y = 10.0 - TINY
    ## Positive direction
    run_static(y, uy=0.2, answer=True)
    ## Negative direction
    run_static(y, uy=-0.2, answer=False)

    # =================================================================================
    # Moving: Surface moving in the positive direction
    # =================================================================================
    t = 8.5  # Surface y-position = 12.0; surface y-velocity = 2.0

    # Not at surface
    uy = 0.3  # Arbitrary
    speed = 3.0  # Arbitrary
    ## Positive side
    run_moving(y=13.0, uy=uy, t=t, speed=speed, answer=True)
    ## Negative side
    run_moving(y=5.0, uy=uy, t=t, speed=speed, answer=False)

    # At surface, positive side
    y = 12.0 + TINY
    #
    ## Positive direction (same direction)
    uy = 0.4
    ### Faster
    run_moving(y, uy, t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(y, uy, t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(y, uy, t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(y, uy=-0.4, t=t, speed=6.0, answer=False)

    # At surface, negative side
    y = 12.0 - TINY
    ## Positive direction (same direction)
    uy = 0.4
    ### Faster
    run_moving(y, uy, t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(y, uy, t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(y, uy, t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(y, uy=-0.4, t=t, speed=6.0, answer=False)

    # =================================================================================
    # Moving: Surface moving in the negative direction
    # =================================================================================
    t = 13.0  # Surface y-position = 6.0; surface y-velocity = -3.0

    # Not at surface
    uy = 0.3  # Arbitrary
    speed = 3.0  # Arbitrary
    ## Positive side
    run_moving(y=13.0, uy=uy, t=t, speed=speed, answer=True)
    ## Negative side
    run_moving(y=5.0, uy=uy, t=t, speed=speed, answer=False)

    # At surface, positive side
    y = 6.0 + TINY
    #
    ## Positive direction (opposite direction)
    run_moving(y, uy=0.6, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    uy = -0.6
    ### Faster
    run_moving(y, uy, t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(y, uy, t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(y, uy, t, speed=5.0, answer=False)

    # At surface, negative side
    y = 6.0 - TINY
    #
    ## Positive direction (opposite direction)
    run_moving(y, uy=0.6, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    uy = -0.6
    ### Faster
    run_moving(y, uy, t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(y, uy, t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(y, uy, t, speed=5.0, answer=False)


def test_interface_get_distance():
    def run_static(y, uy, answer):
        particle["y"] = y
        particle["uy"] = uy
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(y, uy, t, speed, answer):
        particle["y"] = y
        particle["uy"] = uy
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    y = 12.0
    ## Positive direction (moving away)
    run_static(y, uy=0.3, answer=INF)
    ## Negative direction (moving closer)
    run_static(y, uy=-0.4, answer=5.0)
    ## Parallel
    run_static(y, uy=0.0, answer=INF)

    # Negative side
    y = 6.0
    ## Positive direction (moving closer)
    run_static(y, uy=0.4, answer=10.0)
    ## Negative direction (moving away)
    run_static(y, uy=-0.3, answer=INF)
    ## Parallel
    run_static(y, uy=0.0, answer=INF)

    # At surface, on the positive side
    y = 10.0 + TINY
    ## Positive direction
    run_static(y, uy=0.4, answer=INF)
    ## Negative direction
    run_static(y, uy=-0.4, answer=INF)
    ## Parallel
    run_static(y, uy=0.0, answer=INF)

    # At surface, on the negative side
    y = 10.0 - TINY
    ## Positive direction
    run_static(y, uy=0.4, answer=INF)
    ## Negative direction
    run_static(y, uy=-0.4, answer=INF)
    ## Parallel
    run_static(y, uy=0.0, answer=INF)

    # =================================================================================
    # Moving
    # =================================================================================
    # Bin 0: t = [ 0.0,  5.0]; surface_y = [5.0, 10.0]; surface_speed = -1.0
    # Bin 1: t = [ 5.0, 10.0]; surface_y = [5.0, 15.0]; surface_speed =  2.0
    # Bin 2: t = [10.0, 15.0]; surface_y = [0.0, 15.0]; surface_speed = -3.0
    # Bin 3: t = [15.0,  INF]; surface_y = [0.0,  0.0]; surface_speed =  0.0

    def distance(y, uy, speed, bin_idy):
        t0 = 0.0 + np.sum(durations[:bin_idy])
        surface_y = Y + np.sum(durations[:bin_idy] * velocities[:bin_idy, 1])
        surface_speed = velocities[bin_idy, 1]
        return ((surface_speed * -t0) - (y - surface_y)) / (uy - surface_speed / speed)

    # Start from the beginning
    t = 0.0

    # Positive side
    y = 11.0
    #
    ## Positive direction (moving away)
    uy = 0.4
    ### No hit
    run_moving(y, uy, t, speed=1.0, answer=INF)
    ### Hit (rear-ended by the surface)
    answer = distance(y, uy, speed=0.9, bin_idy=1)
    run_moving(y, uy, t, speed=0.9, answer=answer)
    #
    ## Negative direction (moving closer)
    uy = -0.4
    ### Hit (rear-end the surface)
    answer = distance(y, uy, speed=3.0, bin_idy=0)
    run_moving(y, uy, t, speed=3.0, answer=answer)
    ### Hit (head-on)
    answer = distance(y, uy, speed=0.1, bin_idy=1)
    run_moving(y, uy, t, speed=0.1, answer=answer)

    # Negative side
    y = 7.0
    #
    ## Negative direction (moving away)
    uy = -0.4
    ### No hit
    run_moving(y, uy, t, speed=2.0, answer=INF)
    ### Hit (rear-ended by the surface)
    answer = distance(y, uy, speed=0.4, bin_idy=0)
    run_moving(y, uy, t, speed=0.4, answer=answer)
    #
    ## Positive direction (moving closer)
    uy = 0.4
    ### Hit (head-on)
    answer = distance(y, uy, speed=0.1, bin_idy=0)
    run_moving(y, uy, t, speed=0.1, answer=answer)
    ### Hit (rear-end the surface)
    y = -10.0
    answer = distance(y, uy, speed=20.0 / 3.0, bin_idy=1)
    run_moving(y, uy, t, speed=20.0 / 3.0, answer=answer)
