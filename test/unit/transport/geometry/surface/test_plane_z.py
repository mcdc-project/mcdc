import mcdc
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
Z = 10.0
durations = np.array([5.0, 5.0, 5.0])
velocities = np.zeros((3, 3))
velocities[:, 2] = np.array([-1.0, 2.0, -3.0])

static_surface = None
moving_surface = None
data = None
particle_container = None
particle = None

# Miscellanies
TINY = COINCIDENCE_TOLERANCE * 0.8  # Tiny value within coincidence tolerance

# Load modules to be tested
from mcdc.transport.geometry.surface import (
    interface,
    plane_z,
)


@pytest.fixture(autouse=True)
def setup_geometry_case(compile_surfaces):
    global static_surface, moving_surface, data, particle_container, particle

    static_surface_obj = mcdc.Surface.PlaneZ(z=Z)
    moving_surface_obj = mcdc.Surface.PlaneZ(z=Z)
    moving_surface_obj.move(velocities, durations)

    static_surface, moving_surface, data = compile_surfaces(
        static_surface_obj, moving_surface_obj
    )
    particle_container = np.zeros(1, type_.particle_data)
    particle = particle_container[0]


# =====================================================================================
# Plane-Z core functions
# =====================================================================================


def test_evaluate():
    def run(z, answer):
        particle["z"] = z
        result = plane_z.evaluate(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive side
    run(z=13.0, answer=3.0)
    # Negative side
    run(z=6.0, answer=-4.0)


def test_reflect():
    def run(uz, answer):
        particle["uz"] = uz
        plane_z.reflect(particle_container, static_surface)
        assert np.isclose(particle["uz"], answer)

    # From positive direction
    run(uz=0.2, answer=-0.2)
    # From negative direction
    run(uz=-0.1, answer=0.1)


def test_get_normal_component():
    def run(uz, answer):
        particle["uz"] = uz
        result = plane_z.get_normal_component(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive direction
    run(uz=0.4, answer=0.4)
    # Negative direction
    run(uz=-0.2, answer=-0.2)
    # Parallel
    run(uz=0.0, answer=0.0)


def test_get_distance():
    def run(z, uz, answer):
        particle["z"] = z
        particle["uz"] = uz
        result = plane_z.get_distance(particle_container, static_surface)
        assert np.isclose(result, answer)

    # Positive side
    z = 12.0
    ## Moving closer
    run(z, uz=-0.4, answer=5.0)
    ## Moving away
    run(z, uz=0.3, answer=INF)
    ## Parallel
    run(z, uz=0.0, answer=INF)

    # Negative side
    z = 6.0
    ## Moving closer
    run(z, uz=0.4, answer=10.0)
    ## Moving away
    run(z, uz=-0.3, answer=INF)
    ## Parallel
    run(z, uz=0.0, answer=INF)

    # At surface, within tolerance, on the positive side
    z = 10.0 + TINY
    ## Moving away
    run(z, uz=0.4, answer=INF)
    ## Moving closer
    run(z, uz=-0.4, answer=INF)
    ## Parallel
    run(z, uz=0.0, answer=INF)

    # At surface, within tolerance, on the negative side
    z = 10.0 - TINY
    ## Moving away
    run(z, uz=-0.4, answer=INF)
    ## Moving closer
    run(z, uz=0.4, answer=INF)
    ## Parallel
    run(z, uz=0.0, answer=INF)


# =====================================================================================
# Plane-z integrated transport interface
# =====================================================================================


def test_interface_reflect():
    def run(uz, answer):
        particle["uz"] = uz
        interface.reflect(particle_container, static_surface)
        assert np.isclose(particle["uz"], answer)

    # From positive direction
    run(uz=0.2, answer=-0.2)
    # From negative direction
    run(uz=-0.1, answer=0.1)


def test_interface_evaluate():
    def run_static(z, answer):
        particle["z"] = z
        result = interface.evaluate(particle_container, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(z, uz, t, answer):
        particle["z"] = z
        particle["uz"] = uz
        particle["t"] = t
        result = interface.evaluate(particle_container, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    run_static(z=13.0, answer=3.0)
    # Negative side
    run_static(z=6.0, answer=-4.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin
    t = 3.0  # Surface z-position = 7.0
    uz = 0.3  # Arbitrary
    ## Positive side
    run_moving(z=10.0, uz=uz, t=t, answer=3.0)
    ## Negative side
    run_moving(z=1.0, uz=uz, t=t, answer=-6.0)

    # First bin, at grid
    t = 5.0  # Surface z-position = 5.0
    uz = -0.3  # Arbitrary
    ## Positive side
    run_moving(z=10.0, uz=uz, t=t, answer=5.0)
    ## Negative side
    run_moving(z=1.0, uz=uz, t=t, answer=-4.0)

    # Interior bin
    t = 12.0  # Surface z-position = 9.0
    uz = 0.3  # Arbitrary
    ## Positive side
    run_moving(z=10.0, uz=uz, t=t, answer=1.0)
    ## Negative side
    run_moving(z=1.0, uz=uz, t=t, answer=-8.0)

    # Interior bin, at grid
    t = 15.0  # Surface z-position = 0.0
    uz = -0.3  # Arbitrary
    ## Positive side
    run_moving(z=10.0, uz=uz, t=t, answer=10.0)
    ## Negative side
    run_moving(z=-5.0, uz=uz, t=t, answer=-5.0)

    # Final bin
    t = 100.0  # Surface z-position = 0.0
    uz = 0.3  # Arbitrary
    ## Positive side
    run_moving(z=10.0, uz=uz, t=t, answer=10.0)
    ## Negative side
    run_moving(z=-5.0, uz=uz, t=t, answer=-5.0)


def test_interface_get_normal_component():
    def run_static(uz, answer):
        particle["uz"] = uz
        speed = 2.0  # Arbitrary
        result = interface.get_normal_component(
            particle_container, speed, static_surface, data
        )
        assert np.isclose(result, answer)

    def run_moving(uz, t, speed, answer):
        particle["uz"] = uz
        particle["t"] = t
        result = interface.get_normal_component(
            particle_container, speed, moving_surface, data
        )
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive direction
    run_static(uz=0.4, answer=0.4)
    # Negative direction
    run_static(uz=-0.2, answer=-0.2)
    # Parallel
    run_static(uz=0.0, answer=0.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # Surface moving in the positive direction
    t = 8.5  # Surface z-velocity = 2.0
    #
    ## Positive direction
    uz = 0.4
    ### Faster
    run_moving(uz, t, speed=6.0, answer=0.4 / 6.0)
    ### Slower (change sign)
    run_moving(uz, t, speed=2.0, answer=-1.2 / 2.0)
    ### Same speed (cancel out)
    run_moving(uz, t, speed=5.0, answer=0.0)
    #
    ## Negative direction
    run_moving(uz=-0.4, t=t, speed=6.0, answer=-4.4 / 6.0)
    ## Parallel
    run_moving(uz=0.0, t=t, speed=6.0, answer=-2.0 / 6.0)

    # Surface moving in the negative direction
    t = 10.0  # Surface z-velocity = -3.0
    #
    ## Negative direction
    uz = -0.4
    ### Faster
    run_moving(uz, t, speed=8.0, answer=-0.2 / 8.0)
    ### Slower (change sign)
    run_moving(uz, t, speed=2.0, answer=2.2 / 2.0)
    ### Same speed (cancel out)
    run_moving(uz, t, speed=7.5, answer=0.0)
    #
    ## Positive direction
    run_moving(uz=0.4, t=t, speed=8.0, answer=6.2 / 8.0)
    ## Parallel
    run_moving(uz=0.0, t=t, speed=8.0, answer=3.0 / 8.0)


def test_interface_check_sense():
    def run_static(z, uz, answer):
        particle["z"] = z
        particle["uz"] = uz
        speed = 2.0  # Arbitrary
        result = interface.check_sense(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(z, uz, t, speed, answer):
        particle["z"] = z
        particle["uz"] = uz
        particle["t"] = t
        result = interface.check_sense(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Not at surface
    uz = 0.3  # Arbitrary
    ## Positive side
    run_static(z=12.0, uz=uz, answer=True)
    ## Negative side
    run_static(z=4.0, uz=uz, answer=False)

    # At surface, positive side
    z = 10.0 + TINY
    ## Positive direction
    run_static(z, uz=0.4, answer=True)
    ## Negative direction
    run_static(z, uz=-0.4, answer=False)

    # At surface, negative side
    z = 10.0 - TINY
    ## Positive direction
    run_static(z, uz=0.2, answer=True)
    ## Negative direction
    run_static(z, uz=-0.2, answer=False)

    # =================================================================================
    # Moving: Surface moving in the positive direction
    # =================================================================================
    t = 8.5  # Surface z-position = 12.0; surface z-velocity = 2.0

    # Not at surface
    uz = 0.3  # Arbitrary
    speed = 3.0  # Arbitrary
    ## Positive side
    run_moving(z=13.0, uz=uz, t=t, speed=speed, answer=True)
    ## Negative side
    run_moving(z=5.0, uz=uz, t=t, speed=speed, answer=False)

    # At surface, positive side
    z = 12.0 + TINY
    #
    ## Positive direction (same direction)
    uz = 0.4
    ### Faster
    run_moving(z, uz, t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(z, uz, t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(z, uz, t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(z, uz=-0.4, t=t, speed=6.0, answer=False)

    # At surface, negative side
    z = 12.0 - TINY
    ## Positive direction (same direction)
    uz = 0.4
    ### Faster
    run_moving(z, uz, t, speed=6.0, answer=True)
    ### Slower (passed by the surface)
    run_moving(z, uz, t, speed=4.0, answer=False)
    ### Same speed (undefined, but False is returned)
    run_moving(z, uz, t, speed=5.0, answer=False)
    #
    ## Negative direction (opposite direction)
    run_moving(z, uz=-0.4, t=t, speed=6.0, answer=False)

    # =================================================================================
    # Moving: Surface moving in the negative direction
    # =================================================================================
    t = 13.0  # Surface z-position = 6.0; surface z-velocity = -3.0

    # Not at surface
    uz = 0.3  # Arbitrary
    speed = 3.0  # Arbitrary
    ## Positive side
    run_moving(z=13.0, uz=uz, t=t, speed=speed, answer=True)
    ## Negative side
    run_moving(z=5.0, uz=uz, t=t, speed=speed, answer=False)

    # At surface, positive side
    z = 6.0 + TINY
    #
    ## Positive direction (opposite direction)
    run_moving(z, uz=0.6, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    uz = -0.6
    ### Faster
    run_moving(z, uz, t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(z, uz, t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(z, uz, t, speed=5.0, answer=False)

    # At surface, negative side
    z = 6.0 - TINY
    #
    ## Positive direction (opposite direction)
    run_moving(z, uz=0.6, t=t, speed=6.0, answer=True)
    #
    ## Negative direction (same direction)
    uz = -0.6
    ### Faster
    run_moving(z, uz, t, speed=6.0, answer=False)
    ### Slower (passed by surface)
    run_moving(z, uz, t, speed=4.0, answer=True)
    ### Same speed (undefined, but False is returned)
    run_moving(z, uz, t, speed=5.0, answer=False)


def test_interface_get_distance():
    def run_static(z, uz, answer):
        particle["z"] = z
        particle["uz"] = uz
        speed = 2.0  # Arbitrary
        result = interface.get_distance(particle_container, speed, static_surface, data)
        assert np.isclose(result, answer)

    def run_moving(z, uz, t, speed, answer):
        particle["z"] = z
        particle["uz"] = uz
        particle["t"] = t
        result = interface.get_distance(particle_container, speed, moving_surface, data)
        assert np.isclose(result, answer)

    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    z = 12.0
    ## Positive direction (moving away)
    run_static(z, uz=0.3, answer=INF)
    ## Negative direction (moving closer)
    run_static(z, uz=-0.4, answer=5.0)
    ## Parallel
    run_static(z, uz=0.0, answer=INF)

    # Negative side
    z = 6.0
    ## Positive direction (moving closer)
    run_static(z, uz=0.4, answer=10.0)
    ## Negative direction (moving away)
    run_static(z, uz=-0.3, answer=INF)
    ## Parallel
    run_static(z, uz=0.0, answer=INF)

    # At surface, on the positive side
    z = 10.0 + TINY
    ## Positive direction
    run_static(z, uz=0.4, answer=INF)
    ## Negative direction
    run_static(z, uz=-0.4, answer=INF)
    ## Parallel
    run_static(z, uz=0.0, answer=INF)

    # At surface, on the negative side
    z = 10.0 - TINY
    ## Positive direction
    run_static(z, uz=0.4, answer=INF)
    ## Negative direction
    run_static(z, uz=-0.4, answer=INF)
    ## Parallel
    run_static(z, uz=0.0, answer=INF)

    # =================================================================================
    # Moving
    # =================================================================================
    # Bin 0: t = [ 0.0,  5.0]; surface_z = [5.0, 10.0]; surface_speed = -1.0
    # Bin 1: t = [ 5.0, 10.0]; surface_z = [5.0, 15.0]; surface_speed =  2.0
    # Bin 2: t = [10.0, 15.0]; surface_z = [0.0, 15.0]; surface_speed = -3.0
    # Bin 3: t = [15.0,  INF]; surface_z = [0.0,  0.0]; surface_speed =  0.0

    def distance(z, uz, speed, bin_idz):
        t0 = 0.0 + np.sum(durations[:bin_idz])
        surface_z = Z + np.sum(durations[:bin_idz] * velocities[:bin_idz, 2])
        surface_speed = velocities[bin_idz, 2]
        return ((surface_speed * -t0) - (z - surface_z)) / (uz - surface_speed / speed)

    # Start from the beginning
    t = 0.0

    # Positive side
    z = 11.0
    #
    ## Positive direction (moving away)
    uz = 0.4
    ### No hit
    run_moving(z, uz, t, speed=1.0, answer=INF)
    ### Hit (rear-ended by the surface)
    answer = distance(z, uz, speed=0.9, bin_idz=1)
    run_moving(z, uz, t, speed=0.9, answer=answer)
    #
    ## Negative direction (moving closer)
    uz = -0.4
    ### Hit (rear-end the surface)
    answer = distance(z, uz, speed=3.0, bin_idz=0)
    run_moving(z, uz, t, speed=3.0, answer=answer)
    ### Hit (head-on)
    answer = distance(z, uz, speed=0.1, bin_idz=1)
    run_moving(z, uz, t, speed=0.1, answer=answer)

    # Negative side
    z = 7.0
    #
    ## Negative direction (moving away)
    uz = -0.4
    ### No hit
    run_moving(z, uz, t, speed=2.0, answer=INF)
    ### Hit (rear-ended by the surface)
    answer = distance(z, uz, speed=0.4, bin_idz=0)
    run_moving(z, uz, t, speed=0.4, answer=answer)
    #
    ## Positive direction (moving closer)
    uz = 0.4
    ### Hit (head-on)
    answer = distance(z, uz, speed=0.1, bin_idz=0)
    run_moving(z, uz, t, speed=0.1, answer=answer)
    ### Hit (rear-end the surface)
    z = -10.0
    answer = distance(z, uz, speed=20.0 / 3.0, bin_idz=1)
    run_moving(z, uz, t, speed=20.0 / 3.0, answer=answer)
