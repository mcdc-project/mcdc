"""
Surface operations based on the quadric equation:
   f(x,y,z) = Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J
"""

import math

from numba import njit

import mcdc.src.surface.plane_x as plane_x
import mcdc.src.surface.plane_y as plane_y
import mcdc.src.surface.plane_z as plane_z
import mcdc.src.surface.plane as plane
import mcdc.src.surface.cylinder_x as cylinder_x
import mcdc.src.surface.cylinder_y as cylinder_y
import mcdc.src.surface.cylinder_z as cylinder_z
import mcdc.src.surface.sphere as sphere
import mcdc.src.surface.quadric as quadric

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
    SURFACE_LINEAR,
    SURFACE_QUADRIC,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_PLANE,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_SPHERE,
    SURFACE_QUADRIC,
)
from mcdc.src.algorithm import binary_search_with_length


@njit
def check_sense(particle_container, speed, surface):
    """
    Check on which side of the surface the particle is
        - Return True if on positive side
        - Return False otherwise
    Particle direction and speed are used to tiebreak coincidence.
    """
    particle = particle_container[0]
    result = evaluate(particle_container, surface)

    # Check if coincident on the surface
    if abs(result) < COINCIDENCE_TOLERANCE:
        # Determine sense based on the direction
        return (
            get_normal_component(particle_container, speed, surface)
            > 0.0  # TODO: Do we need to include COINCIDENCE TOLERANCE here?
        )

    return result > 0.0


@njit
def evaluate(particle_container, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    particle = particle_container[0]
    if surface["moving"]:
        # Temporarily translate particle position
        x_original = particle["x"]
        y_original = particle["y"]
        z_original = particle["z"]
        idx = _get_move_idx(particle["t"], surface)
        _translate_particle_position(particle_container, surface, idx)

    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            result = plane_x.evaluate(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            result = plane_y.evaluate(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            result = plane_z.evaluate(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE:
            result = plane.evaluate(particle_container, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            result = cylinder_x.evaluate(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            result = cylinder_y.evaluate(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            result = cylinder_z.evaluate(particle_container, surface)
        elif surface["type"] & SURFACE_SPHERE:
            result = sphere.evaluate(particle_container, surface)
        else:
            result = quadric.evaluate(particle_container, surface)

    if surface["moving"]:
        # Restore particle position
        particle["x"] = x_original
        particle["y"] = y_original
        particle["z"] = z_original

    return result


@njit
def get_normal_component(particle_container, speed, surface):
    """
    Get the surface outward-normal component of the particle
    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    particle = particle_container[0]
    if surface["moving"]:
        # Temporarily translate particle parameters
        x_original = particle["x"]
        y_original = particle["y"]
        z_original = particle["z"]
        ux_original = particle["ux"]
        uy_original = particle["uy"]
        uz_original = particle["uz"]
        idx = _get_move_idx(particle["t"], surface)
        _translate_particle_position(particle_container, surface, idx)
        _translate_particle_direction(particle_container, speed, surface, idx)

    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            result = plane_x.get_normal_component(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            result = plane_y.get_normal_component(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            result = plane_z.get_normal_component(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE:
            result = plane.get_normal_component(particle_container, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            result = cylinder_x.get_normal_component(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            result = cylinder_y.get_normal_component(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            result = cylinder_z.get_normal_component(particle_container, surface)
        elif surface["type"] & SURFACE_SPHERE:
            result = sphere.get_normal_component(particle_container, surface)
        else:
            result = quadric.get_normal_component(particle_container, surface)

    if surface["moving"]:
        # Restore particle parameters
        particle["x"] = x_original
        particle["y"] = y_original
        particle["z"] = z_original
        particle["ux"] = ux_original
        particle["uy"] = uy_original
        particle["uz"] = uz_original

    return result


@njit
def reflect(particle_container, surface):
    """
    Reflect the particle off the surface
    """
    particle = particle_container[0]
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.reflect(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.reflect(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.reflect(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE:
            return plane.reflect(particle_container, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.reflect(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.reflect(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.reflect(particle_container, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.reflect(particle_container, surface)
        else:
            return quadric.reflect(particle_container, surface)


@njit
def get_distance(particle_container, speed, surface):
    """
    Get particle distance to surface

    Particle speed is needed if the surface is moving.
    """
    particle = particle_container[0]
    if surface["moving"]:
        return _get_distance_moving(particle_container, speed, surface)
    else:
        return _get_distance_static(particle_container, surface)


@njit
def _get_distance_static(particle_container, surface):
    """
    Get particle distance to static surface
    """
    particle = particle_container[0]
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.get_distance(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.get_distance(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.get_distance(particle_container, surface)
        elif surface["type"] & SURFACE_PLANE:  # SHOULD BE REVIEWED
            return plane.get_distance(particle_container, surface)
        else:
            return INF
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.get_distance(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.get_distance(particle_container, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.get_distance(particle_container, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.get_distance(particle_container, surface)
        else:
            return quadric.get_distance(particle_container, surface)


@njit
def _get_distance_moving(particle_container, speed, surface):
    """
    Get particle distance to moving surface
    """
    particle = particle_container[0]
    # Store original particle parameters (will be temporarily changed)
    x_original = particle["x"]
    y_original = particle["y"]
    z_original = particle["z"]
    ux_original = particle["ux"]
    uy_original = particle["uy"]
    uz_original = particle["uz"]
    t_original = particle["t"]

    # Move interval index
    idx = _get_move_idx(particle["t"], surface)

    # Distance accumulator
    total_distance = 0.0

    # Evaluate the current and the subsequent intervals until intersecting
    while idx < surface["N_move"]:
        # Translate particle position and direction
        _translate_particle_position(particle_container, surface, idx)
        _translate_particle_direction(particle_container, speed, surface, idx)

        # Get distance
        distance = _get_distance_static(particle_container, surface)

        # Intersection within the interval?
        distance_time = distance / speed
        dt = surface["move_time_grid"][idx + 1] - particle["t"]
        if distance_time < dt:
            # Restore particle parameters
            particle["x"] = x_original
            particle["y"] = y_original
            particle["z"] = z_original
            particle["ux"] = ux_original
            particle["uy"] = uy_original
            particle["uz"] = uz_original
            particle["t"] = t_original

            # Return the total distance
            return total_distance + distance

        # Accumulate distance
        total_distance += dt * speed

        # Modify the particle
        particle["x"] = x_original + total_distance * ux_original
        particle["y"] = y_original + total_distance * uy_original
        particle["z"] = z_original + total_distance * uz_original
        particle["ux"] = ux_original
        particle["uy"] = uy_original
        particle["uz"] = uz_original
        particle["t"] = surface["move_time_grid"][idx + 1]

        # Check next interval
        idx += 1

    # Restore particle parameters
    particle["x"] = x_original
    particle["y"] = y_original
    particle["z"] = z_original
    particle["ux"] = ux_original
    particle["uy"] = uy_original
    particle["uz"] = uz_original
    particle["t"] = t_original

    # No intersection
    return INF


# ======================================================================================
# Private
# ======================================================================================


@njit
def _get_move_idx(t, surface):
    """
    Get moving interval index wrt the given time
    """
    N_move = surface["N_move"]
    time_grid = surface["move_time_grid"]
    idx = binary_search_with_length(t, time_grid, N_move)

    # Coinciding cases
    if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
        idx += 1

    return idx


@njit
def _translate_particle_position(particle_container, surface, idx):
    """
    Translate particle position wrt the given surface moving interval index
    """
    particle = particle_container[0]

    # Surface move translations, velocities, and time grid
    trans_0 = surface["move_translations"][idx]
    time_0 = surface["move_time_grid"][idx]
    V = surface["move_velocities"][idx]

    # Translate the particle
    t_local = particle["t"] - time_0
    particle["x"] -= trans_0[0] + V[0] * t_local
    particle["y"] -= trans_0[1] + V[1] * t_local
    particle["z"] -= trans_0[2] + V[2] * t_local


@njit
def _translate_particle_direction(particle_container, speed, surface, idx):
    """
    Translate particle direction wrt the given surface moving interval index
    """
    particle = particle_container[0]

    # Surface move translations, velocities, and time grid
    V = surface["move_velocities"][idx]

    # Translate the particle
    particle["ux"] -= V[0] / speed
    particle["uy"] -= V[1] / speed
    particle["uz"] -= V[2] / speed
