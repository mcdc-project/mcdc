"""Dispatch geometric operations to the specialized surface kernels."""

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.geometry.motion as motion
import mcdc.transport.geometry.surface.plane_x as plane_x
import mcdc.transport.geometry.surface.plane_y as plane_y
import mcdc.transport.geometry.surface.plane_z as plane_z
import mcdc.transport.geometry.surface.plane as plane
import mcdc.transport.geometry.surface.cylinder_x as cylinder_x
import mcdc.transport.geometry.surface.cylinder_y as cylinder_y
import mcdc.transport.geometry.surface.cylinder_z as cylinder_z
import mcdc.transport.geometry.surface.cylinder as cylinder
import mcdc.transport.geometry.surface.cone_x as cone_x
import mcdc.transport.geometry.surface.cone_y as cone_y
import mcdc.transport.geometry.surface.cone_z as cone_z
import mcdc.transport.geometry.surface.cone as cone
import mcdc.transport.geometry.surface.sphere as sphere
import mcdc.transport.geometry.surface.quadric as quadric
import mcdc.transport.geometry.surface.torus_x as torus_x
import mcdc.transport.geometry.surface.torus_y as torus_y
import mcdc.transport.geometry.surface.torus_z as torus_z
import mcdc.transport.geometry.surface.torus as torus

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_PLANE,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_CYLINDER,
    SURFACE_SPHERE,
    SURFACE_QUADRIC,
    SURFACE_CONE_X,
    SURFACE_CONE_Y,
    SURFACE_CONE_Z,
    SURFACE_CONE,
    SURFACE_TORUS_X,
    SURFACE_TORUS_Y,
    SURFACE_TORUS_Z,
    SURFACE_TORUS,
)


@njit
def check_sense(particle_container, speed, surface, data):
    """
    Check on which side of the surface the particle is
        - Return True if on positive side
        - Return False otherwise
    Particle direction and speed are used to tiebreak coincidence.
    """
    particle = particle_container[0]
    result = evaluate(particle_container, surface, data)

    # Check if coincident on the surface
    if abs(result) < COINCIDENCE_TOLERANCE:
        # Determine sense based on the direction
        return (
            get_normal_component(particle_container, speed, surface, data)
            > 0.0  # TODO: Do we need to include COINCIDENCE TOLERANCE here?
        )

    return result > 0.0


@njit
def evaluate(particle_container, surface, data):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    particle = particle_container[0]
    if surface["moving"]:
        # Temporarily translate particle position
        x_original = particle["x"]
        y_original = particle["y"]
        z_original = particle["z"]
        idx = motion.get_move_idx(particle["t"], surface, data)
        _translate_particle_position(particle_container, surface, idx, data)

    if surface["linear"]:
        if surface["type"] == SURFACE_PLANE_X:
            result = plane_x.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Y:
            result = plane_y.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Z:
            result = plane_z.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE:
            result = plane.evaluate(particle_container, surface)
    if surface["quadric"]:
        if surface["type"] == SURFACE_CYLINDER_X:
            result = cylinder_x.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Y:
            result = cylinder_y.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Z:
            result = cylinder_z.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER:
            result = cylinder.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_SPHERE:
            result = sphere.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_X:
            result = cone_x.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Y:
            result = cone_y.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Z:
            result = cone_z.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_CONE:
            result = cone.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_QUADRIC:
            result = quadric.evaluate(particle_container, surface)
    if surface["quartic"]:
        if surface["type"] == SURFACE_TORUS_X:
            result = torus_x.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Y:
            result = torus_y.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Z:
            result = torus_z.evaluate(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS:
            result = torus.evaluate(particle_container, surface)

    if surface["moving"]:
        # Restore particle position
        particle["x"] = x_original
        particle["y"] = y_original
        particle["z"] = z_original

    return result


@njit
def get_normal_component(particle_container, speed, surface, data):
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
        idx = motion.get_move_idx(particle["t"], surface, data)
        _translate_particle_position(particle_container, surface, idx, data)
        _translate_particle_direction(particle_container, speed, surface, idx, data)

    if surface["linear"]:
        if surface["type"] == SURFACE_PLANE_X:
            result = plane_x.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Y:
            result = plane_y.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Z:
            result = plane_z.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE:
            result = plane.get_normal_component(particle_container, surface)
    if surface["quadric"]:
        if surface["type"] == SURFACE_CYLINDER_X:
            result = cylinder_x.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Y:
            result = cylinder_y.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Z:
            result = cylinder_z.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER:
            result = cylinder.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_SPHERE:
            result = sphere.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_X:
            result = cone_x.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Y:
            result = cone_y.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Z:
            result = cone_z.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_CONE:
            result = cone.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_QUADRIC:
            result = quadric.get_normal_component(particle_container, surface)
    if surface["quartic"]:
        if surface["type"] == SURFACE_TORUS_X:
            result = torus_x.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Y:
            result = torus_y.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Z:
            result = torus_z.get_normal_component(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS:
            result = torus.get_normal_component(particle_container, surface)

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
    if surface["linear"]:
        if surface["type"] == SURFACE_PLANE_X:
            return plane_x.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Y:
            return plane_y.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Z:
            return plane_z.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE:
            return plane.reflect(particle_container, surface)
    if surface["quadric"]:
        if surface["type"] == SURFACE_CYLINDER_X:
            return cylinder_x.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Y:
            return cylinder_y.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Z:
            return cylinder_z.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER:
            return cylinder.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_SPHERE:
            return sphere.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_X:
            return cone_x.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Y:
            return cone_y.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Z:
            return cone_z.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_CONE:
            return cone.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_QUADRIC:
            return quadric.reflect(particle_container, surface)
    if surface["quartic"]:
        if surface["type"] == SURFACE_TORUS_X:
            return torus_x.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Y:
            return torus_y.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Z:
            return torus_z.reflect(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS:
            return torus.reflect(particle_container, surface)


@njit
def get_distance(particle_container, speed, surface, data):
    """
    Get particle distance to surface

    Particle speed is needed if the surface is moving.
    """
    particle = particle_container[0]
    if surface["moving"]:
        return _get_distance_moving(particle_container, speed, surface, data)
    else:
        return _get_distance_static(particle_container, surface)


@njit
def _get_distance_static(particle_container, surface):
    """
    Get particle distance to static surface
    """
    particle = particle_container[0]
    if surface["linear"]:
        if surface["type"] == SURFACE_PLANE_X:
            return plane_x.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Y:
            return plane_y.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE_Z:
            return plane_z.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_PLANE:  # SHOULD BE REVIEWED
            return plane.get_distance(particle_container, surface)
        else:
            return INF
    if surface["quadric"]:
        if surface["type"] == SURFACE_CYLINDER_X:
            return cylinder_x.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Y:
            return cylinder_y.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER_Z:
            return cylinder_z.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CYLINDER:
            return cylinder.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_SPHERE:
            return sphere.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_X:
            return cone_x.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Y:
            return cone_y.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CONE_Z:
            return cone_z.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_CONE:
            return cone.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_QUADRIC:
            return quadric.get_distance(particle_container, surface)
        else:
            return INF
    if surface["quartic"]:
        if surface["type"] == SURFACE_TORUS_X:
            return torus_x.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Y:
            return torus_y.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS_Z:
            return torus_z.get_distance(particle_container, surface)
        elif surface["type"] == SURFACE_TORUS:
            return torus.get_distance(particle_container, surface)
        else:
            return INF


@njit
def _get_distance_moving(particle_container, speed, surface, data):
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
    idx = motion.get_move_idx(particle["t"], surface, data)

    # Distance accumulator
    total_distance = 0.0

    # Evaluate the current and the subsequent intervals until intersecting
    while idx < surface["N_move"]:
        # Translate particle position and direction
        _translate_particle_position(particle_container, surface, idx, data)
        _translate_particle_direction(particle_container, speed, surface, idx, data)

        # Get distance
        distance = _get_distance_static(particle_container, surface)

        # Intersection within the interval?
        distance_time = distance / speed
        dt = mcdc_get.surface.move_time_grid(idx + 1, surface, data) - particle["t"]
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
        particle["t"] = mcdc_get.surface.move_time_grid(idx + 1, surface, data)

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
def _translate_particle_position(particle_container, surface, idx, data):
    """Translate the particle into the surface's moving coordinate frame."""
    particle = particle_container[0]
    dx, dy, dz = motion.get_translation(particle["t"], surface, data, idx)
    particle["x"] -= dx
    particle["y"] -= dy
    particle["z"] -= dz


@njit
def _translate_particle_direction(particle_container, speed, surface, idx, data):
    """
    Translate particle direction wrt the given surface moving interval index
    """
    particle = particle_container[0]

    # Surface move velocities
    V = mcdc_get.surface.move_velocities_vector(idx, surface, data)

    # Translate the particle
    particle["ux"] -= V[0] / speed
    particle["uy"] -= V[1] / speed
    particle["uz"] -= V[2] / speed
