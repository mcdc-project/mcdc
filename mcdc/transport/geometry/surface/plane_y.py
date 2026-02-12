"""
Plane-Y: Plane perpendicular to the y-axis

f(y) = y + J
"""

from numba import njit

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle_container, surface):
    particle = particle_container[0]
    return particle["y"] + surface["J"]


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]
    particle["uy"] = -particle["uy"]


@njit
def get_normal_component(particle_container, surface):
    particle = particle_container[0]
    return particle["uy"]


@njit
def get_distance(particle_container, surface):
    particle = particle_container[0]
    # Parallel?
    normal_component = get_normal_component(particle_container, surface)
    if abs(normal_component) == 0.0:
        return INF

    # Coincident?
    f = evaluate(particle_container, surface)
    if abs(f) < COINCIDENCE_TOLERANCE:
        return INF

    # Calculate distance
    distance = -f / normal_component

    # Moving away?
    if distance < 0.0:
        return INF

    return distance
