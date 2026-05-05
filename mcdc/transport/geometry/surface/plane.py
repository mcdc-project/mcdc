"""
Plane: General linear surface

f(x, y, z) = Gx + Hy + Iz + J
"""

from numba import njit

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle_container, surface):
    particle = particle_container[0]
    # Particle parameters
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    return G * x + H * y + I * z + J


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]
    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface normal
    nx = surface["G"]
    ny = surface["H"]
    nz = surface["I"]

    # Reflect
    c = 2.0 * (nx * ux + ny * uy + nz * uz)
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny
    particle["uz"] -= c * nz


@njit
def get_normal_component(particle_container, surface):
    particle = particle_container[0]
    # Surface normal
    nx = surface["G"]
    ny = surface["H"]
    nz = surface["I"]

    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    return nx * ux + ny * uy + nz * uz


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
