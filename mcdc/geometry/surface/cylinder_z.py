"""
Cylinder-Z: Infinite cylinder parallel to the z-axis

f(x, y) = xx + yy + Gx + Hy + J
"""

import math

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

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    J = surface["J"]

    return x**2 + y**2 + G * x + H * y + J


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]
    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]

    # Surface normal
    dx = 2 * particle["x"] + surface["G"]
    dy = 2 * particle["y"] + surface["H"]
    norm = (dx**2 + dy**2) ** 0.5
    nx = dx / norm
    ny = dy / norm

    # Reflect
    c = 2.0 * (nx * ux + ny * uy)
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny


@njit
def get_normal_component(particle_container, surface):
    particle = particle_container[0]
    # Surface normal
    dx = 2 * particle["x"] + surface["G"]
    dy = 2 * particle["y"] + surface["H"]
    norm = (dx**2 + dy**2) ** 0.5
    nx = dx / norm
    ny = dy / norm

    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]

    return nx * ux + ny * uy


@njit
def get_distance(particle_container, surface):
    particle = particle_container[0]
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    ux = particle["ux"]
    uy = particle["uy"]

    # Surface coefficients
    G = surface["G"]
    H = surface["H"]

    # Coincident?
    f = evaluate(particle_container, surface)
    coincident = abs(f) < COINCIDENCE_TOLERANCE
    if coincident:
        # Moving away or tangent?
        if (
            get_normal_component(particle_container, surface)
            >= 0.0 - COINCIDENCE_TOLERANCE
        ):
            return INF

    # Quadratic equation constants
    a = ux * ux + uy * uy
    b = 2 * (x * ux + y * uy) + G * ux + H * uy
    c = f

    determinant = b * b - 4.0 * a * c

    # Roots are complex : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF
    else:
        # Get the roots
        denom = 2.0 * a
        sqrt = math.sqrt(determinant)
        root_1 = (-b + sqrt) / denom
        root_2 = (-b - sqrt) / denom

        # Coincident?
        if coincident:
            return max(root_1, root_2)

        # Negative roots, moving away from the surface
        if root_1 < 0.0:
            root_1 = INF
        if root_2 < 0.0:
            root_2 = INF

        # Return the smaller root
        return min(root_1, root_2)
