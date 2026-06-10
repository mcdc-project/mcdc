"""
Sphere

f(x, y, z) = xx + yy + zz + Gx + Hy + Iz + J
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
    z = particle["z"]

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    return x**2 + y**2 + z**2 + G * x + H * y + I * z + J


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]
    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface normal
    dx = 2 * particle["x"] + surface["G"]
    dy = 2 * particle["y"] + surface["H"]
    dz = 2 * particle["z"] + surface["I"]
    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    # Reflect
    c = 2.0 * (nx * ux + ny * uy + nz * uz)
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny
    particle["uz"] -= c * nz


@njit
def get_normal_component(particle_container, surface):
    particle = particle_container[0]
    # Surface normal
    dx = 2 * particle["x"] + surface["G"]
    dy = 2 * particle["y"] + surface["H"]
    dz = 2 * particle["z"] + surface["I"]
    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    return nx * ux + ny * uy + nz * uz


@njit
def get_distance(particle_container, surface):
    particle = particle_container[0]
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface coefficients
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]

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
    b = 2 * (x * ux + y * uy + z * uz) + G * ux + H * uy + I * uz
    c = f

    determinant = b * b - 4.0 * c

    # Roots are complex : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF
    else:
        # Get the roots
        denom = 2.0
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
