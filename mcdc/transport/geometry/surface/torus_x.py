"""
Torus: Implicit equation of a torus radially symmetric about the x-axis

f(x, y, z) = ( sqrt[(y - B)^2 + (z - C)^2] - R )^2 + (x - A)^2 - r^2

Where R is the radius of the shape as a whole, and r is the radius of the circle that is revolved to create the donut

Removing the square roots leaves you with the following equation:
f (x, y, z) = ( x^2 + y^2 + z^2 + R^2 - r^2 )^2 - 4R^2 * (y^2 + z^2)
"""

import math

import numpy as np

from numba import njit

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle_container, surface):
    """
    Description:
    Checking to see if the particle is on the surface in question

    Returns: (float)
    - If the return is 0, the particle occupies the exact space of the torus
    - If the return is positive, the particle is outside the torus
    - If the return is negative, the particle is inside the torus
    """

    # Particle parameters
    particle = particle_container[0]
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    # Surface parameters
    R = surface["R"]
    r = surface["r"]
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]

    # Shifting the origin point of the particle into the torus space, and treating the torus as centered on (0,0,0)
    x -= A
    y -= B
    z -= C

    return ((x * x + y * y + z * z + R * R - r * r) ** 2) - (
        4 * R * R * (y * y + z * z)
    )


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface coefficients
    R = surface["R"]
    r = surface["r"]
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]

    # Shifting the origin point of the particle into the torus space, and treating the torus as centered on (0,0,0)
    x -= A
    y -= B
    z -= C

    # Taking the partial derivatives of the expanded form of the implicit torus equation
    dx = 4 * x * (-(r**2) + (R**2) + (x**2) + (y**2) + (z**2))
    dy = 4 * y * (-(r**2) - (R**2) + (x**2) + (y**2) + (z**2))
    dz = 4 * z * (-(r**2) - (R**2) + (x**2) + (y**2) + (z**2))

    # Surface Normal
    norm = math.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    # Reflect
    c = 2.0 * (
        nx * ux + ny * uy + nz * uz
    )  # Magnitude component of the projection of the particle onto the surface normal
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny
    particle["uz"] -= c * nz


@njit
def get_normal_component(particle_container, surface):
    particle = particle_container[0]
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface coefficients
    R = surface["R"]
    r = surface["r"]
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]

    # Shifting the origin point of the particle into the torus space, and treating the torus as centered on (0,0,0)
    x -= A
    y -= B
    z -= C

    # Taking the partial derivatives of the expanded form of the implicit torus equation
    dx = 4 * x * (-(r**2) + (R**2) + (x**2) + (y**2) + (z**2))
    dy = 4 * y * (-(r**2) - (R**2) + (x**2) + (y**2) + (z**2))
    dz = 4 * z * (-(r**2) - (R**2) + (x**2) + (y**2) + (z**2))

    # Surface Normal
    norm = math.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

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
    R = surface["R"]
    r = surface["r"]
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]

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

    # Shifting the origin point of the particle into the torus space, and treating the torus as centered on (0,0,0)
    x -= A
    y -= B
    z -= C

    # Dot products that come up frequently in the torus-ray intersection equation
    G = ux * ux + uy * uy + uz * uz
    H = 2.0 * (x * ux + y * uy + z * uz)
    I = x * x + y * y + z * z

    J = uy * uy + uz * uz
    K = 2.0 * (y * uy + z * uz)
    L = y * y + z * z

    # Quartic coefficients from substituting (i = origin_i + direction_i * t) into each axis for i (x,y,z)
    a4 = G * G
    a3 = 2.0 * G * H
    a2 = H * H + 2.0 * G * (I + R * R - r * r) - 4.0 * R * R * J
    a1 = 2.0 * H * (I + R * R - r * r) - 4.0 * R * R * K
    a0 = (I + R * R - r * r) ** 2 - 4.0 * R * R * L

    # TODO: May replace with a fully numba-native quartic solver if torus performance becomes important;
    # np.roots is sufficient for now.
    coefficients = np.array(
        [a4 + 0.0j, a3 + 0.0j, a2 + 0.0j, a1 + 0.0j, a0 + 0.0j],
        dtype=np.complex128,
    )
    roots = np.roots(coefficients)

    min_t = INF

    # TODO: Add stricter coverage/handling for near-tangent and off-axis torus intersections.
    # Current root filtering relies on COINCIDENCE_TOLERANCE for near-real and near-zero roots.
    for solution in roots:
        if abs(solution.imag) >= COINCIDENCE_TOLERANCE:
            continue

        root = solution.real
        if coincident:
            if root <= COINCIDENCE_TOLERANCE:
                continue
        elif root < 0.0:
            continue

        if root < min_t:
            min_t = root

    if min_t == INF:
        return INF

    return min_t
