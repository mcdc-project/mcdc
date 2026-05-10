"""
Torus: General torus with arbitrary axis direction

For a torus centered at point c with unit axis d:

    p = (x, y, z) - c
    a = p dot d
    q = p dot p + R^2 - r^2
    f = q^2 - 4 R^2 (p dot p - a^2)

Where R is the major radius and r is the minor radius.
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
    particle = particle_container[0]

    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]

    dx = surface["nx"]
    dy = surface["ny"]
    dz = surface["nz"]
    R = surface["R"]
    r = surface["r"]

    p_dot_p = x * x + y * y + z * z # squared distance from the center
    p_dot_d = x * dx + y * dy + z * dz
    radial_sq = p_dot_p - p_dot_d * p_dot_d
    q = p_dot_p + R * R - r * r

    return q * q - 4.0 * R * R * radial_sq


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]

    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    dx, dy, dz = _get_gradient(x, y, z, surface)

    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    c = 2.0 * (nx * ux + ny * uy + nz * uz)
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny
    particle["uz"] -= c * nz


@njit
def get_normal_component(particle_container, surface):
    particle = particle_container[0]

    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    dx, dy, dz = _get_gradient(x, y, z, surface)

    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    return nx * ux + ny * uy + nz * uz


@njit
def get_distance(particle_container, surface):
    particle = particle_container[0]

    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    R = surface["R"]
    r = surface["r"]
    dx = surface["nx"]
    dy = surface["ny"]
    dz = surface["nz"]

    f = evaluate(particle_container, surface)
    coincident = abs(f) < COINCIDENCE_TOLERANCE
    if coincident:
        if (
            get_normal_component(particle_container, surface)
            >= 0.0 - COINCIDENCE_TOLERANCE
        ):
            return INF

    p_dot_p = x * x + y * y + z * z
    p_dot_u = x * ux + y * uy + z * uz
    u_dot_u = ux * ux + uy * uy + uz * uz
    p_dot_d = x * dx + y * dy + z * dz
    u_dot_d = ux * dx + uy * dy + uz * dz

    G = u_dot_u
    H = 2.0 * p_dot_u
    I = p_dot_p

    J = u_dot_u - u_dot_d * u_dot_d
    K = 2.0 * (p_dot_u - p_dot_d * u_dot_d)
    L = p_dot_p - p_dot_d * p_dot_d

    a4 = G * G
    a3 = 2.0 * G * H
    a2 = H * H + 2.0 * G * (I + R * R - r * r) - 4.0 * R * R * J
    a1 = 2.0 * H * (I + R * R - r * r) - 4.0 * R * R * K
    a0 = (I + R * R - r * r) ** 2 - 4.0 * R * R * L

    coefficients = np.array(
        [a4 + 0.0j, a3 + 0.0j, a2 + 0.0j, a1 + 0.0j, a0 + 0.0j],
        dtype=np.complex128,
    )
    roots = np.roots(coefficients)

    min_t = INF
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

    return min_t


@njit
def _get_gradient(x, y, z, surface):
    R = surface["R"]
    r = surface["r"]
    ax = surface["nx"]
    ay = surface["ny"]
    az = surface["nz"]

    p_dot_p = x * x + y * y + z * z
    p_dot_d = x * ax + y * ay + z * az
    q = p_dot_p + R * R - r * r
    radial_factor = q - 2.0 * R * R
    axis_factor = 2.0 * R * R * p_dot_d

    gx = 4.0 * (radial_factor * x + axis_factor * ax)
    gy = 4.0 * (radial_factor * y + axis_factor * ay)
    gz = 4.0 * (radial_factor * z + axis_factor * az)

    return gx, gy, gz
