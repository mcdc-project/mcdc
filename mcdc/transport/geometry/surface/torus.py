"""
Torus: Implicit equation of a torus radially symmetric about an arbitrary axis

Let p be a point, c be the torus center, and a be a unit vector parallel to the
torus axis. First shift p into torus-centered coordinates. In the code, x, y,
and z are the shifted coordinates, so:

    p_dot_p = x*x + y*y + z*z
    p_dot_d = x*dx + y*dy + z*dz
    radial_sq = p_dot_p - p_dot_d*p_dot_d

The geometric torus equation is:

    (sqrt(radial_sq) - R)^2 + p_dot_d^2 = r^2

Using radial_sq + p_dot_d^2 = p_dot_p, this becomes:

    p_dot_p + R^2 - r^2 = 2R * sqrt(radial_sq)

The code defines q = p_dot_p + R*R - r*r. Squaring both sides gives the
implicit equation:

    f(p) = q*q - 4*R*R*radial_sq
"""

import math

import numpy as np

from numba import njit

import mcdc.transport.util as util
import mcdc.transport.geometry.root_solve as root_solve

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle_container, surface):
    # Particle parameters
    particle = particle_container[0]
    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]

    # Surface parameters
    dx = surface["nx"]
    dy = surface["ny"]
    dz = surface["nz"]
    R = surface["R"]
    r = surface["r"]

    # Dot products for the implicit arbitrary-axis torus equation
    p_dot_p = x * x + y * y + z * z  # squared distance from the center
    p_dot_d = x * dx + y * dy + z * dz
    radial_sq = p_dot_p - p_dot_d * p_dot_d
    q = p_dot_p + R * R - r * r

    return q * q - 4.0 * R * R * radial_sq


@njit
def reflect(particle_container, surface):
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface gradient
    dx, dy, dz = _get_gradient(x, y, z, surface)

    # Surface normal
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
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

    # Particle coordinate
    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface gradient
    dx, dy, dz = _get_gradient(x, y, z, surface)

    # Surface normal
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    return nx * ux + ny * uy + nz * uz


@njit
def get_distance(particle_container, surface):
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"] - surface["A"]
    y = particle["y"] - surface["B"]
    z = particle["z"] - surface["C"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface coefficients
    R = surface["R"]
    r = surface["r"]
    dx = surface["nx"]
    dy = surface["ny"]
    dz = surface["nz"]

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

    # Dot products that come up frequently in the torus-ray intersection equation
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

    # Quartic coefficients from substituting particle_position + particle_direction * t
    a4 = G * G
    a3 = 2.0 * G * H
    a2 = H * H + 2.0 * G * (I + R * R - r * r) - 4.0 * R * R * J
    a1 = 2.0 * H * (I + R * R - r * r) - 4.0 * R * R * K
    a0 = (I + R * R - r * r) ** 2 - 4.0 * R * R * L

    # TODO: May replace with a fully numba-native quartic solver if torus performance becomes important;
    coefficients = util.local_array(5,np.complex128)
    coefficients[0] = a4 + 0.0j
    coefficients[1] = a3 + 0.0j
    coefficients[2] = a2 + 0.0j
    coefficients[3] = a1 + 0.0j
    coefficients[4] = a0 + 0.0j
    roots = util.local_array(4,np.complex128)
    root_solve.solve_quartic(coefficients,roots)

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

    return min_t


@njit
def _get_gradient(x, y, z, surface):
    # Surface coefficients
    R = surface["R"]
    r = surface["r"]
    ax = surface["nx"]
    ay = surface["ny"]
    az = surface["nz"]

    # Dot products for the gradient of the implicit arbitrary-axis torus equation
    p_dot_p = x * x + y * y + z * z
    p_dot_d = x * ax + y * ay + z * az
    q = p_dot_p + R * R - r * r
    radial_factor = q - 2.0 * R * R
    axis_factor = 2.0 * R * R * p_dot_d

    gx = 4.0 * (radial_factor * x + axis_factor * ax)
    gy = 4.0 * (radial_factor * y + axis_factor * ay)
    gz = 4.0 * (radial_factor * z + axis_factor * az)

    return gx, gy, gz
