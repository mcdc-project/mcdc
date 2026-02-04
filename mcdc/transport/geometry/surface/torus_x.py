"""
Torus: Implicit equation of a torus radially symmetric about the z-axis in the cartesian plane

f(x, y, z) = ( sqrt[(x - A)^2 + (y - B)^2] - R )^2 + (z - C)^2 - r^2

Where R is the radius of the shape as a whole, and r is the radius of the circle that is revolved to create the donut

Removing the square roots leaves you with the following equation:
f (x, y, z) = ( x^2 + y^2 + z^2 + R^2 - a^2 )^2 - 4R^2 * (x^2 + y^2)
"""

import math

import numpy as np

import numpy.polynomial.polynomial as poly

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

    return(
        ( (x*x + y*y + z*z + R*R - r*r)**2 )
        - (4*R*R * (x*x + y*y))
    )


# Due to the sqrt in the derivative functions, if the particle is not in the x-y space where the torus has z values, it will result in complex numbers
# Specificaly the term: math.sqrt(r**2 - (R - inv_root)**2), where inverse root = math.sqrt((A-x)**2 + (B-y)**2)
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

    #Spliting up the derivative math for readability into numerators and denominators
    inv_root = math.sqrt((A-x)**2 + (B-y)**2)

    x_num = (x-A) * (inv_root - R)
    x_den = inv_root * (math.sqrt(r**2 - (R - inv_root)**2))
    y_num = (y-B) * (inv_root - R)
    y_den = (math.sqrt((A-x)**2 + (B-y)**2)) * (math.sqrt(r**2 - (R - inv_root)**2))

    # Surface derivatives, the if statement prevents a divide by 0 error
    # The derivative with respect to x and y are the same except for the first term (x-A) or (y-B)
    if x_den == 0 and x_num == 0:
        dx = 0
    elif x_den == 0:
        dx = -(INF)
    else:
        dx = -(x_num / x_den)

    if y_den == 0 and y_num == 0:
        dy = 0
    elif y_den == 0:
        dy = -(INF)
    else:
        dy = -(y_num / y_den)

    dz = -1

    # If the particle is below the centerline of the torus, it will be interacting with the bottom surface
    if z <= C:
        # The only difference between the derivatives of both halves of the torus is the multiplication of a negative 1
        dx *= -1
        dy *= -1

    # Surface Normal
    norm = math.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    # Reflect
    c = 2.0 * (nx * ux + ny * uy + nz * uz)
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny
    particle["uz"] -= c * nz

    print("Directions: ", ux, uy, uz)
    print("Derivitive Magnitude: ", norm)
    print("Derivitives: ", dx, dy, dz)
    print("C: ", c)
    print("Normals: ", nx, ny, nz)
    print(particle["ux"], particle["uy"], particle["uz"])

# Due to the sqrt in the derivative functions, if the particle is not in the x-y space where the torus has z values, it will result in complex numbers
# Specificaly the term: math.sqrt(r**2 - (R - inv_root)**2), where inverse root = math.sqrt((A-x)**2 + (B-y)**2)
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

    #Spliting up the derivative math for readability into numerators and denominators (inv is short for inverse)
    inv_root = math.sqrt((A-x)**2 + (B-y)**2)

    x_num = (x-A) * (inv_root - R)
    x_den = inv_root * (math.sqrt(r**2 - (R - inv_root)**2))
    y_num = (y-B) * (inv_root - R)
    y_den = (math.sqrt((A-x)**2 + (B-y)**2)) * (math.sqrt(r**2 - (R - inv_root)**2))

    # Surface derivatives, the if statement prevents a divide by 0 error
    # The derivative with respect to x and y are the same except for the first term (x-A) or (y-B)
    if x_den == 0 and x_num == 0:
        dx = 0
    elif x_den == 0:
        dx = -(INF)
    else:
        dx = -(x_num / x_den)

    if y_den == 0 and y_num == 0:
        dy = 0
    elif y_den == 0:
        dy = -(INF)
    else:
        dy = -(y_num / y_den)

    dz = -1

    # If the particle is below the centerline of the torus, it will be interacting with the bottom surface
    if z <= C:
        # The only difference between the derivatives of both halves of the torus is the multiplication of a negative 1
        dx *= -1
        dy *= -1

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

    # Shifting the origin point of the particle into the torus space, and treating the torus as centered on (0,0,0)
    x -= A
    y -= B
    z -= C

    # Dot products that come up frequently in the torus-ray intersection equation
    G = ux*ux + uy*uy + uz*uz
    H = 2.0 * (x*ux + y*uy + z*uz)
    I = x*x + y*y + z*z

    J = ux*ux + uy*uy
    K = 2.0 * (x*ux + y*uy)
    L = x*x + y*y

    # Quartic coefficients from substituting (i = origin_i + direction_i * t) into each axis for i (x,y,z)
    a4 = G * G
    a3 = 2.0 * G * H
    a2 = H*H + 2.0*G*(I + R*R - r*r) - 4.0*R*R*J
    a1 = 2.0*H*(I + R*R - r*r) - 4.0*R*R*K
    a0 = (I + R*R - r*r)**2 - 4.0*R*R*L

    # Use the numpy polynomial library to solve the quartic above for t
    coefficients = [a0, a1, a2, a3, a4]
    roots = np.ndarray.tolist(poly.polyroots(coefficients))
    real_roots = []

    # Filtering the roots for real solutions
    for solution in roots:
        if isinstance(solution, complex):
            pass
        elif solution >= 0:
            real_roots.append(solution)

    if len(real_roots) == 0: # Ending the calculation if there are no valid solutions
        return INF

    # Using the smallest root to get the value of t at the first point of intersection
    # If the direction vector is normalized, the distance to intersection is just the value of t
    min_t = min(real_roots)
    ray_length = math.sqrt(G)

    return min_t * ray_length
