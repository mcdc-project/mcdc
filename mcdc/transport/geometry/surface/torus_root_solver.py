import math
import cmath
import numpy as np
import numba as nb

import mcdc.transport.util as util

from numba import njit


@njit()
def modulus(x):
    return math.sqrt(x.real**2 + x.imag**2)


@njit()
def sqrt(x):
    r = modulus(x)
    real_part = math.sqrt((r + x.real) / 2.0)
    if x.imag < 0:
        imag_part = -math.sqrt((r - x.real) / 2.0)
    else:
        imag_part = math.sqrt((r - x.real) / 2.0)

    return complex(real_part, imag_part)


@njit()
def power(x, n):
    result = 1
    for i in range(n):
        result = result * x
    return result


@njit()
def nth_root(x, n, index):
    # First, convert to polar form
    r = modulus(x)
    a = math.atan2(x.imag, x.real)

    # Apply de Moivre's Formula
    root_modulus = math.pow(r, 1.0 / n)
    root_argument = (a + 2 * math.pi * index) / n

    # ...then convert back to rectangular form
    real = root_modulus * math.cos(root_argument)
    imag = root_modulus * math.sin(root_argument)
    return complex(real, imag)


@njit()
def principal_nth_root(x, n):
    result = nth_root(x, n, 0)
    return result


@njit()
def solve_quadratic(coeff, roots):
    a = coeff[2]
    b = coeff[1]
    c = coeff[0]
    # standard quadratic formula, but with discriminant
    # calculated separately for re-use
    discriminant = sqrt(power(b, 2) - 4 * a * c)
    roots[0] = ((-b) + discriminant) / (2 * a)
    roots[1] = ((-b) - discriminant) / (2 * a)


@njit()
def solve_biquadratic(coeff, roots):
    # Move each coefficient down to one-half it's power
    coeff[1] = coeff[2]
    coeff[2] = coeff[4]

    # Solve as quadratic equation, where the variable is
    # actually x^2
    solve_quadratic(coeff, roots)

    # Yield roots for x by taking square roots of the x^2
    # solution.
    roots[3] = sqrt(roots[1])
    roots[2] = -roots[3]
    roots[1] = sqrt(roots[0])
    roots[0] = -roots[1]

    # Restore the original positions of the coefficients
    coeff[4] = coeff[2]
    coeff[2] = coeff[1]
    coeff[1] = 0.0j


@njit()
def solve_cubic(coeff, roots):
    # TODO
    # General soluton not needed for quartic solve
    pass


@njit()
def solve_depressed_quartic(coeff, roots):
    a = coeff[2]
    b = coeff[1]
    c = coeff[0]

    # To solve the depressed quartic, one must first find one
    # root of a cubic polynomial.

    p = (-power(a, 2) / 12.0) - c
    q = (-power(a, 3) / 108.0) + (a * c / 3.0) - (power(b, 2) / 8.0)

    cube_const = -q / 2.0
    sqrt_body = (power(q, 2) / 4.0) + (power(p, 3) / 27.0)
    w_pos = principal_nth_root(cube_const + sqrt(sqrt_body), 3)
    w_neg = principal_nth_root(cube_const - sqrt(sqrt_body), 3)

    # It's reccomended to opt for the larger w when
    # calculating the root.
    if abs(w_pos) > abs(w_neg):
        w = w_pos
    else:
        w = w_neg

    # A root of the cubic
    y = (a / 6.0) + w - (p / (3.0 * w))

    # The different roots are found by flipping the signs
    # of some terms in a formula. There are three sections
    # unaffected by these flips, represented below by
    # alpha, beta, and gamma

    alpha = sqrt(2.0 * y - a)
    beta = -2.0 * y - a
    gamma = (2.0 * b) / sqrt(2.0 * y - a)

    roots[0] = ((-alpha) + sqrt(beta + gamma)) / 2.0  # - + +
    roots[1] = ((-alpha) - sqrt(beta + gamma)) / 2.0  # - - +
    roots[2] = ((alpha) + sqrt(beta - gamma)) / 2.0  # + + -
    roots[3] = ((alpha) - sqrt(beta - gamma)) / 2.0  # + - -


@njit()
def solve_quartic(coeff, roots):
    # Algorithm logic derived from Wikipedia's quartic
    # equation article. (^-^)=b

    # Coefficients for the general solve
    a = coeff[4]
    b = coeff[3]
    c = coeff[2]
    d = coeff[1]
    e = coeff[0]

    # Coefficients for the sub-solve
    # The sub-solve turns the equation into a depressed
    # quartic by making u the new variable, with:
    #
    #     x = u - (bg/(4*ag))
    #
    # Once the roots for the depressed quartic are found,
    # they can be plugged into this equation to yeild the
    # roots for x.

    sub_coeff = util.local_array(5, np.complex128)
    sub_coeff[4] = 1.0 + 0.0j
    sub_coeff[3] = 0.0j
    sub_coeff[2] = (-3.0 * power(b, 2)) / (8.0 * power(a, 2)) + c / a
    sub_coeff[1] = (
        power(b, 3) / (8.0 * power(a, 3)) - (b * c) / (2.0 * power(a, 2)) + d / a
    )
    sub_coeff[0] = (
        (-3.0 * power(b, 4)) / (256.0 * power(a, 4))
        + (c * power(b, 2)) / (16.0 * power(a, 3))
        - (b * d) / (4.0 * power(a, 2))
        + e / a
    )

    # Get roots of sub-solve
    sub_roots = util.local_array(4, np.complex128)
    if sub_coeff[1] == 0:
        # If the linear term coefficient is zero, the
        # normal depressed quartic solver won't work.
        # Instead, it is a biquadratic, and can be
        # solved as such.

        solve_biquadratic(sub_coeff, sub_roots)
    else:
        solve_depressed_quartic(sub_coeff, sub_roots)

    for idx in range(4):
        roots[idx] = sub_roots[idx] - b / (4.0 * a)
