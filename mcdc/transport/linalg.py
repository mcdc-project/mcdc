import math
from numba import njit


@njit
def cross(result, a, b):
    """Store the three-dimensional cross product ``a x b`` in ``result``.

    Parameters
    ----------
    result : ndarray
        Writable three-component output vector. May alias either input.
    a, b : array_like
        Three-component input vectors, in Cartesian (x, y, z) order.

    Returns
    -------
    None
        The result is written in place using the right-hand convention.
    """
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    result[0] = x
    result[1] = y
    result[2] = z


@njit
def normalize(a):
    """Normalize a nonzero three-dimensional vector in place.

    Parameters
    ----------
    a : ndarray
        Writable three-component floating-point vector with finite, nonzero
        Euclidean norm. The caller is responsible for this precondition.

    Returns
    -------
    None
        The vector is divided by its Euclidean norm, preserving its direction.
    """
    magnitude = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    a[0] /= magnitude
    a[1] /= magnitude
    a[2] /= magnitude


@njit
def make_direction_basis(px, py, pz):
    """Define the azimuthal basis for a normalized polar reference.

    Parameters
    ----------
    px, py, pz : float
        Components of a normalized polar reference vector.

    Returns
    -------
    tuple of float
        Six components ``(e1x, e1y, e1z, e2x, e2y, e2z)`` of the
        zero-azimuth and pi/2-azimuth unit basis vectors.

    Notes
    -----
    When the reference has a nonzero XY projection, e1 is normalized Z
    cross reference and e2 is reference cross e1. At either exact Z pole,
    e1 is positive X and e2 is positive Y. The basis is orthonormal and
    transverse in all cases; (e1, e2, reference) is left-handed at negative Z.
    """
    r = math.hypot(px, py)
    if r == 0.0:
        return 1.0, 0.0, 0.0, 0.0, 1.0, 0.0

    cx = px / r
    cy = py / r
    e1x = -cy
    e1y = cx
    e1z = 0.0
    e2x = -pz * cx
    e2y = -pz * cy
    e2z = r
    return e1x, e1y, e1z, e2x, e2y, e2z


@njit
def direction_from_angles(mu, azimuthal, polar_reference):
    """Construct a direction from polar cosine, azimuth, and reference axis."""
    px = polar_reference[0]
    py = polar_reference[1]
    pz = polar_reference[2]
    e1x, e1y, e1z, e2x, e2y, e2z = make_direction_basis(px, py, pz)

    transverse = math.sqrt(max(0.0, 1.0 - mu * mu))
    cos_azimuthal = math.cos(azimuthal)
    sin_azimuthal = math.sin(azimuthal)

    x = transverse * (cos_azimuthal * e1x + sin_azimuthal * e2x) + mu * px
    y = transverse * (cos_azimuthal * e1y + sin_azimuthal * e2y) + mu * py
    z = transverse * (cos_azimuthal * e1z + sin_azimuthal * e2z) + mu * pz
    return x, y, z


@njit
def rotation_matrix(rotation):
    """Construct a three-dimensional rotation matrix from angles in radians.

    Parameters
    ----------
    rotation : array_like
        Three angles ``(phi, theta, psi)`` about the X, Y, and Z axes.

    Returns
    -------
    tuple of float
        Nine components ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)`` in row-major
        order. Multiply this matrix by a column vector to rotate it.

    Notes
    -----
    The matrix is ``Rz(psi) Ry(theta) Rx(phi)``: right-handed rotations about
    fixed axes, applied in X, then Y, then Z order. The input is not modified.
    """
    phi = rotation[0]
    theta = rotation[1]
    psi = rotation[2]

    xx = math.cos(theta) * math.cos(psi)
    xy = -math.cos(phi) * math.sin(psi) + math.sin(phi) * math.sin(theta) * math.cos(
        psi
    )
    xz = math.sin(phi) * math.sin(psi) + math.cos(phi) * math.sin(theta) * math.cos(psi)

    yx = math.cos(theta) * math.sin(psi)
    yy = math.cos(phi) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
    yz = -math.sin(phi) * math.cos(psi) + math.cos(phi) * math.sin(theta) * math.sin(
        psi
    )

    zx = -math.sin(theta)
    zy = math.sin(phi) * math.cos(theta)
    zz = math.cos(phi) * math.cos(theta)

    return xx, xy, xz, yx, yy, yz, zx, zy, zz
