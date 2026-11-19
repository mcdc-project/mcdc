"""Shared geometric translation for surfaces, sources, and cell products."""

from numba import njit

import mcdc.numba_types as type_
import mcdc.transport.geometry.interface as geometry
import mcdc.transport.util as util

from mcdc.constant import COINCIDENCE_TOLERANCE_TIME, FILL_MATERIAL, INF
from mcdc.transport.linalg import rotation_matrix
from mcdc.transport.util import find_bin_with_rules

# ======================================================================================
# Motion lookup and translation
# ======================================================================================


@njit
def get_move_idx(t, object_, data):
    """Return the motion segment at time ``t``, choosing the later at a boundary."""
    offset = object_["move_time_grid_offset"]
    length = object_["move_time_grid_length"]

    time_grid = data[offset : offset + length]
    idx = find_bin_with_rules(t, time_grid, COINCIDENCE_TOLERANCE_TIME, False)

    # The final stationary segment also covers times beyond the INF sentinel.
    if idx < 0:
        return 0 if t < time_grid[0] else object_["N_move"] - 1
    return idx


@njit
def get_translation(t, object_, data, idx=-1):
    """Return cumulative translation in cm at simulation time ``t`` in seconds."""
    # Get index if helper not provided
    if idx < 0:
        idx = get_move_idx(t, object_, data)

    time_offset = object_["move_time_grid_offset"]
    translation_offset = object_["move_translations_offset"] + 3 * idx
    velocity_offset = object_["move_velocities_offset"] + 3 * idx

    dt = t - data[time_offset + idx]

    return (
        data[translation_offset] + data[velocity_offset] * dt,
        data[translation_offset + 1] + data[velocity_offset + 1] * dt,
        data[translation_offset + 2] + data[velocity_offset + 2] * dt,
    )


# ======================================================================================
# Product motion
# ======================================================================================


@njit
def move_product(product_container, incident_container, simulation, data):
    """Translate a delayed product with its precursor's cell motion.

    Follow the incident geometry hierarchy at the production time. Motion in
    each containing cell is expressed in that cell's surface coordinates and
    transformed back to global coordinates. Bounding surfaces are not modified.

    This is similar to ``geometry.locate_particle``, except that traversal starts
    from the known incident top cell and uses a copy of the birth-time state.
    It accumulates each containing cell's displacement and the rotation needed
    to express that displacement globally, without updating the incident cell
    or material IDs. Like location, it computes no surface or lattice crossing
    distances, unlike ``geometry.inspect_geometry``.
    """
    product = product_container[0]
    incident = incident_container[0]
    cell_ID = incident["cell_ID"]
    cell = simulation["cells"][cell_ID]
    t_birth = incident["t"]
    t_emit = product["t"]

    # Material-filled root cells need no coordinate transformation or traversal.
    if cell["fill_type"] == FILL_MATERIAL:
        dx, dy, dz = get_displacement(t_birth, t_emit, cell, data)
        product["x"] += dx
        product["y"] += dy
        product["z"] += dz
        return

    # Retrace the birth-time hierarchy using the same fill and cell-lookup helpers
    # as locate_particle; retain per-level motion and rotations along the way.
    local_container = util.local_array(1, type_.particle)
    local_container[0] = incident
    # Rotation from global coordinates to the current containing universe.
    xx, xy, xz = 1.0, 0.0, 0.0
    yx, yy, yz = 0.0, 1.0, 0.0
    zx, zy, zz = 0.0, 0.0, 1.0

    while cell_ID >= 0:
        cell = simulation["cells"][cell_ID]
        dx, dy, dz = get_displacement(t_birth, t_emit, cell, data)
        # Inverse rotation converts the local displacement to global coordinates.
        product["x"] += xx * dx + yx * dy + zx * dz
        product["y"] += xy * dx + yy * dy + zy * dz
        product["z"] += xz * dx + yz * dy + zz * dz
        if cell["fill_type"] == FILL_MATERIAL:
            return

        geometry._apply_fill_transform(local_container, cell)
        if cell["fill_rotated"]:
            a, b, c, d, e, f, g, h, i = rotation_matrix(cell["rotation"])
            xx, xy, xz, yx, yy, yz, zx, zy, zz = (
                a * xx + b * yx + c * zx,
                a * xy + b * yy + c * zy,
                a * xz + b * yz + c * zz,
                d * xx + e * yx + f * zx,
                d * xy + e * yy + f * zy,
                d * xz + e * yz + f * zz,
                g * xx + h * yx + i * zx,
                g * xy + h * yy + i * zy,
                g * xz + h * yz + i * zz,
            )
        universe_ID = geometry._enter_fill(local_container, cell, simulation, data)
        if universe_ID < 0:
            break
        cell_ID = geometry._get_cell(
            local_container, INF, universe_ID, simulation, data
        )

    raise ValueError("Delayed-product production cell could not be located")


@njit
def get_displacement(t_birth, t_emit, cell, data):
    """Return the cell's translation between production and delayed emission times."""
    if not cell["moving"]:
        return 0.0, 0.0, 0.0
    x_birth, y_birth, z_birth = get_translation(t_birth, cell, data)
    x_emit, y_emit, z_emit = get_translation(t_emit, cell, data)
    return x_emit - x_birth, y_emit - y_birth, z_emit - z_birth
