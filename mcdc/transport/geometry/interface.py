import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.literals as literals
import mcdc.numba_types as type_
import mcdc.transport.mesh as mesh
import mcdc.transport.physics as physics
import mcdc.transport.util as util
from mcdc.transport.linalg import rotation_matrix

from mcdc.constant import *
from mcdc.transport.geometry.surface import (
    get_distance,
    check_sense,
    get_normal_component,
)

# ======================================================================================
# Geometry traversal
# ======================================================================================


@njit
def inspect_geometry(particle_container, simulation, data):
    """
    Full geometry inspection of the particle:
        - Set particle top cell and material IDs (if not lost)
        - Set surface ID (if surface hit)
        - Set EVENT_LOST if lost; otherwise leave the event unchanged
        - Return distance to boundary (surface or lattice)
    """
    particle = particle_container[0]

    # Preserve global coordinates while traversing nested geometry.
    global_coordinates = _save_global_coordinates(particle_container)
    speed = physics.particle_speed(particle_container, simulation, data)

    # Track the nearest boundary and whether the particle is lost.
    distance = INF
    particle_is_lost = False
    particle["surface_ID"] = -1

    # Find the top cell from the root universe if it is unknown.
    cell_ID = _get_top_cell_ID(particle_container, speed, simulation, data)
    if cell_ID == -1:
        particle_is_lost = True

    # Recursively check cells until material cell is found (or the particle is lost)
    while not particle_is_lost:
        cell = simulation["cells"][cell_ID]

        # Distance to nearest surface
        d_surface, surface_ID = distance_to_nearest_surface(
            particle_container, cell, simulation, data
        )

        # Check if smaller
        if d_surface < distance - COINCIDENCE_TOLERANCE:
            distance = d_surface
            particle["surface_ID"] = surface_ID

        # Check if coincident
        elif surface_ID >= 0 and check_coincidence(d_surface, distance):
            # Retain surface actions when coincident with a lattice boundary.
            if particle["surface_ID"] == -1:
                particle["surface_ID"] = surface_ID
            # If surface crossing is already there, prioritize the outer surface ID

        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice
            _apply_fill_transform(particle_container, cell)

            # Lattice cell?
            if cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = simulation["lattices"][cell["fill_ID"]]

                # Distance to lattice grid
                d_lattice = mesh.uniform.get_crossing_distance(
                    particle_container, speed, lattice
                )

                # Check if smaller
                if d_lattice < distance - COINCIDENCE_TOLERANCE:
                    distance = d_lattice
                    particle["surface_ID"] = -1

            # Find the filled universe and enter its local coordinates.
            universe_ID = _enter_fill(particle_container, cell, simulation, data)
            if universe_ID == -1:
                particle_is_lost = True
                continue

            # Get inner cell
            cell_ID = _get_cell(
                particle_container, speed, universe_ID, simulation, data
            )
            if cell_ID == -1:
                particle_is_lost = True

    # Restore the particle after traversal through local coordinates.
    _restore_global_coordinates(particle_container, global_coordinates)

    # Report lost particle
    if particle_is_lost:
        particle["event"] = EVENT_LOST
        report_lost_particle(particle_container, simulation)

    return distance


@njit
def locate_particle(particle_container, simulation, data):
    """
    Set particle cell and material IDs
    Return False if particle is lost

    This is similar to inspect_geometry, except that distance to nearest surface
    or/and lattice grid are not determined.
    """
    particle = particle_container[0]

    # Preserve global coordinates while traversing nested geometry.
    global_coordinates = _save_global_coordinates(particle_container)

    # Use direction alone to resolve surface coincidence during location.
    # Material-dependent speed is unavailable until location is complete.
    direction_only_speed = INF
    particle_is_lost = False

    # Find the top cell from the root universe if it is unknown.
    cell_ID = _get_top_cell_ID(
        particle_container, direction_only_speed, simulation, data
    )
    if cell_ID == -1:
        particle_is_lost = True

    # Recursively check cells until material cell is found (or the particle is lost)
    while not particle_is_lost:
        cell = simulation["cells"][cell_ID]

        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice
            _apply_fill_transform(particle_container, cell)

            # Find the filled universe and enter its local coordinates.
            universe_ID = _enter_fill(particle_container, cell, simulation, data)
            if universe_ID == -1:
                particle_is_lost = True
                continue

            # Get inner cell
            cell_ID = _get_cell(
                particle_container,
                direction_only_speed,
                universe_ID,
                simulation,
                data,
            )
            if cell_ID == -1:
                particle_is_lost = True

    # Restore the particle after traversal through local coordinates.
    _restore_global_coordinates(particle_container, global_coordinates)

    # Report lost particle
    if particle_is_lost:
        report_lost_particle(particle_container, simulation)

    return not particle_is_lost


# ======================================================================================
# Geometry traversal helpers
# ======================================================================================


@njit
def surface_crossing_valid(
    incident_container, particle_container, distance, simulation, data
):
    """Check whether deflection preserves crossing in the incident surface frame."""
    # Retrace the incident hierarchy from the segment start. Use that same
    # frame for the endpoint, even if its new direction selects another cell.
    start_container = util.local_array(1, type_.particle)
    end_container = util.local_array(1, type_.particle)
    # Access through the arrays to keep their storage alive across Numba calls.
    start_container[0] = incident_container[0]
    end_container[0] = particle_container[0]
    incident_speed = physics.particle_speed(incident_container, simulation, data)
    outgoing_speed = physics.particle_speed(particle_container, simulation, data)
    start_container[0]["x"] -= distance * start_container[0]["ux"]
    start_container[0]["y"] -= distance * start_container[0]["uy"]
    start_container[0]["z"] -= distance * start_container[0]["uz"]
    start_container[0]["t"] -= distance / incident_speed
    surface_ID = start_container[0]["surface_ID"]
    cell_ID = start_container[0]["cell_ID"]

    while cell_ID != -1:
        cell = simulation["cells"][cell_ID]
        for i in range(cell["N_surface"]):
            if mcdc_get.cell.surface_IDs(i, cell, data) == surface_ID:
                surface = simulation["surfaces"][surface_ID]
                outgoing_normal = get_normal_component(
                    end_container, outgoing_speed, surface, data
                )
                end_container[0]["ux"] = start_container[0]["ux"]
                end_container[0]["uy"] = start_container[0]["uy"]
                end_container[0]["uz"] = start_container[0]["uz"]
                incident_normal = get_normal_component(
                    end_container, incident_speed, surface, data
                )
                return incident_normal * outgoing_normal > 0.0

        if cell["fill_type"] == FILL_MATERIAL:
            break
        _apply_fill_transform(start_container, cell)
        _apply_fill_transform(end_container, cell)
        x = start_container[0]["x"]
        y = start_container[0]["y"]
        z = start_container[0]["z"]
        universe_ID = _enter_fill(start_container, cell, simulation, data)
        if universe_ID == -1:
            break
        # Apply the incident lattice element's translation to the endpoint.
        end_container[0]["x"] += start_container[0]["x"] - x
        end_container[0]["y"] += start_container[0]["y"] - y
        end_container[0]["z"] += start_container[0]["z"] - z
        cell_ID = _get_cell(
            start_container, incident_speed, universe_ID, simulation, data
        )

    raise ValueError("Crossing surface not found in the incident geometry")


@njit
def _save_global_coordinates(particle_container):
    particle = particle_container[0]
    return (
        particle["x"],
        particle["y"],
        particle["z"],
        particle["t"],
        particle["ux"],
        particle["uy"],
        particle["uz"],
    )


@njit
def _restore_global_coordinates(particle_container, coordinates):
    particle = particle_container[0]
    particle["x"] = coordinates[0]
    particle["y"] = coordinates[1]
    particle["z"] = coordinates[2]
    particle["t"] = coordinates[3]
    particle["ux"] = coordinates[4]
    particle["uy"] = coordinates[5]
    particle["uz"] = coordinates[6]


@njit
def _get_top_cell_ID(particle_container, speed, simulation, data):
    particle = particle_container[0]
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = _get_cell(
            particle_container, speed, UNIVERSE_ROOT, simulation, data
        )
    return particle["cell_ID"]


@njit
def _apply_fill_transform(particle_container, cell):
    particle = particle_container[0]

    if cell["fill_translated"]:
        particle["x"] -= cell["translation"][0]
        particle["y"] -= cell["translation"][1]
        particle["z"] -= cell["translation"][2]

    if cell["fill_rotated"]:
        _rotate_particle(particle_container, cell["rotation"])


@njit
def _enter_fill(particle_container, cell, simulation, data):
    if cell["fill_type"] == FILL_UNIVERSE:
        return cell["fill_ID"]

    if cell["fill_type"] == FILL_LATTICE:
        particle = particle_container[0]
        lattice = simulation["lattices"][cell["fill_ID"]]
        ix, iy, iz = mesh.uniform.get_indices(particle_container, lattice)
        if ix == -1 or iy == -1 or iz == -1:
            return -1

        universe_ID = mcdc_get.lattice.universe_IDs(ix, iy, iz, lattice, data)
        particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
        particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
        particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]
        return universe_ID

    return -1


@njit
def _rotate_particle(particle_container, rotation):
    # Particle initial coordinate
    particle = particle_container[0]
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Rotation matrix
    xx, xy, xz, yx, yy, yz, zx, zy, zz = rotation_matrix(rotation)

    # Rotate
    x_rotated = x * xx + y * xy + z * xz
    y_rotated = x * yx + y * yy + z * yz
    z_rotated = x * zx + y * zy + z * zz
    ux_rotated = ux * xx + uy * xy + uz * xz
    uy_rotated = ux * yx + uy * yy + uz * yz
    uz_rotated = ux * zx + uy * zy + uz * zz

    # Assign the rotated coordinate
    particle["x"] = x_rotated
    particle["y"] = y_rotated
    particle["z"] = z_rotated
    particle["ux"] = ux_rotated
    particle["uy"] = uy_rotated
    particle["uz"] = uz_rotated


# ======================================================================================
# Particle locator
# ======================================================================================


@njit
def get_cell(particle_container, universe_ID, simulation, data):
    """
    Find and return particle cell ID in the given universe
    Return -1 if particle is lost
    """
    speed = physics.particle_speed(particle_container, simulation, data)
    return _get_cell(particle_container, speed, universe_ID, simulation, data)


@njit
def _get_cell(particle_container, speed, universe_ID, simulation, data):
    """Find the particle cell using the supplied speed for coincidence checks."""
    universe = simulation["universes"][universe_ID]

    # Check over all cells in the universe
    for i in range(universe["N_cell"]):
        cell_ID = mcdc_get.universe.cell_IDs(i, universe, data)
        cell = simulation["cells"][cell_ID]
        if _check_cell(particle_container, speed, cell, simulation, data):
            return cell_ID

    # Particle is not found
    return -1


@njit
def check_cell(particle_container, cell, simulation, data):
    """
    Check if the particle is inside the cell
    """
    speed = physics.particle_speed(particle_container, simulation, data)
    return _check_cell(particle_container, speed, cell, simulation, data)


@njit
def _check_cell(particle_container, speed, cell, simulation, data):
    """Check cell membership using the supplied speed for coincidence checks."""
    # Access RPN data
    N_token = cell["region_RPN_tokens_length"]
    if N_token == 0:
        return True

    # Create local value array
    value = util.local_array(literals.rpn_evaluation_buffer_size(), np.bool_)
    N_value = 0

    # March forward through RPN tokens
    for idx in range(N_token):
        token = mcdc_get.cell.region_RPN_tokens(idx, cell, data)

        if token >= 0:
            surface = simulation["surfaces"][token]
            value[N_value] = check_sense(particle_container, speed, surface, data)
            N_value += 1

        elif token == BOOL_NOT:
            value[N_value - 1] = not value[N_value - 1]

        elif token == BOOL_AND:
            value[N_value - 2] = value[N_value - 2] & value[N_value - 1]
            N_value -= 1

        elif token == BOOL_OR:
            value[N_value - 2] = value[N_value - 2] | value[N_value - 1]
            N_value -= 1

    return value[0]


@njit
def report_lost_particle(particle_container, simulation):
    """
    Report lost particle and terminate it.

    For GPU support, update report_lost_particle in
    mcdc/code_factory/gpu/transport/geometry/interface.py as well.
    """
    particle = particle_container[0]

    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    idx_batch = simulation["idx_batch"]
    idx_census = simulation["idx_census"]
    idx_work = simulation["idx_work"]
    print("A particle is lost at (", x, y, z, t, ")")
    print("  (batch/census/work) indices: (", idx_batch, idx_census, idx_work, ")")
    particle["alive"] = False


# ======================================================================================
# Nearest distance search
# ======================================================================================


@njit
def distance_to_nearest_surface(particle_container, cell, simulation, data):
    """
    Determine the nearest cell surface and the distance to it
    """
    distance = INF
    surface_ID = -1

    # Particle parameters
    speed = physics.particle_speed(particle_container, simulation, data)

    # Iterate over all surfaces and find the minimum distance
    for i in range(cell["N_surface"]):
        candidate_surface_ID = mcdc_get.cell.surface_IDs(i, cell, data)
        surface = simulation["surfaces"][candidate_surface_ID]
        d = get_distance(particle_container, speed, surface, data)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]

    return distance, surface_ID


# ======================================================================================
# Miscellanies
# ======================================================================================


@njit
def check_coincidence(value_1, value_2):
    """
    Check if two values are within coincidence tolerance
    """
    return abs(value_1 - value_2) < COINCIDENCE_TOLERANCE
