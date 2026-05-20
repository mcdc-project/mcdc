import math
import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.literals as literals
import mcdc.transport.mesh as mesh
import mcdc.transport.physics as physics
import mcdc.transport.tally as tally_module
import mcdc.transport.util as util

from mcdc.constant import *
from mcdc.transport.geometry.surface import get_distance, check_sense, reflect

# ======================================================================================
# Geometry inspection
# ======================================================================================


@njit
def inspect_geometry(particle_container, simulation, data):
    """
    Full geometry inspection of the particle:
        - Set particle top cell and material IDs (if not lost)
        - Set surface ID (if surface hit)
        - Set particle boundary event (surface or lattice crossing, or lost)
        - Return distance to boundary (surface or lattice)
    """
    particle = particle_container[0]

    # Store particle global coordinate
    # (particle will be temporarily translated and rotated)
    x_global = particle["x"]
    y_global = particle["y"]
    z_global = particle["z"]
    t_global = particle["t"]
    ux_global = particle["ux"]
    uy_global = particle["uy"]
    uz_global = particle["uz"]
    speed = physics.particle_speed(particle_container, simulation, data)

    # Default returns
    distance = INF
    event = EVENT_NONE

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(
            particle_container, UNIVERSE_ROOT, simulation, data
        )

        # Particle is lost?
        if particle["cell_ID"] == -1:
            event = EVENT_LOST

    # The top cell
    cell = simulation["cells"][particle["cell_ID"]]

    # Recursively check cells until material cell is found (or the particle is lost)
    while event != EVENT_LOST:
        # Distance to nearest surface
        d_surface, surface_ID = distance_to_nearest_surface(
            particle_container, cell, simulation, data
        )

        # Check if smaller
        if d_surface < distance - COINCIDENCE_TOLERANCE:
            distance = d_surface
            event = EVENT_SURFACE_CROSSING
            particle["surface_ID"] = surface_ID

        # Check if coincident
        elif check_coincidence(d_surface, distance):
            # Add event if not there yet
            if not event & EVENT_SURFACE_CROSSING:
                event += EVENT_SURFACE_CROSSING
                particle["surface_ID"] = surface_ID
            # If surface crossing is already there, prioritize the outer surface ID

        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice

            # Apply translation
            if cell["fill_translated"]:
                particle["x"] -= cell["translation"][0]
                particle["y"] -= cell["translation"][1]
                particle["z"] -= cell["translation"][2]

            # Apply rotation
            if cell["fill_rotated"]:
                _rotate_particle(particle_container, cell["rotation"])

            # Universe cell?
            if cell["fill_type"] == FILL_UNIVERSE:
                # Get universe ID
                universe_ID = cell["fill_ID"]

            # Lattice cell?
            elif cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = simulation["lattices"][cell["fill_ID"]]

                # Distance to lattice grid
                d_lattice = mesh.uniform.get_crossing_distance(
                    particle_container, speed, lattice
                )

                # Check if smaller
                if d_lattice < distance - COINCIDENCE_TOLERANCE:
                    distance = d_lattice
                    event = EVENT_LATTICE_CROSSING
                    particle["surface_ID"] = -1

                # Check if coincident
                if check_coincidence(d_lattice, distance):
                    # Add event if not there yet
                    if not event & EVENT_LATTICE_CROSSING:
                        event += EVENT_LATTICE_CROSSING

                # Get universe
                ix, iy, iz = mesh.uniform.get_indices(particle_container, lattice)
                if ix == -1 or iy == -1 or iz == -1:
                    event = EVENT_LOST
                    continue
                universe_ID = int(
                    mcdc_get.lattice.universe_IDs(ix, iy, iz, lattice, data)
                )

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

            # Get inner cell
            cell_ID = get_cell(particle_container, universe_ID, simulation, data)
            if cell_ID > -1:
                cell = simulation["cells"][cell_ID]
            else:
                event = EVENT_LOST

    # Reassign the global coordinate
    particle["x"] = x_global
    particle["y"] = y_global
    particle["z"] = z_global
    particle["t"] = t_global
    particle["ux"] = ux_global
    particle["uy"] = uy_global
    particle["uz"] = uz_global

    # Report lost particle
    if event == EVENT_LOST:
        report_lost_particle(particle_container, simulation)

    # Assign particle event
    particle["event"] = event

    return distance


@njit
def locate_particle(particle_container, simulation, data):
    """
    Set particle cell and material IDs
    Return False if particle is lost

    This is similar to inspect_geometry, except that distance to nearest surface
    or/and lattice grid and the respective boundary event are not determined.
    """
    particle = particle_container[0]

    # Store particle global coordinate
    # (particle will be temporarily translated and rotated)
    x_global = particle["x"]
    y_global = particle["y"]
    z_global = particle["z"]
    t_global = particle["t"]
    ux_global = particle["ux"]
    uy_global = particle["uy"]
    uz_global = particle["uz"]

    particle_is_lost = False

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(
            particle_container, UNIVERSE_ROOT, simulation, data
        )

        # Particle is lost?
        if particle["cell_ID"] == -1:
            particle_is_lost = True

    # The top cell
    cell = simulation["cells"][particle["cell_ID"]]

    # Recursively check cells until material cell is found (or the particle is lost)
    while not particle_is_lost:
        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice

            # Apply translation
            if cell["fill_translated"]:
                particle["x"] -= cell["translation"][0]
                particle["y"] -= cell["translation"][1]
                particle["z"] -= cell["translation"][2]

            # Apply rotation
            if cell["fill_rotated"]:
                _rotate_particle(particle_container, cell["rotation"])

            # Universe cell?
            if cell["fill_type"] == FILL_UNIVERSE:
                # Get universe ID
                universe_ID = cell["fill_ID"]

            # Lattice cell?
            elif cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = simulation["lattices"][cell["fill_ID"]]

                # Get universe
                ix, iy, iz = mesh.uniform.get_indices(particle_container, lattice)
                if ix == -1 or iy == -1 or iz == -1:
                    particle_is_lost = True
                    continue
                universe_ID = int(
                    mcdc_get.lattice.universe_IDs(ix, iy, iz, lattice, data)
                )

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

            # Get inner cell
            cell_ID = get_cell(particle_container, universe_ID, simulation, data)
            if cell_ID > -1:
                cell = simulation["cells"][cell_ID]
            else:
                particle_is_lost = True

    # Reassign the global coordinate
    particle["x"] = x_global
    particle["y"] = y_global
    particle["z"] = z_global
    particle["t"] = t_global
    particle["ux"] = ux_global
    particle["uy"] = uy_global
    particle["uz"] = uz_global

    # Report lost particle
    if particle_is_lost:
        report_lost_particle(particle_container, simulation)

    return not particle_is_lost


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
    xx, xy, xz, yx, yy, yz, zx, zy, zz = _rotation_matrix(rotation)

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


@njit
def _rotation_matrix(rotation):
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


# ======================================================================================
# Particle locator
# ======================================================================================


@njit
def get_cell(particle_container, universe_ID, simulation, data):
    """
    Find and return particle cell ID in the given universe
    Return -1 if particle is lost
    """
    particle = particle_container[0]
    universe = simulation["universes"][universe_ID]

    # Check over all cells in the universe
    for i in range(universe["N_cell"]):
        cell_ID = int(mcdc_get.universe.cell_IDs(i, universe, data))
        cell = simulation["cells"][cell_ID]
        if check_cell(particle_container, cell, simulation, data):
            return cell_ID

    # Particle is not found
    return -1


@njit
def check_cell(particle_container, cell, simulation, data):
    """
    Check if the particle is inside the cell
    """
    particle = particle_container[0]

    # Access RPN data
    N_token = cell["region_RPN_tokens_length"]
    if N_token == 0:
        return True

    # Create local value array
    value = util.local_array(literals.rpn_evaluation_buffer_size(), np.bool_)
    N_value = 0

    # Particle parameters
    speed = physics.particle_speed(particle_container, simulation, data)

    # March forward through RPN tokens
    for idx in range(N_token):
        token = int(mcdc_get.cell.region_RPN_tokens(idx, cell, data))

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
    Report lost particle and terminate it
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
        candidate_surface_ID = int(mcdc_get.cell.surface_IDs(i, cell, data))
        surface = simulation["surfaces"][candidate_surface_ID]
        d = get_distance(particle_container, speed, surface, data)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]

    return distance, surface_ID


@njit
def surface_crossing(P_arr, simulation, data):
    P = P_arr[0]
    crossed_surface_ID = P["surface_ID"]

    surface = simulation["surfaces"][crossed_surface_ID]
    BC = surface["boundary_condition"]

    # Apply BC
    if BC == BC_VACUUM:
        P["alive"] = False
    elif BC == BC_REFLECTIVE:
        reflect(P_arr, surface)

    # Score tally
    for i in range(surface["N_tally"]):
        tally_ID = int(mcdc_get.surface.tally_IDs(i, surface, data))
        tally = simulation["surface_tallies"][tally_ID]

        # Optional bounded surface tally: score only if the crossing point is
        # within the specified in-plane bounds.
        if tally["filter_surface_bounds"]:
            if not _surface_tally_crossing_in_bounds(P, surface, tally):
                continue

        tally_module.score.surface_tally(P_arr, surface, tally, simulation, data)

    # Score cell net-current tallies tied to this crossed surface.
    if BC != BC_REFLECTIVE:
        if simulation["N_cell_tally"] > 0:
            pre_cell_ID, post_cell_ID = _get_crossing_top_cell_IDs(
                P_arr, simulation, data
            )
            _score_cell_current_tallies(
                P_arr, pre_cell_ID, pre_cell_ID, post_cell_ID, simulation, data
            )
            if post_cell_ID != pre_cell_ID:
                _score_cell_current_tallies(
                    P_arr, post_cell_ID, pre_cell_ID, post_cell_ID, simulation, data
                )

    # Need to check new cell later?
    if P["alive"] and not BC == BC_REFLECTIVE:
        P["cell_ID"] = -1
        P["material_ID"] = -1


@njit
def _get_crossing_top_cell_IDs(particle_container, simulation, data):
    P = particle_container[0]
    x = P["x"]
    y = P["y"]
    z = P["z"]
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    epsilon = 10.0 * COINCIDENCE_TOLERANCE

    P["x"] = x - epsilon * ux
    P["y"] = y - epsilon * uy
    P["z"] = z - epsilon * uz
    pre_cell_ID = get_cell(particle_container, UNIVERSE_ROOT, simulation, data)

    P["x"] = x + epsilon * ux
    P["y"] = y + epsilon * uy
    P["z"] = z + epsilon * uz
    post_cell_ID = get_cell(particle_container, UNIVERSE_ROOT, simulation, data)

    P["x"] = x
    P["y"] = y
    P["z"] = z

    return pre_cell_ID, post_cell_ID


@njit
def _score_cell_current_tallies(
    particle_container,
    cell_ID,
    pre_cell_ID,
    post_cell_ID,
    simulation,
    data,
):
    if cell_ID < 0:
        return

    cell = simulation["cells"][cell_ID]
    for i in range(cell["N_tally"]):
        tally_base_ID = int(mcdc_get.cell.tally_IDs(i, cell, data))
        tally_base = simulation["tallies"][tally_base_ID]
        if tally_base["child_type"] != TALLY_CELL:
            continue

        tally = simulation["cell_tallies"][tally_base["child_ID"]]
        tally_module.score.cell_tally(
            particle_container, tally, pre_cell_ID, post_cell_ID, simulation, data
        )


@njit
def _surface_tally_crossing_in_bounds(P, surface, tally):
    surface_type = surface["type"]

    if surface_type == SURFACE_PLANE_X:
        if tally["has_y_bounds"]:
            if P["y"] < tally["y_min"] or P["y"] > tally["y_max"]:
                return False
        if tally["has_z_bounds"]:
            if P["z"] < tally["z_min"] or P["z"] > tally["z_max"]:
                return False
        return True

    if surface_type == SURFACE_PLANE_Y:
        if tally["has_x_bounds"]:
            if P["x"] < tally["x_min"] or P["x"] > tally["x_max"]:
                return False
        if tally["has_z_bounds"]:
            if P["z"] < tally["z_min"] or P["z"] > tally["z_max"]:
                return False
        return True

    if surface_type == SURFACE_PLANE_Z:
        if tally["has_x_bounds"]:
            if P["x"] < tally["x_min"] or P["x"] > tally["x_max"]:
                return False
        if tally["has_y_bounds"]:
            if P["y"] < tally["y_min"] or P["y"] > tally["y_max"]:
                return False
        return True

    # Constructor validation should prevent non-planar bounded surface tallies.
    return True


# ======================================================================================
# Miscellanies
# ======================================================================================


@njit
def check_coincidence(value_1, value_2):
    """
    Check if two values are within coincidence tolerance
    """
    return abs(value_1 - value_2) < COINCIDENCE_TOLERANCE
