import math

from numba import njit, int64

import mcdc.adapt as adapt
import mcdc.src.mesh as mesh
import mcdc.src.physics as physics
import mcdc.src.surface as surface_
import mcdc.type_ as type_

from mcdc.adapt import for_cpu, for_gpu
from mcdc.constant import *


# ======================================================================================
# Geometry inspection
# ======================================================================================


@njit
def inspect_geometry(particle_container, mcdc):
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
    speed = physics.get_speed(particle_container, mcdc)

    # Default returns
    distance = INF
    event = EVENT_NONE

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(particle_container, UNIVERSE_ROOT, mcdc)

        # Particle is lost?
        if particle["cell_ID"] == -1:
            event = EVENT_LOST

    # The top cell
    cell = mcdc["cells"][particle["cell_ID"]]

    # Recursively check cells until material cell is found (or the particle is lost)
    while event != EVENT_LOST:
        # Distance to nearest surface
        d_surface, surface_ID = distance_to_nearest_surface(
            particle_container, cell, mcdc
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
                _rotate_particle(particle, cell["rotation"])

            # Universe cell?
            if cell["fill_type"] == FILL_UNIVERSE:
                # Get universe ID
                universe_ID = cell["fill_ID"]

            # Lattice cell?
            elif cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

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
                ix, iy, iz, it, outside = mesh.uniform.get_indices(
                    particle_container, lattice
                )
                if outside:
                    event = EVENT_LOST
                    continue
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

            # Get inner cell
            cell_ID = get_cell(particle_container, universe_ID, mcdc)
            if cell_ID > -1:
                cell = mcdc["cells"][cell_ID]
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
        # Cheaper then keeping track of vac bounds
        # reflecting boundaries are still tracked
        # needed in this function for hybrid methods
        if mcdc["technique"]["delta_tracking"]:
            particle["alive"] = False
        else:
            report_lost(particle_container)

    # Assign particle event
    particle["event"] = event

    return distance


@njit
def locate_particle(particle_container, mcdc):
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
        particle["cell_ID"] = get_cell(particle_container, UNIVERSE_ROOT, mcdc)

        # Particle is lost?
        if particle["cell_ID"] == -1:
            particle_is_lost = True

    # The top cell
    cell = mcdc["cells"][particle["cell_ID"]]

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
                _rotate_particle(particle, cell["rotation"])

            # Universe cell?
            if cell["fill_type"] == FILL_UNIVERSE:
                # Get universe ID
                universe_ID = cell["fill_ID"]

            # Lattice cell?
            elif cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

                # Get universe
                ix, iy, iz, it, outside = mesh.uniform.get_indices(
                    particle_container, lattice
                )
                if outside:
                    particle_is_lost = True
                    continue
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

            # Get inner cell
            cell_ID = get_cell(particle_container, universe_ID, mcdc)
            if cell_ID > -1:
                cell = mcdc["cells"][cell_ID]
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

    # Report lost particle if not delta tracking
    if particle_is_lost:
        # Cheaper then keeping track of vac bounds
        #       reflecting boundaries are still tracked
        if mcdc["technique"]["delta_tracking"]:
            particle["alive"] = False
        else:
            report_lost(particle_container)

    return not particle_is_lost


@nb.njit
def _rotate_particle(particle, rotation):
    # Particle initial coordinate
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


@nb.njit
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
def get_cell(particle_container, universe_ID, mcdc):
    """
    Find and return particle cell ID in the given universe
    Return -1 if particle is lost
    """
    particle = particle_container[0]
    universe = mcdc["universes"][universe_ID]

    # Access universe cell data
    idx = universe["cell_data_idx"]
    N_cell = universe["N_cell"]

    # Check over all cells in the universe
    idx_end = idx + N_cell
    while idx < idx_end:
        cell_ID = mcdc["universes_data_cell"][idx]
        cell = mcdc["cells"][cell_ID]
        if check_cell(particle_container, cell, mcdc):
            return cell["ID"]
        idx += 1

    # Particle is not found
    return -1


@njit
def check_cell(particle_container, cell, mcdc):
    """
    Check if the particle is inside the cell
    """
    particle = particle_container[0]

    # Access RPN data
    idx = cell["region_data_idx"]
    N_token = cell["N_region"]

    # No region description
    if N_token == 0:
        return True

    # Create local value array
    value = adapt.local_array(type_.rpn_buffer_size(), type_.bool_)
    N_value = 0

    # Particle parameters
    speed = physics.get_speed(particle_container, mcdc)

    # March forward through RPN tokens
    idx_end = idx + N_token
    while idx < idx_end:
        token = mcdc["cells_data_region"][idx]

        if token >= 0:
            surface = mcdc["surfaces"][token]
            value[N_value] = surface_.check_sense(particle_container, speed, surface)
            N_value += 1

        elif token == BOOL_NOT:
            value[N_value - 1] = not value[N_value - 1]

        elif token == BOOL_AND:
            value[N_value - 2] = value[N_value - 2] & value[N_value - 1]
            N_value -= 1

        elif token == BOOL_OR:
            value[N_value - 2] = value[N_value - 2] | value[N_value - 1]
            N_value -= 1

        idx += 1

    return value[0]


@for_cpu()
def report_lost(particle_container):
    """
    Report lost particle and terminate it
    """
    particle = particle_container[0]

    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    print("A particle is lost at (", x, y, z, t, ")")
    particle["alive"] = False


@for_gpu()
def report_lost(particle_container):
    particle = particle_container[0]

    particle["alive"] = False


# ======================================================================================
# Nearest distance search
# ======================================================================================


@njit
def distance_to_nearest_surface(particle_container, cell, mcdc):
    """
    Determine the nearest cell surface and the distance to it
    """
    particle = particle_container[0]
    distance = INF
    surface_ID = -1

    # Particle parameters
    speed = physics.get_speed(particle_container, mcdc)

    # Access cell surface data
    idx = cell["surface_data_idx"]
    N_surface = cell["N_surface"]

    # Iterate over all surfaces
    idx_end = idx + N_surface
    while idx < idx_end:
        candidate_surface_ID = mcdc["cells_data_surface"][idx]
        surface = mcdc["surfaces"][candidate_surface_ID]
        d = surface_.get_distance(particle_container, speed, surface)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]
        idx += 1
    return distance, surface_ID


@njit
def distance_to_boundary(particle_container, mcdc):
    """
    Determine the nearest cell surface and the distance to it
    """

    particle = particle_container[0]
    distance = INF
    surface_ID = -1

    # Particle parameters
    speed = physics.get_speed(particle_container, mcdc)
    N_surface = len(mcdc["boundary_surface_IDs"])  # array of ints

    for surf in range(N_surface):
        candidate_surface_ID = mcdc["boundary_surface_IDs"][surf]
        surface = mcdc["surfaces"][candidate_surface_ID]
        d = surface_.get_distance(particle_container, speed, surface)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]

    # if it's an int it never hits a surface!
    if distance is not INF:
        particle["surface_ID"] = surface_ID
        particle["event"] = EVENT_SURFACE_CROSSING

    return distance


# ======================================================================================
# Miscellanies
# ======================================================================================


@njit
def check_coincidence(value_1, value_2):
    """
    Check if two values are within coincidence tolerance
    """
    return abs(value_1 - value_2) < COINCIDENCE_TOLERANCE
