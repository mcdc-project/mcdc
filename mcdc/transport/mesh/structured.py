from numba import njit

####

from mcdc.constant import COINCIDENCE_TOLERANCE, INF
from mcdc.transport.util import find_bin_with_rules


@njit
def get_indices(particle_container, structured_mesh, data):
    """
    Get structured_mesh indices given the particle coordinate
    """
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    grid_x = data[
        structured_mesh["x_offset"] : (
            structured_mesh["x_offset"] + structured_mesh["x_length"]
        )
    ]
    # Above is equivalent to: grid_x = mcdc_get.structured_mesh.x_all(structured_mesh, data)
    grid_y = data[
        structured_mesh["y_offset"] : (
            structured_mesh["y_offset"] + structured_mesh["y_length"]
        )
    ]
    # Above is equivalent to: grid_y = mcdc_get.structured_structured_mesh.y_all(structured_mesh, data)
    grid_z = data[
        structured_mesh["z_offset"] : (
            structured_mesh["z_offset"] + structured_mesh["z_length"]
        )
    ]
    # Above is equivalent to: grid_z = mcdc_get.structured_structured_mesh.z_all(structured_mesh, data)

    tolerance = COINCIDENCE_TOLERANCE
    ux_go_lower = ux < 0.0
    uy_go_lower = uy < 0.0
    uz_go_lower = uz < 0.0

    ix = find_bin_with_rules(x, grid_x, tolerance, ux_go_lower)
    iy = find_bin_with_rules(y, grid_y, tolerance, uy_go_lower)
    iz = find_bin_with_rules(z, grid_z, tolerance, uz_go_lower)

    return ix, iy, iz


@njit
def get_crossing_distance(particle_arr, speed, structured_mesh):
    """
    Get distance for the particle, moving with the given speed,
    to cross the nearest grid of the structured_mesh
    """
    particle = particle_arr[0]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Mesh parameters
    Nx = structured_mesh["Nx"]
    Ny = structured_mesh["Ny"]
    Nz = structured_mesh["Nz"]

    # Check if particle is outside the structured_mesh grid and moving away
    if (
        (x < structured_mesh["x"][0] + COINCIDENCE_TOLERANCE and ux < 0.0)
        or (x > structured_mesh["x"][Nx] - COINCIDENCE_TOLERANCE and ux > 0.0)
        or (y < structured_mesh["y"][0] + COINCIDENCE_TOLERANCE and uy < 0.0)
        or (y > structured_mesh["y"][Ny] - COINCIDENCE_TOLERANCE and uy > 0.0)
        or (z < structured_mesh["z"][0] + COINCIDENCE_TOLERANCE and uz < 0.0)
        or (z > structured_mesh["z"][Nz] - COINCIDENCE_TOLERANCE and uz > 0.0)
    ):
        return INF

    d = INF
    d = min(
        d, _grid_distance(x, ux, structured_mesh["x"], Nx + 1, COINCIDENCE_TOLERANCE)
    )
    d = min(
        d, _grid_distance(y, uy, structured_mesh["y"], Ny + 1, COINCIDENCE_TOLERANCE)
    )
    d = min(
        d, _grid_distance(z, uz, structured_mesh["z"], Nz + 1, COINCIDENCE_TOLERANCE)
    )
    return d


@njit
def _grid_distance(value, direction, grid, length, tolerance):
    """
    Get distance to nearest grid given a value and direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    Note: It assumes that a grid must be hit
    """
    if direction == 0.0:
        return INF

    idx = binary_search_with_length(value, grid, length)

    if direction > 0.0:
        idx += 1

    # Coinciding cases
    if abs(grid[idx] - value) < tolerance:
        if direction > 0.0:
            idx += 1
        else:
            idx -= 1

    dist = (grid[idx] - value) / direction

    return dist
