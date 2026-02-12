from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import COINCIDENCE_TOLERANCE, COINCIDENCE_TOLERANCE_TIME, INF
from mcdc.transport.util import find_bin


@njit
def get_indices(particle_container, mesh, data):
    """
    Get mesh indices given the particle coordinate
    """
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    tolerance = COINCIDENCE_TOLERANCE
    ix = find_bin(x, mcdc_get.structured_mesh.x_all(mesh, data), tolerance, ux < 0.0)
    iy = find_bin(y, mcdc_get.structured_mesh.y_all(mesh, data), tolerance, uy < 0.0)
    iz = find_bin(z, mcdc_get.structured_mesh.z_all(mesh, data), tolerance, uz < 0.0)

    return ix, iy, iz


@njit
def get_crossing_distance(particle_arr, speed, mesh):
    """
    Get distance for the particle, moving with the given speed,
    to cross the nearest grid of the mesh
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
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]

    # Check if particle is outside the mesh grid and moving away
    if (
        (x < mesh["x"][0] + COINCIDENCE_TOLERANCE and ux < 0.0)
        or (x > mesh["x"][Nx] - COINCIDENCE_TOLERANCE and ux > 0.0)
        or (y < mesh["y"][0] + COINCIDENCE_TOLERANCE and uy < 0.0)
        or (y > mesh["y"][Ny] - COINCIDENCE_TOLERANCE and uy > 0.0)
        or (z < mesh["z"][0] + COINCIDENCE_TOLERANCE and uz < 0.0)
        or (z > mesh["z"][Nz] - COINCIDENCE_TOLERANCE and uz > 0.0)
    ):
        return INF

    d = INF
    d = min(d, _grid_distance(x, ux, mesh["x"], Nx + 1, COINCIDENCE_TOLERANCE))
    d = min(d, _grid_distance(y, uy, mesh["y"], Ny + 1, COINCIDENCE_TOLERANCE))
    d = min(d, _grid_distance(z, uz, mesh["z"], Nz + 1, COINCIDENCE_TOLERANCE))
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
