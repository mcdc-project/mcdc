from numba import njit

from mcdc.util import binary_search_with_length
from mcdc.constant import COINCIDENCE_TOLERANCE, COINCIDENCE_TOLERANCE_TIME, INF


@njit
def get_indices(particle_container, mesh):
    """
    Get mesh indices given the particle coordinate
    """
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Mesh parameters
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]

    # Check if particle is outside the mesh grid
    outside = False
    if (
        # Outside the mesh condition
        x < mesh["x"][0] - COINCIDENCE_TOLERANCE
        or x > mesh["x"][Nx] + COINCIDENCE_TOLERANCE
        or y < mesh["y"][0] - COINCIDENCE_TOLERANCE
        or y > mesh["y"][Ny] + COINCIDENCE_TOLERANCE
        or z < mesh["z"][0] - COINCIDENCE_TOLERANCE
        or z > mesh["z"][Nz] + COINCIDENCE_TOLERANCE
        or t < mesh["t"][0] - COINCIDENCE_TOLERANCE_TIME
        or t > mesh["t"][Nt] + COINCIDENCE_TOLERANCE_TIME
        # At the outermost-grid but moving away
        or (abs(x - mesh["x"][0]) < COINCIDENCE_TOLERANCE and ux < 0.0)
        or (abs(x - mesh["x"][Nx]) < COINCIDENCE_TOLERANCE and ux > 0.0)
        or (abs(y - mesh["y"][0]) < COINCIDENCE_TOLERANCE and uy < 0.0)
        or (abs(y - mesh["y"][Ny]) < COINCIDENCE_TOLERANCE and uy > 0.0)
        or (abs(z - mesh["z"][0]) < COINCIDENCE_TOLERANCE and uz < 0.0)
        or (abs(z - mesh["z"][Nz]) < COINCIDENCE_TOLERANCE and uz > 0.0)
        or (abs(t - mesh["t"][Nt]) < COINCIDENCE_TOLERANCE_TIME)
    ):
        outside = True
        return -1, -1, -1, -1, outside

    ix = _grid_index(x, ux, mesh["x"], Nx + 1, COINCIDENCE_TOLERANCE)
    iy = _grid_index(y, uy, mesh["y"], Ny + 1, COINCIDENCE_TOLERANCE)
    iz = _grid_index(z, uz, mesh["z"], Nz + 1, COINCIDENCE_TOLERANCE)
    it = _grid_index(
        t, 1.0, mesh["t"], Nt + 1, COINCIDENCE_TOLERANCE_TIME
    )  # Particle always moves forward in time

    return ix, iy, iz, it, outside


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
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Mesh parameters
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]

    # Check if particle is outside the mesh grid and moving away
    outside = False
    if (
        (t > mesh["t"][Nt] - COINCIDENCE_TOLERANCE_TIME)
        or (x < mesh["x"][0] + COINCIDENCE_TOLERANCE and ux < 0.0)
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
    d = min(
        d, _grid_distance(t, 1.0 / speed, mesh["t"], Nt + 1, COINCIDENCE_TOLERANCE_TIME)
    )
    return d


@njit
def _grid_index(value, direction, grid, length, tolerance):
    """
    Get grid index given the value and the direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    Note: It assumes the value is inside the grid.
    """
    idx = binary_search_with_length(value, grid, length)

    # Coinciding cases
    if direction > 0.0:
        if abs(grid[idx + 1] - value) < tolerance:
            idx += 1
    else:
        if abs(grid[idx] - value) < tolerance:
            idx -= 1

    return idx


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
