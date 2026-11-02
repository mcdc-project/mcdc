from numba import njit

####

import mcdc.mcdc_get.rectilinear_unstructured_mesh as mesh_get
from mcdc.constant import COINCIDENCE_TOLERANCE, INF

@njit
def get_indices(particle_container, mesh, data):
    particle = particle_container[0]
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    N_cells = mesh["N_cells"]

    for cell_ID in range(N_cells):
        x_lower, x_upper = mesh_get.x_bounds_vector(cell_ID, mesh, data)
        y_lower, y_upper = mesh_get.y_bounds_vector(cell_ID, mesh, data)
        z_lower, z_upper = mesh_get.z_bounds_vector(cell_ID, mesh, data)

        if not x_lower < x and x < x_upper:
            continue
        if not y_lower < y and y < y_upper:
            continue
        if not z_lower < z and z < z_upper:
            continue
        return cell_ID
    # not found
    return -1

