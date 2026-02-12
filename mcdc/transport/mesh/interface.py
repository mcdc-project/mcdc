from numba import njit

from mcdc import mcdc_get
from mcdc.constant import MESH_STRUCTURED, MESH_UNIFORM
import mcdc.transport.mesh.structured as structured
import mcdc.transport.mesh.uniform as uniform


@njit
def get_indices(particle_container, mesh_base, mcdc, data):
    mesh_type = mesh_base["child_type"]
    mesh_ID = mesh_base["child_ID"]

    if mesh_type == MESH_UNIFORM:
        mesh = mcdc["uniform_meshes"][mesh_ID]
        return uniform.get_indices(particle_container, mesh)
    elif mesh_type == MESH_STRUCTURED:
        mesh = mcdc["structured_meshes"][mesh_ID]
        return structured.get_indices(particle_container, mesh, data)

    return -1, -1, -1


@njit
def get_x(index, mesh_base, mcdc, data):
    mesh_type = mesh_base["child_type"]
    mesh_ID = mesh_base["child_ID"]

    if mesh_type == MESH_UNIFORM:
        mesh = mcdc["uniform_meshes"][mesh_ID]
        return mesh["x0"] + mesh["dx"] * index
    elif mesh_type == MESH_STRUCTURED:
        mesh = mcdc["structured_meshes"][mesh_ID]
        return mcdc_get.structured_mesh.x(index, mesh, data)
    return 0.0


@njit
def get_y(index, mesh_base, mcdc, data):
    mesh_type = mesh_base["child_type"]
    mesh_ID = mesh_base["child_ID"]

    if mesh_type == MESH_UNIFORM:
        mesh = mcdc["uniform_meshes"][mesh_ID]
        return mesh["y0"] + mesh["dy"] * index
    elif mesh_type == MESH_STRUCTURED:
        mesh = mcdc["structured_meshes"][mesh_ID]
        return mcdc_get.structured_mesh.y(index, mesh, data)
    return 0.0


@njit
def get_z(index, mesh_base, mcdc, data):
    mesh_type = mesh_base["child_type"]
    mesh_ID = mesh_base["child_ID"]

    if mesh_type == MESH_UNIFORM:
        mesh = mcdc["uniform_meshes"][mesh_ID]
        return mesh["z0"] + mesh["dz"] * index
    elif mesh_type == MESH_STRUCTURED:
        mesh = mcdc["structured_meshes"][mesh_ID]
        return mcdc_get.structured_mesh.z(index, mesh, data)
    return 0.0
