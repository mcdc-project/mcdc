from numba import njit

from mcdc import mcdc_get
from mcdc.constant import MESH_STRUCTURED, MESH_UNIFORM, MESH_RECTILINEAR_UNSTRUCTURED
import mcdc.transport.mesh.structured as structured
import mcdc.transport.mesh.uniform as uniform
import mcdc.transport.mesh.rectilinear_unstructured as rectilinear_unstructured


@njit
def get_indices(particle_container, mesh, simulation, data):
    mesh_type = mesh["sub_type"]
    mesh_ID = mesh["sub_ID"]

    if mesh_type == MESH_UNIFORM:
        uniform_mesh = simulation["uniform_meshes"][mesh_ID]
        return uniform.get_indices(particle_container, uniform_mesh)
    elif mesh_type == MESH_STRUCTURED:
        structured_mesh = simulation["structured_meshes"][mesh_ID]
        return structured.get_indices(particle_container, structured_mesh, data)

    return -1, -1, -1

@njit
def get_flat_index(particle_container, mesh, simulation, data):
    mesh_type = mesh["sub_type"]
    mesh_ID = mesh["sub_ID"]

    if mesh_type == MESH_UNIFORM:
        mesh = simulation["uniform_meshes"][mesh_ID]
        i_x, i_y, i_z = uniform.get_indices(particle_container, mesh)
        if i_x == -1 or i_y == -1 or i_z == -1:
            return -1
        return ((i_x * mesh["Ny"]) + i_y) * mesh["Nz"] + i_z
    elif mesh_type == MESH_STRUCTURED:
        mesh = simulation["structured_meshes"][mesh_ID]
        i_x, i_y, i_z = structured.get_indices(particle_container, mesh, data)
        if i_x == -1 or i_y == -1 or i_z == -1:
            return -1
        return ((i_x * mesh["Ny"]) + i_y) * mesh["Nz"] + i_z


@njit
def get_x(index, mesh, simulation, data):
    mesh_type = mesh["sub_type"]
    mesh_ID = mesh["sub_ID"]

    if mesh_type == MESH_UNIFORM:
        uniform_mesh = simulation["uniform_meshes"][mesh_ID]
        return uniform_mesh["x0"] + uniform_mesh["dx"] * index
    elif mesh_type == MESH_STRUCTURED:
        structured_mesh = simulation["structured_meshes"][mesh_ID]
        return mcdc_get.structured_mesh.x(index, structured_mesh, data)
    elif mesh_type == MESH_RECTILINEAR_UNSTRUCTURED:
        mesh = simulation["rectilinear_unstructured_meshes"][mesh_ID]
        return mcdc_get.rectilinear_unstructured_mesh.x_bounds(index, )
    return 0.0


@njit
def get_y(index, mesh, simulation, data):
    mesh_type = mesh["sub_type"]
    mesh_ID = mesh["sub_ID"]

    if mesh_type == MESH_UNIFORM:
        uniform_mesh = simulation["uniform_meshes"][mesh_ID]
        return uniform_mesh["y0"] + uniform_mesh["dy"] * index
    elif mesh_type == MESH_STRUCTURED:
        structured_mesh = simulation["structured_meshes"][mesh_ID]
        return mcdc_get.structured_mesh.y(index, structured_mesh, data)
    return 0.0


@njit
def get_z(index, mesh, simulation, data):
    mesh_type = mesh["sub_type"]
    mesh_ID = mesh["sub_ID"]

    if mesh_type == MESH_UNIFORM:
        uniform_mesh = simulation["uniform_meshes"][mesh_ID]
        return uniform_mesh["z0"] + uniform_mesh["dz"] * index
    elif mesh_type == MESH_STRUCTURED:
        structured_mesh = simulation["structured_meshes"][mesh_ID]
        return mcdc_get.structured_mesh.z(index, structured_mesh, data)
    return 0.0
