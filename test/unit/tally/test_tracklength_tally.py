import mcdc
from mcdc.object_.tally import TallyTracklength


def test_tracklength_tally_with_cell_filter(slab_plane_x):
    tally = mcdc.Tally(cell=slab_plane_x["c_left"], scores=["flux", "capture"])

    assert isinstance(tally, TallyTracklength)
    assert tally.cell_filtered
    assert tally.cell_filter_ID == slab_plane_x["c_left"].ID
    assert not tally.mesh_filtered
    assert tally.mesh_filter_ID == -1


def test_tracklength_tally_with_mesh_filter():
    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )
    tally = mcdc.Tally(mesh=mesh, scores=["flux"])

    assert isinstance(tally, TallyTracklength)
    assert not tally.cell_filtered
    assert tally.cell_filter_ID == -1
    assert tally.mesh_filtered
    assert tally.mesh_filter_ID == mesh.ID
    assert tally.mesh_stride_z == 1
    assert tally.mesh_stride_y == mesh.Nz
    assert tally.mesh_stride_x == mesh.Nz * mesh.Ny
