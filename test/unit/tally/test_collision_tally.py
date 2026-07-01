import mcdc
from mcdc.object_.tally import TallyCollision


def test_collision_tally_with_mesh_filter():
    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )
    tally = mcdc.Tally(mesh=mesh, scores=["energy_deposition"])

    assert isinstance(tally, TallyCollision)
    assert not tally.cell_filtered
    assert tally.cell_filter_ID == -1
    assert tally.mesh_filtered
    assert tally.mesh_filter_ID == mesh.ID


def test_collision_tally_without_spatial_filter():
    tally = mcdc.Tally(scores=["energy_deposition"])

    assert isinstance(tally, TallyCollision)
    assert not tally.cell_filtered
    assert tally.cell_filter_ID == -1
    assert not tally.mesh_filtered
    assert tally.mesh_filter_ID == -1
