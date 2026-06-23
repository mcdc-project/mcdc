import pytest

import mcdc
from mcdc.constant import SPATIAL_FILTER_MESH
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
    assert tally.spatial_filter_type == SPATIAL_FILTER_MESH
    assert tally.spatial_filter_ID == mesh.ID


def test_collision_tally_requires_mesh(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["energy_deposition"])
    out = capsys.readouterr().out
    assert "currently only supported with a mesh spatial filter" in out
