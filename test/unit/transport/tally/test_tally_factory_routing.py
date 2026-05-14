import pytest

import mcdc
from mcdc.constant import SPATIAL_FILTER_NONE
from mcdc.object_.tally import TallyCollision, TallySurface, TallyTracklength


def test_tally_factory_routing_surface_vs_tracklength_vs_collision(slab_plane_x):
    surface_tally = mcdc.Tally(surface=slab_plane_x["s_mid"], scores=["net-current"])
    tracklength_tally = mcdc.Tally(scores=["flux"])

    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )
    collision_tally = mcdc.Tally(mesh=mesh, scores=["energy_deposition"])

    assert isinstance(surface_tally, TallySurface)
    assert isinstance(tracklength_tally, TallyTracklength)
    assert isinstance(collision_tally, TallyCollision)
    assert tracklength_tally.spatial_filter_type == SPATIAL_FILTER_NONE


def test_tally_factory_routing_invalid_score_mix(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["flux", "energy_deposition"])
    out = capsys.readouterr().out
    assert "Cannot mix tracklength scores with collision ones." in out
