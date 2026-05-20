import pytest

import mcdc
from mcdc.constant import SPATIAL_FILTER_NONE
from mcdc.object_.tally import (
    TallyCell,
    TallyCollision,
    TallySurface,
    TallyTracklength,
)


def test_tally_factory_routing_surface_vs_tracklength_vs_collision(slab_plane_x):
    surface_tally = mcdc.Tally(surface=slab_plane_x["s_mid"], scores=["net-current"])
    cell_tally = mcdc.Tally(cell=slab_plane_x["c_right"], scores=["net-current"])
    tracklength_tally = mcdc.Tally(scores=["flux"])

    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )
    collision_tally = mcdc.Tally(mesh=mesh, scores=["energy_deposition"])

    assert isinstance(surface_tally, TallySurface)
    assert isinstance(cell_tally, TallyCell)
    assert isinstance(tracklength_tally, TallyTracklength)
    assert isinstance(collision_tally, TallyCollision)
    assert tracklength_tally.spatial_filter_type == SPATIAL_FILTER_NONE


def test_tally_factory_routing_invalid_score_mix(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["flux", "energy_deposition"])
    out = capsys.readouterr().out
    assert "Cannot mix tracklength scores with collision ones." in out


def test_tally_factory_routing_invalid_net_current_selectors(slab_plane_x, capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=slab_plane_x["s_mid"],
            cell=slab_plane_x["c_right"],
            scores=["net-current"],
        )
    out = capsys.readouterr().out
    assert "must specify exactly one of surface or cell" in out

    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["net-current"])
    out = capsys.readouterr().out
    assert "Current scores need either a surface or a cell tally" in out


def test_surface_tally_rejects_current_in_out_scores(slab_plane_x, capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(surface=slab_plane_x["s_mid"], scores=["current-in"])
    out = capsys.readouterr().out
    assert "Surface tally currently supports only scores=['net-current']." in out
