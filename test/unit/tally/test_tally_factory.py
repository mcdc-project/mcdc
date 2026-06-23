import pytest

import mcdc

from mcdc.constant import SPATIAL_FILTER_NONE
from mcdc.object_.tally import (
    TallyCollision,
    TallySurfaceCrossing,
    TallyTracklength,
)


def test_tally_factory_routing_surface_vs_tracklength_vs_collision(slab_plane_x):
    surface_crossing_tally = mcdc.Tally(
        surface=slab_plane_x["s_mid"],
        scores=["current-net"],
    )

    surface_cell_filter_tally = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        scores=["current-net"],
    )

    tracklength_tally = mcdc.Tally(scores=["flux"])

    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )

    collision_tally = mcdc.Tally(
        mesh=mesh,
        scores=["energy_deposition"],
    )

    assert isinstance(surface_crossing_tally, TallySurfaceCrossing)
    assert isinstance(surface_cell_filter_tally, TallySurfaceCrossing)
    assert isinstance(tracklength_tally, TallyTracklength)
    assert isinstance(collision_tally, TallyCollision)

    assert tracklength_tally.spatial_filter_type == SPATIAL_FILTER_NONE


def test_tally_factory_rejects_empty_scores(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=[])

    out = capsys.readouterr().out
    assert "Tally needs a score" in out


def test_tally_factory_rejects_unsupported_score(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["bad_score"])

    out = capsys.readouterr().out
    assert "Unsupported tally scores" in out
    assert "bad_score" in out


def test_tally_factory_rejects_multiple_spatial_filters(slab_plane_x, capsys):
    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )

    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=slab_plane_x["s_mid"],
            cell=slab_plane_x["c_right"],
            scores=["current-net"],
        )

    out = capsys.readouterr().out
    assert "Tally only supports one spatial filter" in out

    with pytest.raises(SystemExit):
        mcdc.Tally(
            cell=slab_plane_x["c_right"],
            mesh=mesh,
            scores=["flux"],
        )

    out = capsys.readouterr().out
    assert "Tally only supports one spatial filter" in out


def test_tally_factory_rejects_mixed_estimator_scores(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["flux", "energy_deposition"])

    out = capsys.readouterr().out
    assert "Cannot mix tally scores with different estimators" in out
    assert "flux" in out
    assert "energy_deposition" in out


def test_tally_factory_rejects_surface_crossing_without_surface_or_cell(capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(scores=["current-net"])

    out = capsys.readouterr().out
    assert "need either a surface or cell filter" in out


def test_tally_factory_rejects_tracklength_score_with_surface_filter(
    slab_plane_x,
    capsys,
):
    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=slab_plane_x["s_mid"],
            scores=["flux"],
        )

    out = capsys.readouterr().out
    assert "does not support surface filter" in out
    assert "flux" in out


def test_tally_factory_rejects_collision_score_with_surface_filter(
    slab_plane_x,
    capsys,
):
    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=slab_plane_x["s_mid"],
            scores=["energy_deposition"],
        )

    out = capsys.readouterr().out
    assert "does not support surface filter" in out
    assert "energy_deposition" in out
