import pytest

import mcdc

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

    assert not tracklength_tally.cell_filtered
    assert tracklength_tally.cell_filter_ID == -1
    assert not tracklength_tally.mesh_filtered
    assert tracklength_tally.mesh_filter_ID == -1


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


def test_tally_factory_allows_combined_supported_filters(slab_plane_x):
    mesh = mcdc.MeshUniform(
        "mesh",
        x=(-1.0, 0.5, 2),
        y=(-1.0, 1.0, 1),
        z=(-1.0, 1.0, 1),
    )

    surface_cell_tally = mcdc.Tally(
        surface=slab_plane_x["s_mid"],
        cell=slab_plane_x["c_right"],
        scores=["current-net"],
    )
    assert surface_cell_tally.surface_filtered
    assert surface_cell_tally.surface_filter_ID == slab_plane_x["s_mid"].ID
    assert surface_cell_tally.cell_filtered
    assert surface_cell_tally.cell_filter_ID == slab_plane_x["c_right"].ID

    cell_mesh_tally = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        mesh=mesh,
        scores=["flux"],
    )
    assert cell_mesh_tally.cell_filtered
    assert cell_mesh_tally.cell_filter_ID == slab_plane_x["c_right"].ID
    assert cell_mesh_tally.mesh_filtered
    assert cell_mesh_tally.mesh_filter_ID == mesh.ID


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
    assert "needs surface or cell filter" in out


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
