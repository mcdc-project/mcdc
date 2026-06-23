import pytest

import mcdc


def test_surface_bounds_validation_wrong_axis_on_planex(slab_plane_x, capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(surface=slab_plane_x["s_mid"], x=[-1.0, 1.0], scores=["current-net"])
    out = capsys.readouterr().out
    assert "PlaneX surface tally bounds may only use y and/or z." in out


def test_surface_bounds_validation_bound_shape(slab_plane_x, capsys):
    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=slab_plane_x["s_mid"],
            y=[-1.0, 0.0, 1.0],
            scores=["current-net"],
        )
    out = capsys.readouterr().out
    assert "Surface tally bound y must have exactly two values [min, max]." in out


def test_surface_bounds_validation_nonplanar_surface(capsys, material_mg):
    s_cyl = mcdc.Surface.CylinderZ(center=(0.0, 0.0), radius=1.0)
    mcdc.Cell(region=-s_cyl, fill=material_mg)
    with pytest.raises(SystemExit):
        mcdc.Tally(surface=s_cyl, x=[-0.5, 0.5], scores=["current-net"])
    out = capsys.readouterr().out
    assert (
        "Bounded surface tally currently supports only PlaneX, PlaneY, and PlaneZ"
        in out
    )
