import numpy as np

import mcdc


def test_surfaces_are_available_without_compiling_geometry(monkeypatch):
    material = mcdc.Material.multigroup(capture=np.array([1.0]))
    lower = mcdc.Surface.PlaneX(x=0.0)
    upper = mcdc.Surface.PlaneX(x=1.0)
    cell = mcdc.Cell(region=+lower & -upper, fill=material)

    def unexpected_compilation(*args, **kwargs):
        raise AssertionError("Reading cell surfaces must not compile model objects")

    for object_ in (cell, lower, upper, material):
        monkeypatch.setattr(object_, "_compile_into_simulation", unexpected_compilation)

    assert {id(surface) for surface in cell.surfaces} == {id(lower), id(upper)}
    for object_ in (cell, lower, upper, material):
        assert object_.ID == -1
        assert object_.compile_ID == 0


def test_all_space_cell_has_no_surfaces():
    assert mcdc.Cell().surfaces == []


def test_boolean_regions_list_each_surface_identity_once():
    left = mcdc.Surface.PlaneX(x=0.0)
    coincident_left = mcdc.Surface.PlaneX(x=0.0)
    right = mcdc.Surface.PlaneX(x=1.0)
    cell = mcdc.Cell(region=(+left & -right) | ~(+left & -coincident_left))

    surfaces = cell.surfaces

    assert len(surfaces) == 3
    assert {id(surface) for surface in surfaces} == {
        id(left),
        id(coincident_left),
        id(right),
    }


def test_surfaces_follow_reassigned_region():
    original = mcdc.Surface.PlaneX(x=0.0)
    replacement = mcdc.Surface.PlaneY(y=1.0)
    cell = mcdc.Cell(region=+original)
    assert cell.surfaces == [original]

    cell.region = -replacement

    assert cell.surfaces == [replacement]


def test_surfaces_follow_in_place_nested_region_edits():
    original = mcdc.Surface.PlaneX(x=0.0)
    replacement = mcdc.Surface.PlaneY(y=1.0)
    nested = +original
    cell = mcdc.Cell(region=~nested)
    assert cell.surfaces == [original]

    nested.A = replacement

    assert cell.surfaces == [replacement]


def test_cell_repr_lists_surfaces_before_compilation():
    surface = mcdc.Surface.PlaneX(x=0.0, name="Left boundary")
    cell = mcdc.Cell(region=+surface)

    assert "Bounding surfaces: ['Left boundary']" in repr(cell)
    assert cell.compile_ID == 0
