import numpy as np
import pytest

import mcdc
from mcdc.main import preparation
from mcdc.transport.geometry import interface as geometry_interface


def test_surface_tally_bounds_fields(surface_tally_context):
    unbounded_tally = surface_tally_context["unbounded_tally"]
    bounded_tally = surface_tally_context["bounded_tally"]

    assert unbounded_tally["filter_surface_bounds"] == 0
    assert bounded_tally["filter_surface_bounds"] == 1
    assert bounded_tally["has_x_bounds"] == 0
    assert bounded_tally["has_y_bounds"] == 1
    assert bounded_tally["has_z_bounds"] == 1


@pytest.mark.parametrize(
    "ux, y, z, expected_unbounded, expected_bounded",
    [
        (0.5, 0.0, 0.0, 2.0, 2.0),  # inside bounds: left -> right
        (-0.5, 0.0, 0.0, -2.0, -2.0),  # inside bounds: right -> left
        (0.5, 0.8, 0.0, 2.0, 0.0),  # outside y bounds
        (0.5, 0.0, 0.3, 2.0, 0.0),  # outside z bounds
    ],
    ids=["inside_l2r", "inside_r2l", "outside_y_bounds", "outside_z_bounds"],
)
def test_unbounded_surface_tally_scoring(
    surface_tally_context,
    bin_value,
    ux,
    y,
    z,
    expected_unbounded,
    expected_bounded,
):
    data = surface_tally_context["data"]
    mcdc_struct = surface_tally_context["mcdc_struct"]
    particle = surface_tally_context["particle"]
    particle_container = surface_tally_context["particle_container"]
    s_mid = surface_tally_context["s_mid"]
    unbounded_tally = surface_tally_context["unbounded_tally"]
    bounded_tally = surface_tally_context["bounded_tally"]

    particle["alive"] = True
    particle["surface_ID"] = s_mid.ID
    particle["x"] = 0.0
    particle["y"] = y
    particle["z"] = z
    particle["ux"] = ux
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    assert np.isclose(bin_value(unbounded_tally, mcdc_struct, data), expected_unbounded)
    assert np.isclose(bin_value(bounded_tally, mcdc_struct, data), expected_bounded)


def test_surface_tally_scores_vacuum_boundary(
    material_mg, bin_value, crossing_particle
):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")
    mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    tally_obj = mcdc.Tally(surface=s_right, scores=["net-current"])

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_tallies"][tally_obj.child_ID]

    particle_container = crossing_particle(s_right.ID, x=1.0, ux=0.5)
    particle = particle_container[0]

    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    assert not particle["alive"]
    assert np.isclose(bin_value(tally, mcdc_struct, data), 2.0)


def test_surface_tally_scores_after_reflective_boundary(
    material_mg, bin_value, crossing_particle
):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="reflective")
    mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    tally_obj = mcdc.Tally(surface=s_right, scores=["net-current"])

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_tallies"][tally_obj.child_ID]

    particle_container = crossing_particle(s_right.ID, x=1.0, ux=0.5)
    particle = particle_container[0]

    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    assert particle["alive"]
    assert np.isclose(particle["ux"], -0.5)
    assert np.isclose(bin_value(tally, mcdc_struct, data), -2.0)
