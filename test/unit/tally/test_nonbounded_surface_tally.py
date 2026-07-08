import numpy as np
import pytest

import mcdc
from mcdc.main import preparation
from mcdc.transport.simulation import surface_crossing


def test_surface_crossing_tally_filter_fields(surface_crossing_tally_context):
    unbounded_tally = surface_crossing_tally_context["unbounded_tally"]
    s_mid = surface_crossing_tally_context["s_mid"]

    assert unbounded_tally["surface_filtered"]
    assert unbounded_tally["surface_filter_ID"] == s_mid.ID
    assert not unbounded_tally["cell_filtered"]
    assert unbounded_tally["cell_filter_ID"] == -1


@pytest.mark.parametrize(
    "ux, y, z, expected",
    [
        (0.5, 0.0, 0.0, 2.0),  # left -> right
        (-0.5, 0.0, 0.0, -2.0),  # right -> left
        (0.5, 0.8, 0.0, 2.0),
        (0.5, 0.0, 0.3, 2.0),
    ],
    ids=["left_to_right", "right_to_left", "shifted_y", "shifted_z"],
)
def test_unbounded_surface_crossing_tally_scoring(
    surface_crossing_tally_context,
    bin_value,
    ux,
    y,
    z,
    expected,
):
    data = surface_crossing_tally_context["data"]
    mcdc_struct = surface_crossing_tally_context["mcdc_struct"]
    particle = surface_crossing_tally_context["particle"]
    particle_container = surface_crossing_tally_context["particle_container"]
    s_mid = surface_crossing_tally_context["s_mid"]
    unbounded_tally = surface_crossing_tally_context["unbounded_tally"]

    particle["alive"] = True
    particle["surface_ID"] = s_mid.ID
    particle["x"] = 0.0
    particle["y"] = y
    particle["z"] = z
    particle["ux"] = ux
    surface_crossing(particle_container, mcdc_struct, data)

    np.testing.assert_allclose(bin_value(unbounded_tally, mcdc_struct, data), expected, rtol=1e-5, atol=1e-8)


def test_surface_crossing_tally_scores_vacuum_boundary(
    material_mg, bin_value, crossing_particle
):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")
    mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    tally_obj = mcdc.Tally(surface=s_right, scores=["current-net"])

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.child_ID]

    particle_container = crossing_particle(s_right.ID, x=1.0, ux=0.5)
    particle = particle_container[0]

    surface_crossing(particle_container, mcdc_struct, data)

    assert not particle["alive"]
    np.testing.assert_allclose(bin_value(tally, mcdc_struct, data), 2.0, rtol=1e-5, atol=1e-8)


def test_surface_crossing_tally_scores_after_reflective_boundary(
    material_mg, bin_value, crossing_particle
):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="reflective")
    mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    tally_obj = mcdc.Tally(surface=s_right, scores=["current-net"])

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.child_ID]

    particle_container = crossing_particle(s_right.ID, x=1.0, ux=0.5)
    particle = particle_container[0]

    surface_crossing(particle_container, mcdc_struct, data)

    assert particle["alive"]
    np.testing.assert_allclose(particle["ux"], -0.5, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(bin_value(tally, mcdc_struct, data), 0.0, rtol=1e-5, atol=1e-8)
