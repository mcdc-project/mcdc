import numpy as np
import pytest

from mcdc.transport.geometry import interface as geometry_interface


@pytest.mark.parametrize(
    "y, z, expected_bounded",
    [
        (-0.5, 0.0, 2.0),  # exactly at y lower bound: included
        (0.5, 0.0, 2.0),  # exactly at y upper bound: included
        (-0.500001, 0.0, 0.0),  # just outside y lower bound: excluded
        (0.500001, 0.0, 0.0),  # just outside y upper bound: excluded
        (0.0, -0.25, 2.0),  # exactly at z lower bound: included
        (0.0, 0.25, 2.0),  # exactly at z upper bound: included
        (0.0, -0.250001, 0.0),  # just outside z lower bound: excluded
        (0.0, 0.250001, 0.0),  # just outside z upper bound: excluded
    ],
    ids=[
        "y_at_lower",
        "y_at_upper",
        "y_outside_lower",
        "y_outside_upper",
        "z_at_lower",
        "z_at_upper",
        "z_outside_lower",
        "z_outside_upper",
    ],
)
def test_surface_bounds_edges(
    surface_crossing_tally_context, bin_value, y, z, expected_bounded
):
    data = surface_crossing_tally_context["data"]
    mcdc_struct = surface_crossing_tally_context["mcdc_struct"]
    particle = surface_crossing_tally_context["particle"]
    particle_container = surface_crossing_tally_context["particle_container"]
    s_mid = surface_crossing_tally_context["s_mid"]
    unbounded_tally = surface_crossing_tally_context["unbounded_tally"]
    bounded_tally = surface_crossing_tally_context["bounded_tally"]

    particle["alive"] = True
    particle["surface_ID"] = s_mid.ID
    particle["x"] = 0.0
    particle["y"] = y
    particle["z"] = z
    particle["ux"] = 0.5
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    assert np.isclose(bin_value(unbounded_tally, mcdc_struct, data), 2.0)
    assert np.isclose(bin_value(bounded_tally, mcdc_struct, data), expected_bounded)
