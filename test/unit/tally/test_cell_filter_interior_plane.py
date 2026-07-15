import numpy as np

import mcdc
from mcdc.main import preparation
from mcdc.transport.simulation import surface_crossing


def _bin_value_score(tally, mcdc_struct, data, score_idx):
    tally_base = mcdc_struct["tallies"][tally["parent_ID"]]
    return data[tally_base["bin_offset"] + score_idx]


def test_cell_filter_ignores_redundant_interior_surface_crossing(
    material_mg, crossing_particle
):
    s3 = mcdc.Surface.PlaneX(x=3.0)
    s5 = mcdc.Surface.PlaneX(x=5.0)

    # This region is equivalent to x < 5, but the expression includes a surface at s3.
    c = mcdc.Cell(region=(-s5) | (-s3), fill=material_mg)
    mcdc.Cell(region=+s5, fill=material_mg)
    tally_obj = mcdc.Tally(
        cell=c,
        scores=["current-net", "current-in", "current-out"],
    )

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.child_ID]

    # The particle crosses s3, but it remains inside c before and after crossing
    particle_container = crossing_particle(s3.ID, x=3.0, ux=0.5, cell_ID=c.ID)
    surface_crossing(particle_container, mcdc_struct, data)

    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 0), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 1), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 2), 0.0, rtol=1e-5, atol=1e-8
    )


def _build_cube_with_interior_plane(material_mg):
    x_min = mcdc.Surface.PlaneX(x=-1.0)
    x_mid = mcdc.Surface.PlaneX(x=0.0)  # middle union between two halves
    x_max = mcdc.Surface.PlaneX(x=1.0)
    y_min = mcdc.Surface.PlaneY(y=-1.0)
    y_max = mcdc.Surface.PlaneY(y=1.0)
    z_min = mcdc.Surface.PlaneZ(z=-1.0)
    z_max = mcdc.Surface.PlaneZ(z=1.0)

    left = +x_min & -x_mid & +y_min & -y_max & +z_min & -z_max
    right = +x_mid & -x_max & +y_min & -y_max & +z_min & -z_max
    cube = mcdc.Cell(region=left | right, fill=material_mg)

    outside_region = -x_min | +x_max | -y_min | +y_max | -z_min | +z_max
    outside = mcdc.Cell(region=outside_region, fill=material_mg)

    return {"cube": cube, "outside": outside, "x_min": x_min, "x_mid": x_mid}


def _cube_current_tally(geom):
    tally_obj = mcdc.Tally(
        cell=geom["cube"],
        scores=["current-net", "current-in", "current-out"],
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.child_ID]
    return tally, mcdc_struct, data


def test_cell_filter_ignores_interior_plane_crossing(material_mg, crossing_particle):
    """
    A particle crossing the cube's interior plane while remaining inside the
    cube does not cross the cell boundary, so current-net/in/out must all be 0.

    This fails against the check_cell-based cell-current scoring:
    "was inside the filter cell" fires current-out, and "still inside the filter
    cell" fires current-in, so an interior crossing is double-counted
    (net = +w, in = +w, out = +w instead of 0).
    """
    geom = _build_cube_with_interior_plane(material_mg)
    tally, mcdc_struct, data = _cube_current_tally(geom)

    # The particle crosses the cube's interior plane but remains in the cube.
    particle_container = crossing_particle(
        geom["x_mid"].ID,
        x=0.0,
        ux=0.5,
        cell_ID=geom["cube"].ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)

    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 0), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 1), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 2), 0.0, rtol=1e-5, atol=1e-8
    )


def test_cell_filter_scores_real_face_entry(material_mg, crossing_particle):
    """
    Contrast case on the same geometry: entering the cube through a real outer
    face (x_min) from the outside cell is a genuine boundary crossing, so it
    must score current-in (+w) and current-net (-w, incoming-negative
    convention).
    """
    geom = _build_cube_with_interior_plane(material_mg)
    tally, mcdc_struct, data = _cube_current_tally(geom)

    # Entering through a real outer face should score incoming current.
    particle_container = crossing_particle(
        geom["x_min"].ID,
        x=-1.0,
        ux=0.5,
        cell_ID=geom["outside"].ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)

    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 0), -2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 1), 2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(tally, mcdc_struct, data, 2), 0.0, rtol=1e-5, atol=1e-8
    )
