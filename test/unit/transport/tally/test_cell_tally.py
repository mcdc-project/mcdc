import numpy as np

import mcdc
from mcdc.main import preparation
from mcdc.transport.geometry import interface as geometry_interface


def _bin_value(surface_tally, mcdc_struct, data):
    tally_base = mcdc_struct["tallies"][surface_tally["parent_ID"]]
    return data[tally_base["bin_offset"]]


def _bin_value_score(surface_tally, mcdc_struct, data, score_idx):
    tally_base = mcdc_struct["tallies"][surface_tally["parent_ID"]]
    return data[tally_base["bin_offset"] + score_idx]


def test_surface_cell_filter_current_sign_for_incoming_and_outgoing(
    slab_plane_x, crossing_particle
):
    current_tally_obj = mcdc.Tally(cell=slab_plane_x["c_right"], scores=["net-current"])
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_tallies"][current_tally_obj.child_ID]

    # Left -> right across the shared interior surface: incoming to c_right (+)
    particle_container = crossing_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert np.isclose(_bin_value(current_tally, mcdc_struct, data), 2.0)

    # Right -> left across the shared interior surface: outgoing from c_right (-)
    particle_container = crossing_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=-0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert np.isclose(_bin_value(current_tally, mcdc_struct, data), 0.0)


def test_surface_cell_filter_current_records_in_and_out_separately(
    slab_plane_x, crossing_particle
):
    current_tally_obj = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        scores=["net-current", "current-in", "current-out"],
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_tallies"][current_tally_obj.child_ID]

    # One incoming and one outgoing crossing.
    particle_container = crossing_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    particle_container = crossing_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=-0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    # Scores are in requested order: [net-current, current-in, current-out].
    # Partial currents are positive; net-current carries the sign.
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 0), 0.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 1), 2.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 2), 2.0)


def test_surface_cell_filter_current_counts_outgoing_to_vacuum(
    slab_plane_x, crossing_particle
):
    current_tally_obj = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        scores=["net-current", "current-in", "current-out"],
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_tallies"][current_tally_obj.child_ID]

    # c_right -> vacuum across the right boundary: outgoing (-)
    particle_container = crossing_particle(slab_plane_x["s_right"].ID, x=1.0, ux=0.5)
    particle = particle_container[0]
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert not particle["alive"]
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 0), -2.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 1), 0.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 2), 2.0)


def test_surface_cell_filter_current_ignores_reflective_crossing(
    material_mg, crossing_particle
):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="reflective")
    c_mid = mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    current_tally_obj = mcdc.Tally(
        cell=c_mid,
        scores=["net-current", "current-in", "current-out"],
    )

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_tallies"][current_tally_obj.child_ID]

    particle_container = crossing_particle(s_right.ID, x=1.0, ux=0.5)
    particle = particle_container[0]
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert particle["alive"]
    assert np.isclose(particle["ux"], -0.5)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 0), 0.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 1), 0.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 2), 0.0)


def test_surface_cell_filter_current_scores_curved_boundary(
    material_mg, crossing_particle
):
    s_cyl = mcdc.Surface.CylinderZ(center=(0.0, 0.0), radius=1.0)
    c_inner = mcdc.Cell(region=-s_cyl, fill=material_mg)
    mcdc.Cell(region=+s_cyl, fill=material_mg)
    current_tally_obj = mcdc.Tally(
        cell=c_inner,
        scores=["net-current", "current-in", "current-out"],
    )

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_tallies"][current_tally_obj.child_ID]

    # Inner -> outer across the cylindrical surface: outgoing from c_inner (-)
    particle_container = crossing_particle(s_cyl.ID, x=1.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 0), -2.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 1), 0.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 2), 2.0)

    # Outer -> inner across the same curved surface: incoming to c_inner (+)
    particle_container = crossing_particle(s_cyl.ID, x=1.0, ux=-0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 0), 0.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 1), 2.0)
    assert np.isclose(_bin_value_score(current_tally, mcdc_struct, data, 2), 2.0)
