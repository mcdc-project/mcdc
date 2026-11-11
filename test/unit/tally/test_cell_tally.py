import numpy as np

import mcdc

from mcdc.transport.simulation import surface_crossing


def _bin_value(surface_crossing_tally, mcdc_struct, data):
    tally_base = mcdc_struct["tallies"][surface_crossing_tally["base_ID"]]
    return data[tally_base["bin_offset"]]


def _bin_value_score(surface_crossing_tally, mcdc_struct, data, score_idx):
    tally_base = mcdc_struct["tallies"][surface_crossing_tally["base_ID"]]
    return data[tally_base["bin_offset"] + score_idx]


def test_box_current_tally_before_compilation_scores_each_face_once(
    material_mg, crossing_particle, prepare_simulation
):
    lower = np.array([3.2, 3.65, 2.445])
    upper = np.array([6.8, 6.35, 3.045])
    surfaces = [
        mcdc.Surface.PlaneX(x=lower[0]),
        mcdc.Surface.PlaneX(x=upper[0]),
        mcdc.Surface.PlaneY(y=lower[1]),
        mcdc.Surface.PlaneY(y=upper[1]),
        mcdc.Surface.PlaneZ(z=lower[2]),
        mcdc.Surface.PlaneZ(z=upper[2]),
    ]
    x0, x1, y0, y1, z0, z1 = surfaces
    # Repeating a half-space must not attach the tally twice to that surface.
    region = +x0 & -x1 & +y0 & -y1 & +z0 & -z1 & +x0
    cell = mcdc.Cell(region=region, fill=material_mg)
    outside = mcdc.Cell(region=~region, fill=material_mg)
    tally_obj = mcdc.Tally(
        name="computer-current",
        cell=cell,
        scores=["current-net", "current-in", "current-out"],
    )
    assert cell.ID == -1
    assert all(surface.ID == -1 for surface in surfaces)

    mcdc_container, data = prepare_simulation(
        cells=[cell, outside], tallies=[tally_obj]
    )
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.sub_ID]
    assert mcdc_struct["cells"][cell.ID]["N_surface"] == 6
    assert all(surface.surface_crossing_tallies == [tally_obj] for surface in surfaces)

    for face, surface in enumerate(surfaces):
        axis = face // 2
        position = (lower + upper) / 2
        position[axis] = lower[axis] if face % 2 == 0 else upper[axis]
        inward = np.zeros(3)
        inward[axis] = 1.0 if face % 2 == 0 else -1.0

        for direction, from_cell in ((inward, outside), (-inward, cell)):
            particle_container = crossing_particle(
                surface.ID,
                x=position[0],
                ux=direction[0],
                cell_ID=from_cell.ID,
            )
            particle = particle_container[0]
            particle["y"], particle["z"] = position[1:]
            particle["uy"], particle["uz"] = direction[1:]
            surface_crossing(particle_container, mcdc_struct, data)

        np.testing.assert_allclose(
            [_bin_value_score(tally, mcdc_struct, data, i) for i in range(3)],
            [0.0, 2.0 * (face + 1), 2.0 * (face + 1)],
        )


def test_surface_cell_filter_current_net_is_incoming_negative_outgoing_positive(
    slab_plane_x, crossing_particle, prepare_simulation
):
    current_tally_obj = mcdc.Tally(cell=slab_plane_x["c_right"], scores=["current-net"])
    mcdc_container, data = prepare_simulation(
        cells=[slab_plane_x["c_left"], slab_plane_x["c_right"]],
        tallies=[current_tally_obj],
    )
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_crossing_tallies"][current_tally_obj.sub_ID]

    # Left -> right across the shared interior surface: incoming to c_right (-)
    particle_container = crossing_particle(
        slab_plane_x["s_mid"].ID,
        x=0.0,
        ux=0.5,
        cell_ID=slab_plane_x["c_left"].ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)
    np.testing.assert_allclose(
        _bin_value(current_tally, mcdc_struct, data), -2.0, rtol=1e-5, atol=1e-8
    )

    # Right -> left across the shared interior surface: outgoing from c_right (+)
    particle_container = crossing_particle(
        slab_plane_x["s_mid"].ID,
        x=0.0,
        ux=-0.5,
        cell_ID=slab_plane_x["c_right"].ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)
    np.testing.assert_allclose(
        _bin_value(current_tally, mcdc_struct, data), 0.0, rtol=1e-5, atol=1e-8
    )


def test_surface_cell_filter_current_records_in_and_out_separately(
    slab_plane_x, crossing_particle, prepare_simulation
):
    current_tally_obj = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        scores=["current-net", "current-in", "current-out"],
    )
    mcdc_container, data = prepare_simulation(
        cells=[slab_plane_x["c_left"], slab_plane_x["c_right"]],
        tallies=[current_tally_obj],
    )
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_crossing_tallies"][current_tally_obj.sub_ID]

    # One incoming and one outgoing crossing.
    particle_container = crossing_particle(
        slab_plane_x["s_mid"].ID,
        x=0.0,
        ux=0.5,
        cell_ID=slab_plane_x["c_left"].ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)
    particle_container = crossing_particle(
        slab_plane_x["s_mid"].ID,
        x=0.0,
        ux=-0.5,
        cell_ID=slab_plane_x["c_right"].ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)

    # Scores are in requested order: [current-net, current-in, current-out].
    # Partial currents are positive; current-net is incoming-negative/outgoing-positive.
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 0), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 1), 2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 2), 2.0, rtol=1e-5, atol=1e-8
    )


def test_surface_cell_filter_current_counts_outgoing_to_vacuum(
    slab_plane_x, crossing_particle, prepare_simulation
):
    current_tally_obj = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        scores=["current-net", "current-in", "current-out"],
    )
    mcdc_container, data = prepare_simulation(
        cells=[slab_plane_x["c_left"], slab_plane_x["c_right"]],
        tallies=[current_tally_obj],
    )
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_crossing_tallies"][current_tally_obj.sub_ID]

    # c_right -> vacuum across the right boundary: outgoing (+)
    particle_container = crossing_particle(
        slab_plane_x["s_right"].ID,
        x=1.0,
        ux=0.5,
        cell_ID=slab_plane_x["c_right"].ID,
    )
    particle = particle_container[0]
    surface_crossing(particle_container, mcdc_struct, data)
    assert not particle["alive"]
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 0), 2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 1), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 2), 2.0, rtol=1e-5, atol=1e-8
    )


def test_surface_cell_filter_current_ignores_reflective_crossing(
    material_mg, crossing_particle, prepare_simulation
):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="reflective")
    c_mid = mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    simulation = mcdc.Simulation()
    simulation.set_model([c_mid])
    simulation.compile()
    current_tally_obj = mcdc.Tally(
        cell=c_mid,
        scores=["current-net", "current-in", "current-out"],
    )

    mcdc_container, data = prepare_simulation(
        cells=[c_mid],
        tallies=[current_tally_obj],
    )
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_crossing_tallies"][current_tally_obj.sub_ID]

    particle_container = crossing_particle(s_right.ID, x=1.0, ux=0.5)
    particle = particle_container[0]
    surface_crossing(particle_container, mcdc_struct, data)
    assert particle["alive"]
    np.testing.assert_allclose(particle["ux"], -0.5, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 0), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 1), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 2), 0.0, rtol=1e-5, atol=1e-8
    )


def test_surface_cell_filter_current_scores_curved_boundary(
    material_mg, crossing_particle, prepare_simulation
):
    s_cyl = mcdc.Surface.CylinderZ(center=(0.0, 0.0), radius=1.0)
    c_inner = mcdc.Cell(region=-s_cyl, fill=material_mg)
    c_outer = mcdc.Cell(region=+s_cyl, fill=material_mg)
    simulation = mcdc.Simulation()
    simulation.set_model([c_inner, c_outer])
    simulation.compile()
    current_tally_obj = mcdc.Tally(
        cell=c_inner,
        scores=["current-net", "current-in", "current-out"],
    )

    mcdc_container, data = prepare_simulation(
        cells=[c_inner, c_outer],
        tallies=[current_tally_obj],
    )
    mcdc_struct = mcdc_container[0]
    current_tally = mcdc_struct["surface_crossing_tallies"][current_tally_obj.sub_ID]

    # Inner -> outer across the cylindrical surface: outgoing from c_inner (+)
    particle_container = crossing_particle(
        s_cyl.ID,
        x=1.0,
        ux=0.5,
        cell_ID=c_inner.ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)

    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 0), 2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 1), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 2), 2.0, rtol=1e-5, atol=1e-8
    )

    # Outer -> inner across the same curved surface: incoming to c_inner (-)
    particle_container = crossing_particle(
        s_cyl.ID,
        x=1.0,
        ux=-0.5,
        cell_ID=c_outer.ID,
    )
    surface_crossing(particle_container, mcdc_struct, data)

    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 0), 0.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 1), 2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _bin_value_score(current_tally, mcdc_struct, data, 2), 2.0, rtol=1e-5, atol=1e-8
    )


def test_surface_cell_filter_scores_only_selected_surface(
    slab_plane_x, crossing_particle, prepare_simulation
):
    tally_obj = mcdc.Tally(
        surface=slab_plane_x["s_mid"],
        cell=slab_plane_x["c_right"],
        scores=["current-net", "current-in", "current-out"],
    )

    mcdc_container, data = prepare_simulation(
        cells=[slab_plane_x["c_left"], slab_plane_x["c_right"]],
        tallies=[tally_obj],
    )
    mcdc_struct = mcdc_container[0]
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.sub_ID]

    particle_container = crossing_particle(
        slab_plane_x["s_right"].ID,
        x=1.0,
        ux=0.5,
        cell_ID=slab_plane_x["c_right"].ID,
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

    particle_container = crossing_particle(
        slab_plane_x["s_mid"].ID,
        x=0.0,
        ux=0.5,
        cell_ID=slab_plane_x["c_left"].ID,
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
