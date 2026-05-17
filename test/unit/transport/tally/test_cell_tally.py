import numpy as np

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.main import preparation
from mcdc.transport.geometry import interface as geometry_interface


def _new_particle(surface_ID, x, ux, w=2.0):
    particle_container = np.zeros(1, type_.particle)
    particle = particle_container[0]
    particle["alive"] = True
    particle["particle_type"] = PARTICLE_NEUTRON
    particle["surface_ID"] = surface_ID
    particle["material_ID"] = 0
    particle["g"] = 0
    particle["x"] = x
    particle["y"] = 0.0
    particle["z"] = 0.0
    particle["t"] = 0.0
    particle["ux"] = ux
    particle["uy"] = 0.0
    particle["uz"] = 0.0
    particle["w"] = w
    return particle_container


def _bin_value(cell_tally, mcdc_struct, data):
    tally_base = mcdc_struct["tallies"][cell_tally["parent_ID"]]
    return data[tally_base["bin_offset"]]


def _bin_value_score(cell_tally, mcdc_struct, data, score_idx):
    tally_base = mcdc_struct["tallies"][cell_tally["parent_ID"]]
    return data[tally_base["bin_offset"] + score_idx]


def test_cell_tally_sign_for_incoming_and_outgoing(slab_plane_x):
    cell_tally_obj = mcdc.Tally(cell=slab_plane_x["c_right"], scores=["net-current"])
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    cell_tally = mcdc_struct["cell_tallies"][cell_tally_obj.child_ID]

    # Left -> right across the shared interior surface: incoming to c_right (+)
    particle_container = _new_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert np.isclose(_bin_value(cell_tally, mcdc_struct, data), 2.0)

    # Right -> left across the shared interior surface: outgoing from c_right (-)
    particle_container = _new_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=-0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert np.isclose(_bin_value(cell_tally, mcdc_struct, data), 0.0)


def test_cell_tally_records_in_and_out_separately(slab_plane_x):
    cell_tally_obj = mcdc.Tally(
        cell=slab_plane_x["c_right"],
        scores=["net-current", "current-in", "current-out"],
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    cell_tally = mcdc_struct["cell_tallies"][cell_tally_obj.child_ID]

    # One incoming and one outgoing crossing.
    particle_container = _new_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    particle_container = _new_particle(slab_plane_x["s_mid"].ID, x=0.0, ux=-0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)

    # Scores are in requested order: [net-current, current-in, current-out].
    # current-out uses negative sign for exiting particles.
    assert np.isclose(_bin_value_score(cell_tally, mcdc_struct, data, 0), 0.0)
    assert np.isclose(_bin_value_score(cell_tally, mcdc_struct, data, 1), 2.0)
    assert np.isclose(_bin_value_score(cell_tally, mcdc_struct, data, 2), -2.0)


def test_cell_tally_counts_outgoing_to_vacuum(slab_plane_x):
    cell_tally_obj = mcdc.Tally(cell=slab_plane_x["c_right"], scores=["net-current"])
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    cell_tally = mcdc_struct["cell_tallies"][cell_tally_obj.child_ID]

    # c_right -> vacuum across the right boundary: outgoing (-)
    particle_container = _new_particle(slab_plane_x["s_right"].ID, x=1.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert np.isclose(_bin_value(cell_tally, mcdc_struct, data), -2.0)


def test_cell_tally_ignores_reflective_crossing(material_mg):
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="reflective")
    c_mid = mcdc.Cell(region=+s_left & -s_right, fill=material_mg)
    cell_tally_obj = mcdc.Tally(cell=c_mid, scores=["net-current"])

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    cell_tally = mcdc_struct["cell_tallies"][cell_tally_obj.child_ID]

    particle_container = _new_particle(s_right.ID, x=1.0, ux=0.5)
    geometry_interface.surface_crossing(particle_container, mcdc_struct, data)
    assert np.isclose(_bin_value(cell_tally, mcdc_struct, data), 0.0)
