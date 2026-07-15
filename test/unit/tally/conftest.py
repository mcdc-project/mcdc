import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.main import preparation
from mcdc.object_.simulation import simulation


@pytest.fixture(autouse=True)
def reset_simulation():
    # Keep simulation state isolated per test.
    simulation.__init__()
    yield
    simulation.__init__()


@pytest.fixture
def material_mg():
    # Minimal multigroup material so particle speed is defined.
    return mcdc.MaterialMG(capture=np.array([1.0]))


@pytest.fixture
def crossing_particle():
    def _particle(surface_ID, x, ux, cell_ID=-1, w=2.0):
        particle_container = np.zeros(1, type_.particle)
        particle = particle_container[0]
        particle["alive"] = True
        particle["particle_type"] = PARTICLE_NEUTRON
        particle["surface_ID"] = surface_ID
        particle["cell_ID"] = cell_ID
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

    return _particle


@pytest.fixture
def slab_plane_x(material_mg):
    # Two-cell 1D slab with shared interior PlaneX.
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_mid = mcdc.Surface.PlaneX(x=0.0)
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")
    c_left = mcdc.Cell(region=+s_left & -s_mid, fill=material_mg)
    c_right = mcdc.Cell(region=+s_mid & -s_right, fill=material_mg)
    return {
        "s_left": s_left,
        "s_mid": s_mid,
        "s_right": s_right,
        "c_left": c_left,
        "c_right": c_right,
    }


@pytest.fixture
def surface_crossing_tally_context(slab_plane_x):
    s_mid = slab_plane_x["s_mid"]

    unbounded_tally_obj = mcdc.Tally(surface=s_mid, scores=["current-net"])

    # Build compiled structures.
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]

    # Compiled tally handles.
    unbounded_tally = mcdc_struct["surface_crossing_tallies"][
        unbounded_tally_obj.child_ID
    ]

    # Particle for direct crossing-based tally-kernel testing.
    particle_container = np.zeros(1, type_.particle)
    particle = particle_container[0]
    particle["particle_type"] = PARTICLE_NEUTRON
    particle["material_ID"] = 0
    particle["g"] = 0
    particle["x"] = 0.0
    particle["y"] = 0.0
    particle["z"] = 0.0
    particle["t"] = 0.0
    particle["ux"] = 0.5
    particle["uy"] = 0.0
    particle["uz"] = 0.0
    particle["w"] = 2.0

    return {
        "data": data,
        "mcdc_struct": mcdc_struct,
        "particle": particle,
        "particle_container": particle_container,
        "s_mid": s_mid,
        "unbounded_tally": unbounded_tally,
    }


@pytest.fixture
def bin_value():
    def _value(surface_crossing_tally, mcdc_struct, data):
        tally_base = mcdc_struct["tallies"][surface_crossing_tally["parent_ID"]]
        return data[tally_base["bin_offset"]]

    return _value
