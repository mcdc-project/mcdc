import numpy as np
import pytest

import mcdc

from mcdc.transport.simulation import surface_crossing


@pytest.fixture
def box(material_mg):
    surfaces = [
        mcdc.Surface.PlaneX(x=-1.0),
        mcdc.Surface.PlaneX(x=1.0),
        mcdc.Surface.PlaneY(y=-2.0),
        mcdc.Surface.PlaneY(y=2.0),
        mcdc.Surface.PlaneZ(z=-3.0),
        mcdc.Surface.PlaneZ(z=3.0),
    ]
    x0, x1, y0, y1, z0, z1 = surfaces
    region = +x0 & -x1 & +y0 & -y1 & +z0 & -z1
    cell = mcdc.Cell(region=region, fill=material_mg)
    outside = mcdc.Cell(region=~region, fill=material_mg)
    return {"surfaces": surfaces, "cell": cell, "outside": outside}


def _cross(crossing_particle, mcdc_struct, data, surface, cell, position, direction):
    particle_container = crossing_particle(
        surface.ID, x=position[0], ux=direction[0], cell_ID=cell.ID
    )
    particle = particle_container[0]
    particle["y"], particle["z"] = position[1:]
    particle["uy"], particle["uz"] = direction[1:]
    surface_crossing(particle_container, mcdc_struct, data)


def _tally_scores(tally_obj, mcdc_struct, data):
    tally = mcdc_struct["surface_crossing_tallies"][tally_obj.sub_ID]
    tally_base = mcdc_struct["tallies"][tally["base_ID"]]
    start = tally_base["bin_offset"]
    return data[start : start + tally_base["bin_length"]].reshape(tally_obj.bin_shape)


def test_surface_mesh_bins_box_face_crossing(
    box, crossing_particle, prepare_simulation
):
    # x-min face of the box, binned in (y, z); the mesh omits x and covers z < 1
    x0 = box["surfaces"][0]
    mesh = mcdc.MeshUniform(y=(-2.0, 2.0, 2), z=(-3.0, 2.0, 2))
    tally_obj = mcdc.Tally(
        surface=x0, cell=box["cell"], mesh=mesh, scores=["current-in"]
    )
    assert tally_obj.bin_shape == [1, 1, 1, 1, 1, 2, 2, 1]

    mcdc_container, data = prepare_simulation(
        cells=[box["cell"], box["outside"]], tallies=[tally_obj]
    )
    mcdc_struct = mcdc_container[0]

    # The second crossing (z = 2) is outside the mesh and must not be scored
    for y, z in [(1.0, 0.5), (1.0, 2.0)]:
        _cross(
            crossing_particle,
            mcdc_struct,
            data,
            x0,
            box["outside"],
            position=(-1.0, y, z),
            direction=(1.0, 0.0, 0.0),
        )

    scores = _tally_scores(tally_obj, mcdc_struct, data)
    expected = np.zeros_like(scores)
    expected[0, 0, 0, 0, 0, 1, 1, 0] = 2.0
    np.testing.assert_array_equal(scores, expected)


def test_surface_mesh_cell_filter_on_shared_plane(
    material_mg, crossing_particle, prepare_simulation
):
    # A cylinder and the annulus around it share their upper end cap, z = 1
    cylinder_surface = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=1.0)
    bottom = mcdc.Surface.PlaneZ(z=0.0)
    top = mcdc.Surface.PlaneZ(z=1.0)
    cylinder = mcdc.Cell(region=-cylinder_surface & +bottom & -top, fill=material_mg)
    annulus = mcdc.Cell(region=+cylinder_surface & +bottom & -top, fill=material_mg)
    outside = mcdc.Cell(region=-bottom | +top, fill=material_mg)

    # Nonuniform x-y mesh on the shared plane, extending beyond the cylinder cap
    mesh = mcdc.MeshStructured(x=[-3.0, -1.0, 0.0, 0.5, 3.0], y=[-3.0, 0.0, 3.0])
    plane_tally_obj = mcdc.Tally(surface=top, mesh=mesh, scores=["current-out"])
    cylinder_tally_obj = mcdc.Tally(
        surface=top, cell=cylinder, mesh=mesh, scores=["current-out"]
    )
    assert plane_tally_obj.bin_shape == [1, 1, 1, 1, 4, 2, 1, 1]

    mcdc_container, data = prepare_simulation(
        cells=[cylinder, annulus, outside],
        tallies=[plane_tally_obj, cylinder_tally_obj],
    )
    mcdc_struct = mcdc_container[0]

    # One outward crossing through the cylinder cap, one through the annulus
    for cell, position in [(cylinder, (0.25, 0.5, 1.0)), (annulus, (-2.0, -1.0, 1.0))]:
        _cross(
            crossing_particle,
            mcdc_struct,
            data,
            top,
            cell,
            position=position,
            direction=(0.0, 0.0, 1.0),
        )

    # Without the cell filter, both crossings are scored
    scores = _tally_scores(plane_tally_obj, mcdc_struct, data)
    expected = np.zeros_like(scores)
    expected[0, 0, 0, 0, 2, 1, 0, 0] = 2.0
    expected[0, 0, 0, 0, 0, 0, 0, 0] = 2.0
    np.testing.assert_array_equal(scores, expected)

    # With cell=cylinder, only the crossing leaving the cylinder is scored
    scores = _tally_scores(cylinder_tally_obj, mcdc_struct, data)
    expected = np.zeros_like(scores)
    expected[0, 0, 0, 0, 2, 1, 0, 0] = 2.0
    np.testing.assert_array_equal(scores, expected)


def test_surface_mesh_must_be_2d_on_a_plane(box, capsys):
    # Mesh needs a PlaneX/Y/Z surface filter
    with pytest.raises(SystemExit):
        mcdc.Tally(
            cell=box["cell"],
            mesh=mcdc.MeshUniform(y=(-2.0, 2.0, 2)),
            scores=["current-out"],
        )
    assert "needs a PlaneX, PlaneY, or PlaneZ" in capsys.readouterr().out

    # Mesh must omit the plane's normal axis
    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=box["surfaces"][0],
            mesh=mcdc.MeshStructured(x=[-2.0, 2.0], y=[-2.0, 0.0, 2.0]),
            scores=["current-out"],
        )
    assert "must omit the x axis" in capsys.readouterr().out
