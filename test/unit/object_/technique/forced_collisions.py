import numpy as np
import pytest
import mcdc

# =============================================================================
# Model base fixture
# =============================================================================


@pytest.fixture
def pin_cell_model():
    # Material
    fuel = mcdc.MaterialMG(
        capture=np.array([1.0 / 3.0]),
        scatter=np.array([[1.0 / 3.0]]),
    )
    moderator = mcdc.MaterialMG(
        capture=np.array([1.0 / 3.0]),
        scatter=np.array([[1.0 / 3.0]]),
    )

    # Geometry
    cylinder = mcdc.Surface.CylinderZ(radius=0.5)
    pitch = 1.5
    x0 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
    x1 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
    y0 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
    y1 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")
    #
    fuel_cell = mcdc.Cell(-cylinder, fill=fuel)
    mod_cell = mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +cylinder, fill=moderator)

    yield fuel_cell, mod_cell


# =============================================================================
# Error throwing in object creation
# =============================================================================


@pytest.mark.parametrize(
    "cells_builder, thresholds, targets, expected_msg",
    [
        (
            lambda fuel_cell: [fuel_cell],
            [0.5, 0.5],
            [1.0],
            "Expected cells, threshold_weights, and target_weights to be the same size",
        ),
        (
            lambda fuel_cell: [fuel_cell],
            [0.5],
            [1.0, 1.0],
            "Expected cells, threshold_weights, and target_weights to be the same size",
        ),
        (
            lambda fuel_cell: [mcdc.Cell(fill=mcdc.Universe(cells=[fuel_cell]))],
            None,
            None,
            "Invalid cell fill on cell",
        ),
    ],
)
def test_forced_collisions_error_throw(
    pin_cell_model, capsys, cells_builder, thresholds, targets, expected_msg
):
    fuel_cell, mod_cell = pin_cell_model

    cells = cells_builder(fuel_cell)

    with pytest.raises(SystemExit):
        mcdc.simulation.forced_collisions(
            cells,
            threshold_weights=thresholds,
            target_weights=targets,
        )

    captured = capsys.readouterr()
    assert expected_msg in captured.out


# =============================================================================
# Method tests
# =============================================================================
