import numpy as np
import os
import sys
import pytest

os.environ["MCDC_LIB"] = "../../../regression/mcdc-regression_test_data/"

# =============================================================================
# State helpers
# =============================================================================


def _clear_mcdc_modules():
    for name in list(sys.modules):
        if name == "mcdc" or name.startswith("mcdc."):
            del sys.modules[name]

@pytest.fixture
def fresh_mcdc():
    _clear_mcdc_modules()
    import mcdc
    yield mcdc 
    _clear_mcdc_modules()


# =============================================================================
# Model base fixture
# =============================================================================


@pytest.fixture
def pin_cell_model(fresh_mcdc):
  mcdc = fresh_mcdc
  # Material
  fuel = mcdc.Material(
      nuclide_composition={
          "U235": 1.0 
      }
  )
  moderator = mcdc.Material(
      nuclide_composition={
          "H1": 1.0
      }
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

  yield mcdc, fuel_cell, mod_cell


# =============================================================================
# Error throwing in object creation
# =============================================================================


@pytest.mark.parametrize(
    "cells_builder, thresholds, targets, expected_msg",
    [
        (
            lambda fuel_cell, mcdc: [fuel_cell],
            [0.5, 0.5],
            [1.0],
            "Expected cells, threshold_weights, and target_weights to be the same size",
        ),
        (
            lambda fuel_cell, mcdc: [fuel_cell],
            [0.5],
            [1.0, 1.0],
            "Expected cells, threshold_weights, and target_weights to be the same size",
        ),
        (
            lambda fuel_cell, mcdc: [mcdc.Cell(fill=mcdc.Universe(cells=[fuel_cell]))],
            None,
            None,
            "Invalid cell fill on cell",
        ),
    ],
)
def test_forced_collisions_error_throw(
    pin_cell_model, capsys, cells_builder, thresholds, targets, expected_msg
):
    mcdc, fuel_cell, mod_cell = pin_cell_model

    cells = cells_builder(fuel_cell, mcdc)

    with pytest.raises(SystemExit):
        mcdc.simulation.forced_collisions(
            cells,
            threshold_weights=thresholds,
            target_weights=targets,
        )

    captured = capsys.readouterr()
    assert expected_msg in captured.out

