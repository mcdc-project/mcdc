import mcdc
import numpy as np
import pytest
import os

from mcdc.main import preparation

def make_ww_model(lower=0.1, target=1.0, upper=1.0, mess_up_size=False):
  # Geometry
  # surfaces
  cylinder = mcdc.Surface.CylinderZ(radius=0.5)
  pitch = 2.0
  height = 10.0
  x0 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
  x1 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
  y0 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
  y1 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")
  z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="reflective")
  z1 = mcdc.Surface.PlaneZ(z=height, boundary_condition="reflective")

  # cells
  mcdc.Cell(-cylinder)
  mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1 & +cylinder)

  # Source
  mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True, time=0.0, energy=14.1e6)

  # Setting
  mcdc.settings.N_particle = 20
  mcdc.settings.N_batch = 2
  mcdc.settings.time_boundary = 1.0
  mcdc.settings.active_bank_buffer = 1000

  # weight windows
  N = 3
  mesh = mcdc.MeshUniform(
    "mesh",
    x=(-pitch/2, pitch/2, N),
    y=(-pitch/2, pitch/2, N),
    z=(0.0, height, N)
  )
  if mess_up_size:
    shape = (N, N, 4, 3)
  else:
    shape = (N, N, N, 3)
  ww_array = np.ones(shape)
  ww_array[:, :, :, 0] = lower
  ww_array[:, :, :, 1] = target
  ww_array[:, :, :, 2] = upper
  mcdc.simulation.weight_windows(mesh, ww_array)

  return preparation()

@pytest.mark.parametrize(
    "kwargs, expected_msg",
    [
        # incorrect size
        (
            {"mess_up_size": True},
            "Weight window array has shape (3, 3, 4, 3), but expected (3, 3, 3, 3)",
        ),
        # negative lower
        (
            {"lower": -1.0},
            "Lower bound weights must be strictly positive",
        ),
        # lower > target
        (
            {"lower": 1.0, "target": 0.5},
            "Lower bound weight can not be greater than the target weight",
        ),
        # target > upper
        (
            {"target": 1.5},
            "Target weight can not be greater than the upper bound weight",
        ),
    ],
)
def test_error_throw(capsys, kwargs, expected_msg):
  with pytest.raises(SystemExit):
    make_ww_model(**kwargs)

  out = capsys.readouterr().out
  assert expected_msg in out