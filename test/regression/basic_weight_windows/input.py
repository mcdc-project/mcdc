import mcdc
from mcdc.constant import INF
import numpy as np
import os

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Uses Pincell model as the basis

# Material
fuel = mcdc.Material(
    nuclide_composition={
        "U235": 0.0001654509603995036,
        "U238": 0.022801089905717036,
        "O16": 0.04593308173223308,
    }
)
moderator = mcdc.Material(
    nuclide_composition={
        "H1": 0.05129627050184732,
        "O16": 0.024622209840886707,
        "B10": 4.103701640147785e-05,
    }
)

# Geometry
cylinder = mcdc.Surface.CylinderZ(radius=0.45720)
pitch = 1.25984
x0 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")
#
mcdc.Cell(-cylinder, fill=fuel)
mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +cylinder, fill=moderator)

# Source
mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True, time=0.0, energy=14.1e6)

# Setting
mcdc.settings.N_particle = 20
mcdc.settings.N_batch = 2
mcdc.settings.time_boundary = 1.0
mcdc.settings.active_bank_buffer = 1000

# Mesh
Nx, Ny = 20, 20
x0, y0 = -pitch / 2, -pitch / 2
dx, dy = pitch / Nx, pitch / Ny
mesh = mcdc.MeshUniform(x=(x0, dx, Nx), y=(y0, dy, Ny))

# Tally
mcdc.Tally(mesh=mesh, scores=["flux"])

# Weight windows
ww_array = np.ones((1, 20, 20, 1, 3))
# Actual bounds are set to arbitrary numbers
ww_array[..., 0] = 0.55  # Forces roulette on split particles from 1.0
ww_array[..., 1] = 0.7  # arbitrary in the middle
ww_array[..., 2] = 0.9  # forces splitting on all particles born with w=1.0
mcdc.simulation.weight_windows(ww_array, mesh=mesh)

mcdc.run()
