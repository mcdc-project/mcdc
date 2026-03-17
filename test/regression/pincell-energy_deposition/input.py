import mcdc
import numpy as np
import os

os.environ["MCDC_LIB"] = "../MCDC-regression_test_data/"

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

mcdc.Cell(-cylinder, fill=fuel)
mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +cylinder, fill=moderator)

# Source
mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True, time=0.0, energy=14.1e6)

# Settings
mcdc.settings.N_particle = 10
mcdc.settings.N_batch = 2
mcdc.settings.time_boundary = 1.0
mcdc.settings.active_bank_buffer = 1000

# Edep tally
mesh = mcdc.MeshUniform(
    x=(-pitch / 2, pitch / 8, 8),
    y=(-pitch / 2, pitch / 8, 8),
)

mcdc.Tally(name="edep_mesh", mesh=mesh, scores=["energy_deposition"])

mcdc.run()
