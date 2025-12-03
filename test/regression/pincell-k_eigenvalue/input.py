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
#
mcdc.Cell(-cylinder, fill=fuel)
mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +cylinder, fill=moderator)

# Source
mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True, time=0.0, energy=14.1e6)

# Setting
mcdc.settings.N_particle = 30
mcdc.settings.time_boundary = 1.0
mcdc.settings.active_bank_buffer = 1000
mcdc.settings.census_bank_buffer_ratio = 3.0
mcdc.settings.source_bank_buffer_ratio = 3.0
mcdc.settings.set_eigenmode(N_inactive=1, N_active=2)

# Tally
t_grid = np.insert(np.logspace(-9, -4, 200), 0, 0.0)
e_min, e_max = 1e-5, 20.0e6
groups = 500
energies = np.logspace(np.log10(e_min), np.log10(e_max), groups + 1)

mcdc.TallyGlobal(scores=["flux"], time=t_grid, energy=energies)

mcdc.run()
