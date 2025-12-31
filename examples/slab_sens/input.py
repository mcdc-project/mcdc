import os
import numpy as np
import mcdc

# -----------------------------------------------------------------------------
# Continuous-energy data
# -----------------------------------------------------------------------------
# You must point to a directory of nuclide HDF5s, e.g.:
#   export MCDC_LIB=/path/to/mcdc_ce_library
#
# Nuclide names must match the file prefix in that directory.
# Example filename: H1-293.6K.h5  -> nuclide name "H1"
#
if "MCDC_LIB" not in os.environ:
    raise RuntimeError("Set MCDC_LIB to your continuous-energy nuclear data directory")

# Material: simple scatterer (use a nuclide you actually have in MCDC_LIB)
mat = mcdc.Material(nuclide_composition={"H1": 0.066})  # density units per your library

# -----------------------------------------------------------------------------
# Geometry: slab in x, reflective in y/z
# -----------------------------------------------------------------------------
L = 10.0
xm = 5.0

sx0 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
sxm = mcdc.Surface.PlaneX(x=xm)
sxL = mcdc.Surface.PlaneX(x=L, boundary_condition="vacuum")

sy0 = mcdc.Surface.PlaneY(y=-1.0, boundary_condition="reflective")
sy1 = mcdc.Surface.PlaneY(y=+1.0, boundary_condition="reflective")
sz0 = mcdc.Surface.PlaneZ(z=-1.0, boundary_condition="reflective")
sz1 = mcdc.Surface.PlaneZ(z=+1.0, boundary_condition="reflective")

left = mcdc.Cell(
    name="left", region=(+sx0 & -sxm & +sy0 & -sy1 & +sz0 & -sz1), fill=mat
)
right = mcdc.Cell(
    name="right", region=(+sxm & -sxL & +sy0 & -sy1 & +sz0 & -sz1), fill=mat
)

# -----------------------------------------------------------------------------
# Source: isotropic point near left boundary
# -----------------------------------------------------------------------------
mcdc.Source(
    position=[0.1, 0.0, 0.0],
    isotropic=True,
    energy=2.0e6,  # eV
)

# -----------------------------------------------------------------------------
# Tallies: flux in each cell (baseline sanity check)
# -----------------------------------------------------------------------------
mcdc.TallyCell(name="flux_left", cell=left, scores=["flux"])
mcdc.TallyCell(name="flux_right", cell=right, scores=["flux"])

# -----------------------------------------------------------------------------
# Sensitivity response regions: per-cell resp_cum
# -----------------------------------------------------------------------------
mcdc.settings.sensitivity_mode = True
mcdc.settings.sensitivity_n_resp = 2
mcdc.settings.sensitivity_resp_cell_IDs = np.array([left.ID, right.ID], dtype=np.int64)

# -----------------------------------------------------------------------------
# Run settings
# -----------------------------------------------------------------------------
mcdc.settings.N_particle = 20000
mcdc.settings.N_batch = 10
mcdc.simulation.implicit_capture()

mcdc.run()
