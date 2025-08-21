from mcdc.input_ import nuclide_registered
import numpy as np
import os, h5py
from mpi4py import MPI

import mcdc


# Set the XS library directory
os.environ["MCDC_XSLIB"] = os.getcwd()

# Create the material
dummy_material = mcdc.Material(
    nuclide_composition={
        "dummy_nuclide": 1.0,
    }
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
s2 = mcdc.surface("plane-x", x=2.0, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, dummy_material)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(
    x=[0.95, 1.05],
    energy=np.array([[0.9, 1.1], [1.0, 1.0]]),
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=20, N_batch=2)

mcdc.tally.mesh_tally(
    scores=["flux"], x=np.linspace(0.0, 2.0, 21), E=np.array([0.0, 1.0, 20e6])
)

mcdc.run()
