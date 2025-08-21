import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different materials
# Based on William H. Reed, NSE (1971), 46:2, 309-314, DOI: 10.13182/NSE46-309

# Set materials
m1 = mcdc.MaterialMG(capture=np.array([50.0]))
m2 = mcdc.MaterialMG(capture=np.array([5.0]))
m3 = mcdc.MaterialMG(capture=np.array([0.0]))  # Vacuum
m4 = mcdc.MaterialMG(capture=np.array([0.1]), scatter=np.array([[0.9]]))

# Set surfaces
s1 = mcdc.surface("plane-z", z=0.0, bc="reflective")
s2 = mcdc.surface("plane-z", z=2.0)
s3 = mcdc.surface("plane-z", z=3.0)
s4 = mcdc.surface("plane-z", z=5.0)
s5 = mcdc.surface("plane-z", z=8.0, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, m1)
mcdc.cell(+s2 & -s3, m2)
mcdc.cell(+s3 & -s4, m3)
mcdc.cell(+s4 & -s5, m4)

# =============================================================================
# Set source
# =============================================================================

# Isotropic source in the absorbing medium
mcdc.source(z=[0.0, 2.0], isotropic=True, prob=50.0)

# Isotropic source in the first half of the outermost medium,
# with 1/100 strength
mcdc.source(z=[5.0, 6.0], isotropic=True, prob=0.5)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=5000, N_batch=2)

mcdc.tally.mesh_tally(scores=["flux"], z=np.linspace(0.0, 8.0, 81))

mcdc.run()
