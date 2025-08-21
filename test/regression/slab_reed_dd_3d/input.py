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
sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
sx2 = mcdc.surface("plane-x", x=8.0, bc="vacuum")
sx3 = mcdc.surface("plane-x", x=4.0)
sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
sy2 = mcdc.surface("plane-y", y=8.0, bc="vacuum")
sy3 = mcdc.surface("plane-y", y=4.0)

# Set cells
mcdc.cell(+s1 & -s2 & +sx1 & -sx3 & +sy1 & -sy3, m1)
mcdc.cell(+s2 & -s3 & +sx1 & -sx3 & +sy1 & -sy3, m2)
mcdc.cell(+s3 & -s4 & +sx1 & -sx3 & +sy1 & -sy3, m3)
mcdc.cell(+s4 & -s5 & +sx1 & -sx3 & +sy1 & -sy3, m4)

mcdc.cell(+s1 & -s2 & +sx3 & -sx2 & +sy1 & -sy3, m1)
mcdc.cell(+s2 & -s3 & +sx3 & -sx2 & +sy1 & -sy3, m2)
mcdc.cell(+s3 & -s4 & +sx3 & -sx2 & +sy1 & -sy3, m3)
mcdc.cell(+s4 & -s5 & +sx3 & -sx2 & +sy1 & -sy3, m4)

mcdc.cell(+s1 & -s2 & +sx1 & -sx3 & +sy3 & -sy2, m1)
mcdc.cell(+s2 & -s3 & +sx1 & -sx3 & +sy3 & -sy2, m2)
mcdc.cell(+s3 & -s4 & +sx1 & -sx3 & +sy3 & -sy2, m3)
mcdc.cell(+s4 & -s5 & +sx1 & -sx3 & +sy3 & -sy2, m4)

mcdc.cell(+s1 & -s2 & +sx3 & -sx2 & +sy3 & -sy2, m1)
mcdc.cell(+s2 & -s3 & +sx3 & -sx2 & +sy3 & -sy2, m2)
mcdc.cell(+s3 & -s4 & +sx3 & -sx2 & +sy3 & -sy2, m3)
mcdc.cell(+s4 & -s5 & +sx3 & -sx2 & +sy3 & -sy2, m4)

# =============================================================================
# Set source
# =============================================================================

# Isotropic source in the absorbing medium
mcdc.source(x=[0.0, 4.0], y=[0.0, 4.0], z=[0.0, 2.0], isotropic=True, prob=50.0)
mcdc.source(x=[4.0, 8.0], y=[0.0, 4.0], z=[0.0, 2.0], isotropic=True, prob=50.0)
mcdc.source(x=[0.0, 4.0], y=[4.0, 8.0], z=[0.0, 2.0], isotropic=True, prob=50.0)
mcdc.source(x=[4.0, 8.0], y=[4.0, 8.0], z=[0.0, 2.0], isotropic=True, prob=50.0)

# Isotropic source in the first half of the outermost medium,
# with 1/100 strength
mcdc.source(x=[0.0, 4.0], y=[0.0, 4.0], z=[5.0, 6.0], isotropic=True, prob=0.5)
mcdc.source(x=[4.0, 8.0], y=[0.0, 4.0], z=[5.0, 6.0], isotropic=True, prob=0.5)
mcdc.source(x=[0.0, 4.0], y=[4.0, 8.0], z=[5.0, 6.0], isotropic=True, prob=0.5)
mcdc.source(x=[4.0, 8.0], y=[4.0, 8.0], z=[5.0, 6.0], isotropic=True, prob=0.5)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 8.0, 9),
    y=np.linspace(0.0, 8.0, 9),
    z=np.linspace(0.0, 8.0, 9),
)

# Setting
mcdc.setting(N_particle=5000, N_batch=2)
dd_x = np.array([0.0, 4.0, 8.0])
dd_y = np.array([0.0, 4.0, 8.0])
dd_z = np.array([0.0, 2.0, 3.0, 5.0, 8.0])
mcdc.domain_decomposition(x=dd_x, y=dd_y, z=dd_z)
# Run
mcdc.run()
