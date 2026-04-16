import os
import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================
# Based on Kobayashi dog-leg benchmark problem
# (PNE 2001, https://doi.org/10.1016/S0149-1970(01)00007-5)


x_sect = X_SECT

# Set materials
m = mcdc.material(
    capture=np.array([0.01 * x_sect]), scatter=np.array([[0.01 * x_sect]])
)
m_void = mcdc.material(
    capture=np.array([5e-5 * x_sect]), scatter=np.array([[5e-5 * x_sect]])
)

# Set surfaces
sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
sx2 = mcdc.surface("plane-x", x=10.0)
sx3 = mcdc.surface("plane-x", x=30.0)
sx4 = mcdc.surface("plane-x", x=40.0)
sx5 = mcdc.surface("plane-x", x=60.0, bc="vacuum")
sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
sy2 = mcdc.surface("plane-y", y=10.0)
sy3 = mcdc.surface("plane-y", y=50.0)
sy4 = mcdc.surface("plane-y", y=60.0)
sy5 = mcdc.surface("plane-y", y=100.0, bc="vacuum")
sz1 = mcdc.surface("plane-z", z=0.0, bc="reflective")
sz2 = mcdc.surface("plane-z", z=10.0)
sz3 = mcdc.surface("plane-z", z=30.0)
sz4 = mcdc.surface("plane-z", z=40.0)
sz5 = mcdc.surface("plane-z", z=60.0, bc="vacuum")

# Set cells
# Soruce
mcdc.cell(+sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2, m)
# Voids
mcdc.cell(+sx1 & -sx2 & +sy2 & -sy3 & +sz1 & -sz2, m_void)
mcdc.cell(+sx1 & -sx3 & +sy3 & -sy4 & +sz1 & -sz2, m_void)
mcdc.cell(+sx3 & -sx4 & +sy3 & -sy4 & +sz1 & -sz3, m_void)
mcdc.cell(+sx3 & -sx4 & +sy3 & -sy5 & +sz3 & -sz4, m_void)
# Shield
mcdc.cell(+sx1 & -sx3 & +sy1 & -sy5 & +sz2 & -sz5, m)
mcdc.cell(+sx2 & -sx5 & +sy1 & -sy3 & +sz1 & -sz2, m)
mcdc.cell(+sx3 & -sx5 & +sy1 & -sy3 & +sz2 & -sz5, m)
mcdc.cell(+sx3 & -sx5 & +sy4 & -sy5 & +sz1 & -sz3, m)
mcdc.cell(+sx4 & -sx5 & +sy4 & -sy5 & +sz3 & -sz5, m)
mcdc.cell(+sx4 & -sx5 & +sy3 & -sy4 & +sz1 & -sz5, m)
mcdc.cell(+sx3 & -sx4 & +sy3 & -sy5 & +sz4 & -sz5, m)
mcdc.cell(+sx1 & -sx3 & +sy4 & -sy5 & +sz1 & -sz2, m)

# =============================================================================
# Set source
# =============================================================================
# The source pulses in t=[0,5]

mcdc.source(
    x=[0.0, 10.0], y=[0.0, 10.0], z=[0.0, 10.0], time=[0.0, 50.0], isotropic=True
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: z-integrated flux (X-Y section view)
mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 60.0, 61),
    y=np.linspace(0.0, 100.0, 101),
    z=np.linspace(0.0, 60.0, 61),
    t=np.linspace(0.0, 200.0, TIME_SPLITS),
)

# Setting
mcdc.setting(N_particle=PARTICLE_COUNT)
# mcdc.implicit_capture()

# Run
mcdc.run()
