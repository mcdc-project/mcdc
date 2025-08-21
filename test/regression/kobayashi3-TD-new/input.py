import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================
# Based on Kobayashi dog-leg benchmark problem
# (PNE 2001, https://doi.org/10.1016/S0149-1970(01)00007-5)

# Set materials
m = mcdc.MaterialMG(capture=np.array([0.05]), scatter=np.array([[0.05]]))
m_void = mcdc.MaterialMG(capture=np.array([5e-5]), scatter=np.array([[5e-5]]))

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
# Source
mcdc.cell(+sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2, m)
# Voids
channel_1 = +sx1 & -sx2 & +sy2 & -sy3 & +sz1 & -sz2
channel_2 = +sx1 & -sx3 & +sy3 & -sy4 & +sz1 & -sz2
channel_3 = +sx3 & -sx4 & +sy3 & -sy4 & +sz1 & -sz3
channel_4 = +sx3 & -sx4 & +sy3 & -sy5 & +sz3 & -sz4
void_channel = channel_1 | channel_2 | channel_3 | channel_4
mcdc.cell(void_channel, m_void)
# Shield
box = +sx1 & -sx5 & +sy1 & -sy5 & +sz1 & -sz5
mcdc.cell(box & ~void_channel, m)

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

mcdc.Settings(N_particle=80, N_batch=2)
mcdc.implicit_capture()

time_grid = np.linspace(0.0, 200.0, 21)
mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 60.0, 61),
    y=np.linspace(0.0, 100.0, 101),
    t=time_grid,
)
mcdc.tally.mesh_tally(
    scores=["density"],
    t=time_grid,
)

mcdc.run()
