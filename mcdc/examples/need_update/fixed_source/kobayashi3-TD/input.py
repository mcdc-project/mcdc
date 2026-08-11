import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================
# Based on Kobayashi dog-leg benchmark problem
# (PNE 2001, https://doi.org/10.1016/S0149-1970(01)00007-5)

# Set materials
m = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.05]]))
m_void = mcdc.material(capture=np.array([5e-5]), scatter=np.array([[5e-5]]))

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
source_cell = mcdc.cell(+sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2, m)
# Voids
channel_1 = +sx1 & -sx2 & +sy2 & -sy3 & +sz1 & -sz2
channel_2 = +sx1 & -sx3 & +sy3 & -sy4 & +sz1 & -sz2
channel_3 = +sx3 & -sx4 & +sy3 & -sy4 & +sz1 & -sz3
channel_4 = +sx3 & -sx4 & +sy3 & -sy5 & +sz3 & -sz4
void_channel = channel_1 | channel_2 | channel_3 | channel_4
void_cell = mcdc.cell(void_channel, m_void)
# Shield
box = +sx1 & -sx5 & +sy1 & -sy5 & +sz1 & -sz5
shield_cell = mcdc.cell(box & ~void_channel, m)

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
    x=np.linspace(0.0, 60.0, 31),
    y=np.linspace(0.0, 100.0, 51),
    t=np.linspace(0.0, 200.0, 21),
    # g=np.array([-0.5, 3.5, 6.5]) # fast (0, 1, 2, 3) and thermal (4, 5, 6) groups
)

mcdc.tally.cell_tally(source_cell, scores=["flux"])
mcdc.tally.cell_tally(void_cell, scores=["flux"])
mcdc.tally.cell_tally(shield_cell, scores=["flux"])


mcdc.tally.cs_tally(
    N_cs_bins=[150],
    cs_bin_size=[8.0, 8.0],
    x=np.linspace(0.0, 60.0, 31),
    y=np.linspace(0.0, 100.0, 51),
    scores=["flux"],
)


# Setting
mcdc.setting(N_particle=1e5)

# Run
mcdc.run()
