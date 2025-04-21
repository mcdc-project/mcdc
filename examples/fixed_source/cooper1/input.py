import numpy as np
import mcdc


# =============================================================================
# Set model
# =============================================================================
# A problem based on Problem 1 of [Coper NSE 2001]
# https://ans.tandfonline.com/action/showCitFormats?doi=10.13182/NSE00-34

# Set materials
SigmaT = 0.01
c = 0.8
m_duct = mcdc.material(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))
SigmaT = 1.0
m_room = mcdc.material(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))

# Set surfaces
sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
sx2 = mcdc.surface("plane-x", x=2.0)
sx3 = mcdc.surface("plane-x", x=2.4)
sx4 = mcdc.surface("plane-x", x=4.0, bc="vacuum")
sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
sy2 = mcdc.surface("plane-y", y=1.0)
sy3 = mcdc.surface("plane-y", y=2.6)
sy4 = mcdc.surface("plane-y", y=3)
sy5 = mcdc.surface("plane-y", y=4.0, bc="vacuum")

# Set cells
# Room
mcdc.cell(+sx1 & -sx2 & +sy1 & -sy5, m_room)
mcdc.cell(+sx1 & -sx4 & +sy4 & -sy5, m_room)
mcdc.cell(+sx2 & -sx3 & +sy1 & -sy2, m_room)
mcdc.cell(+sx3 & -sx4 & +sy1 & -sy3, m_room)

# Duct Channel
channel_1 = +sx2 & -sx3 & +sy2 & -sy3
channel_2 = +sx2 & -sx4 & +sy3 & -sy4
mcdc.cell(channel_1 | channel_2, m_duct)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source(x=[0.0, 1.0], y=[0.0, 1.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 4.0, 41),
    y=np.linspace(0.0, 4.0, 41),
)

mcdc.tally.cs_tally(
    N_cs_bins=[150],
    cs_bin_size=[5.0, 5.0],
    x=np.linspace(0.0, 4.0, 41),
    y=np.linspace(0.0, 4.0, 41),
    scores=["flux"],
)

# Setting
mcdc.setting(N_particle=1e3)
mcdc.implicit_capture()

# Run
mcdc.run()
