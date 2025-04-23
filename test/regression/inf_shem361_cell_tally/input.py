import numpy as np
import sys

import mcdc

# This regression test adds time census and time/energy binned cell tallies to the inf_shem361 test

# =============================================================================
# Set model
# =============================================================================
# The infinite homogenous medium is modeled with reflecting slab

# Load material data
with np.load("SHEM-361.npz") as data:
    SigmaC = data["SigmaC"] * 5  # /cm
    SigmaS = data["SigmaS"]
    SigmaF = data["SigmaF"]
    nu_p = data["nu_p"]
    nu_d = data["nu_d"]
    chi_p = data["chi_p"]
    chi_d = data["chi_d"]
    G = data["G"]
    speed = data["v"]
    lamd = data["lamd"]

# Set material
m = mcdc.material(
    capture=SigmaC,
    scatter=SigmaS,
    fission=SigmaF,
    nu_p=nu_p,
    chi_p=chi_p,
    nu_d=nu_d,
    chi_d=chi_d,
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Set cells
c = mcdc.cell(+s1 & -s2, m)

# =============================================================================
# Set initial source
# =============================================================================

energy = np.zeros(G)
energy[-1] = 1.0
source = mcdc.source(energy=energy)

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

# Tally
# mcdc.tally.mesh_tally(scores=["flux"], g="all")
mcdc.tally.cell_tally(c, scores=["flux"], g="all", t=np.linspace(0.0, 20.0, 21)[1:-1])

# Setting
mcdc.setting(N_particle=1e2, active_bank_buff=1000, N_batch=2)
mcdc.time_census(np.linspace(0.0, 20.0, 21)[1:-1])

# Run
mcdc.run()
