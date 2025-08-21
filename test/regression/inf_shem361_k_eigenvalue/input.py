import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# The infinite homogenous medium is modeled with reflecting slab

# Load material data
with np.load("SHEM-361.npz") as data:
    SigmaC = data["SigmaC"]  # /cm
    SigmaS = data["SigmaS"]
    SigmaF = data["SigmaF"]
    nu_p = data["nu_p"]
    nu_d = data["nu_d"]
    chi_p = data["chi_p"]
    chi_d = data["chi_d"]
    G = data["G"]

# Set material
m = mcdc.MaterialMG(
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

source = mcdc.source(energy=np.ones(G))  # Arbitrary

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=1e2, source_bank_buffer_ratio=2.0)
settings.set_eigenmode(N_inactive=1, N_active=2)
mcdc.population_control()

mcdc.tally.mesh_tally(
    scores=["flux"],
    g="all",
)

mcdc.run()
