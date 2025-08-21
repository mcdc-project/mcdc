import numpy as np
import sys

import mcdc


# =============================================================================
# Set model
# =============================================================================
# The infinite homogenous medium is modeled with reflecting slab

# Load material data
with np.load("SHEM-361.npz") as data:
    SigmaC = data["SigmaC"] * 1.5  # /cm
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
m = mcdc.MaterialMG(
    capture=SigmaC,
    scatter=SigmaS,
    fission=SigmaF,
    nu_p=nu_p,
    chi_p=chi_p,
    nu_d=nu_d,
    chi_d=chi_d,
    decay_rate=lamd,
    speed=speed,
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

settings = mcdc.Settings(
    N_particle=30,
    active_bank_buffer=1000,
    census_bank_buffer_ratio=5,
    source_bank_buffer_ratio=5,
    rng_seed=7,
    N_batch=2,
)
settings.set_time_census(np.logspace(-5, 1, 6))
mcdc.population_control()

mcdc.tally.mesh_tally(
    scores=["flux"], t=np.insert(np.logspace(-8, 1, 100), 0, 0.0), g="all"
)

mcdc.run()
