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
s1 = mcdc.Surface.PlaneX(x=-1e10, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneX(x=1e10, boundary_condition="reflective")

# Set cells
c = mcdc.Cell(region=+s1 & -s2, fill=m)

# =============================================================================
# Set initial source
# =============================================================================

mcdc.Source(
    position=(0.0, 0.0, 0.0), isotropic=True, energy_group=np.array([[360], [1.0]])
)

# =============================================================================
# Set tallies, settings, techniques, and run MC/DC
# =============================================================================

# Tallies
mcdc.TallyGlobal(scores=["flux"], energy="all_groups")

# Settings
mcdc.settings.N_particle = 70
mcdc.settings.source_bank_buffer_ratio = 2.0
mcdc.settings.census_bank_buffer_ratio = 3.0
mcdc.settings.set_eigenmode(N_inactive=1, N_active=2)

# Techniques
mcdc.simulation.population_control()

# Run
mcdc.run()
