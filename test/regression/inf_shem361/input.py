import numpy as np
import mcdc

# ======================================================================================
# Set model
# ======================================================================================
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
)

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=-1e10, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneX(x=1e10, boundary_condition="reflective")

# Set cells
c = mcdc.Cell(region=+s1 & -s2, fill=m)

# ======================================================================================
# Set source
# ======================================================================================

mcdc.Source(
    position=(0.0, 0.0, 0.0), isotropic=True, energy_group=np.array([[360], [1.0]])
)

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mcdc.Tally(scores=["flux"], energy="all_groups")

# Swttings
mcdc.settings.N_particle = 25
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000

# Run
mcdc.run()
