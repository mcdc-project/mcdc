import numpy as np
import mcdc

# ======================================================================================
# Set model
# ======================================================================================
# The infinite homogenous medium is modeled with reflecting slab

# Load material data
with np.load("SHEM-361.npz") as data:
    SigmaC = data["SigmaC"] * 3.0  # /cm
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
# Set tallies, settings, techniques, and run MC/DC
# ======================================================================================

# Tallies
mcdc.TallyGlobal(
    scores=["flux"],
    time=np.insert(np.logspace(-8, 1, 100), 0, 0.0),
    energy="all_groups",
)

# Settings
mcdc.settings.N_particle = 40
mcdc.settings.N_batch = 2
mcdc.settings.set_time_census(np.logspace(-5, 1, 6))
mcdc.settings.active_bank_buffer = 1000
mcdc.settings.census_bank_buffer_ratio = 5.0
mcdc.settings.source_bank_buffer_ratio = 5.0

# Techniques
mcdc.simulation.population_control()

# Run
mcdc.run()
