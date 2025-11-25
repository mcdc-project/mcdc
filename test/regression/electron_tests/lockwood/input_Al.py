import numpy as np
import os
import math
import mcdc
from datetime import datetime

# Set the XS library directory
os.environ["MCDC_XSLIB"] = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "dummy_data")

# =============================================================================
# Set problem parameters
# =============================================================================
MATERIAL_SYMBOL = "Al"
CSDA_RANGE = 0.569 # g/cm2
RHO_G_CM3 = 2.70   # g/cm3
ATOMIC_WEIGHT_G_MOL = 26.7497084 # g/mol
AREAL_DENSITY_G_CM2 = 5.05e-3 #g/cm2
dz = AREAL_DENSITY_G_CM2 / RHO_G_CM3
AVAGADRO_NUMBER = 6.02214076e23  # atoms/mol
MAT_DENSITY_ATOMS_PER_BARN_CM = AVAGADRO_NUMBER / ATOMIC_WEIGHT_G_MOL * RHO_G_CM3 / 1e24  # atoms/barn-cm
TINY = 1e-8
M = 5

# parameters
L = CSDA_RANGE / RHO_G_CM3 * M # cm
N_LAYERS = int(L / dz)

print(f"[DEBUG] Material Symbol = {MATERIAL_SYMBOL}")
print(f"[DEBUG] CSDA Range = {CSDA_RANGE:.6e} g/cm2")
print(f"[DEBUG] Density = {RHO_G_CM3:.6e} g/cm3")
print(f"[DEBUG] Atomic Weight = {ATOMIC_WEIGHT_G_MOL:.6e} g/mol")
print(f"[DEBUG] Areal density per layer = {AREAL_DENSITY_G_CM2:.6e} g/cm2")
print(f"[DEBUG] Material density = {MAT_DENSITY_ATOMS_PER_BARN_CM:.6e} atoms/barn-cm")
print(f"[DEBUG] Layer thickness = {dz:.6e} cm")
print(f"[DEBUG] Total thickness = {L:.6e} cm")
print(f"[DEBUG] Number of layers = {N_LAYERS}")

# =============================================================================
# Set materials
# =============================================================================
mat = mcdc.Material(
    element_composition={MATERIAL_SYMBOL: MAT_DENSITY_ATOMS_PER_BARN_CM}
)

# =============================================================================
# Set geometry (surfaces and cells)
# =============================================================================
# Z-direction surfaces for layers

s1 = mcdc.surface("plane-z", z=0, bc="vacuum")
s2 = mcdc.surface("plane-z", z=L, bc="vacuum")

surfL = mcdc.surface("plane-z", z=0+1e-6)
surfR = mcdc.surface("plane-z", z=L-1e-6)

mcdc.cell(+s1 & -s2, mat)

# =============================================================================
# Set source
# =============================================================================
# Parallel beam of 1 MeV electrons entering at z=0
theta = math.radians(0)

mcdc.source(
    z = [0.0, 0.0],
    particle_type='electron',
    energy=np.array([[1e6 - 1, 1e6 + 1], [0.5, 0.5]]),
    direction=[math.sin(theta), 0.0+TINY, math.cos(theta)]
)

# =============================================================================
# Set tally
# =============================================================================
# Energy deposition tally along z-axis
z_bins = np.linspace(0.0, L, N_LAYERS + 1)

mcdc.tally.mesh_tally(
    scores=["edep", "flux"],
    z=z_bins
)

mcdc.tally.surface_tally(surfL, scores=["net-current"])
mcdc.tally.surface_tally(surfR, scores=["net-current"])


# =============================================================================
# Settings and run
# =============================================================================
settings = mcdc.Settings(
    N_particle=1,
    active_bank_buffer=1000000
)

settings.save_input_deck = True
settings.output_name = f"lockwood_output_{datetime.now():%Y%m%d_%H%M%S}"
#settings.debug_energy = True


mcdc.run()
