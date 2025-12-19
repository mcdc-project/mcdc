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

# Other parameters (don't change)
AVAGADRO_NUMBER = 6.02214076e23  # atoms/mol
MAT_DENSITY_ATOMS_PER_BARN_CM = AVAGADRO_NUMBER / ATOMIC_WEIGHT_G_MOL * RHO_G_CM3 / 1e24  # atoms/barn-cm
dz = AREAL_DENSITY_G_CM2 / RHO_G_CM3
TINY = 1e-12
VOID_SCALE = 1.0e-30
VOID_DENSITY = MAT_DENSITY_ATOMS_PER_BARN_CM * VOID_SCALE

L = CSDA_RANGE / RHO_G_CM3 # cm
N_LAYERS = int(L / dz)

# =============================================================================
# Exactly the same computational configuration as MCNP for Lockwood exp (from the MCNP VV slides)
# https://mcnp.lanl.gov/pdf_files/TechReport_2024_LANL_LA-UR-23-32743Rev.1_Kulesza.pdf
# =============================================================================
# pz 0.0
# pz 0.0233           - front foil thickness = 0.0233 cm
# pz 0.1233           - +0.1 cm void gap
# pz 0.12517037037    - calorimeter foil thickness = 0.00187037037 cm  (5.05e-3/2.7)
# pz 0.22517037037    - +0.1 cm void gap
# pz 5.0              - thick Al plate end
#
# cz 100.0            - radius 100 cm boundary
z0 = 0.0
z1 = 0.0233
z2 = 0.1233
z3 = 0.12517037037037037
z4 = 0.22517037037037037
z5 = 5.0

R_CYL = 100.0

# =============================================================================
# Set materials
# =============================================================================
mat_al = mcdc.Material(
    element_composition={MATERIAL_SYMBOL: MAT_DENSITY_ATOMS_PER_BARN_CM}
)

mat_void = mcdc.Material(
    element_composition={MATERIAL_SYMBOL: VOID_DENSITY}
)

# =============================================================================
# Surfaces
# =============================================================================
# Z-direction surfaces for layers

s0 = mcdc.surface("plane-z", z=z0, bc="vacuum")
s1 = mcdc.surface("plane-z", z=z1)
s2 = mcdc.surface("plane-z", z=z2)
s3 = mcdc.surface("plane-z", z=z3)
s4 = mcdc.surface("plane-z", z=z4)
s5 = mcdc.surface("plane-z", z=z5, bc="vacuum")

# =============================================================================
# Cells
# =============================================================================

# Front foil (Al): z0 < z < z1
mcdc.cell(+s0 & -s1, mat_al)

# Void gap (almost vacuum): z1 < z < z2
mcdc.cell(+s1 & -s2, mat_void)

# Calorimeter foil (Al): z2 < z < z3
mcdc.cell(+s2 & -s3, mat_al)

# Void gap: z3 < z < z4
mcdc.cell(+s3 & -s4, mat_void)

# Thick Al plate: z4 < z < z5
mcdc.cell(+s4 & -s5, mat_al)

# =============================================================================
# Set source
# =============================================================================
# Parallel beam of 1 MeV electrons entering at z=0, 0 degree incidence
theta = 0.0
mcdc.source(
    z=[z0 + 1e-8, z0 + 1e-8],
    particle_type="electron",
    energy=np.array([[1e6 - 1.0, 1e6 + 1.0], [0.5, 0.5]]),
    direction=[math.sin(theta), TINY, math.cos(theta)],
)

# =============================================================================
# Set tally
# =============================================================================
# Energy deposition tally along z-axis
z_bins = np.linspace(z4, z4+L, 101)

mcdc.tally.mesh_tally(
    scores=["edep", "flux"],
    z=z_bins
)

mcdc.tally.surface_tally(s0, scores=["net-current"])
mcdc.tally.surface_tally(s5, scores=["net-current"])


# =============================================================================
# Settings and run
# =============================================================================
settings = mcdc.Settings(
    N_particle=100,
    active_bank_buffer=10000
)

settings.save_input_deck = True
settings.output_name = f"lockwood_exp_setup_{datetime.now():%Y%m%d_%H%M%S}"
settings.debug_energy = False
settings.use_progress_bar = True

mcdc.run()