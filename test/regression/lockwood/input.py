import numpy as np
import os
import math
import mcdc
import h5py
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from mcdc.constant import NEUTRON_MASS_AMU

# Isolate electron data so isotope files cannot supply the elemental mass.
library = TemporaryDirectory()
source_library = Path(__file__).resolve().parents[1] / "mcdc-regression_test_data"
copyfile(source_library / "electron-Al.h5", Path(library.name) / "electron-Al.h5")
os.environ["MCDC_LIB"] = library.name

# Create MC/DC simulation
simulation = mcdc.Simulation("Lockwood")

# =============================================================================
# Set problem parameters
# =============================================================================
# Energy and Angle Parameters
MATERIAL_SYMBOL = "Al"
ENERGY = 1e6  # eV
CSDA_RANGE = 0.569  # g/cm2
ANGLE = 0.0

# MCDC Simulation Parameters
N_PARTICLES = 2
z0 = 0.0  # Starting source position

# Material Properties
RHO_G_CM3 = 2.70  # g/cm3
with h5py.File(Path(library.name) / "electron-Al.h5") as file:
    ATOMIC_WEIGHT_G_MOL = float(file["atomic-weight-ratio"][()]) * NEUTRON_MASS_AMU
AREAL_DENSITY_G_CM2 = 5.05e-3  # g/cm2

# Standard Calculations
dz = AREAL_DENSITY_G_CM2 / RHO_G_CM3
AVAGADRO_NUMBER = 6.02214076e23  # atoms/mol
MAT_DENSITY_ATOMS_PER_BARN_CM = (
    AVAGADRO_NUMBER / ATOMIC_WEIGHT_G_MOL * RHO_G_CM3 / 1e24
)  # atoms/barn-cm
TINY = 1e-30
# A thin target exercises electron transport and deposition without full slowing down.
L = dz / 10  # cm
N_LAYERS = 4
THETA = math.radians(ANGLE)

# Output variables for naming
np_name = len(str(N_PARTICLES)) - 1
e_name = f"{ENERGY:.2g}"

# =============================================================================
# Set materials
# =============================================================================
mat_number_density = mcdc.Material(
    number_density={MATERIAL_SYMBOL: MAT_DENSITY_ATOMS_PER_BARN_CM}
)
mat_mass_density = mcdc.Material(mass_density={MATERIAL_SYMBOL: RHO_G_CM3})

# =============================================================================
# Set geometry (surfaces and cells)
# =============================================================================
# Z-direction surfaces for layers

s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
midplane = mcdc.Surface.PlaneZ(z=0.5 * L)
s2 = mcdc.Surface.PlaneZ(z=L, boundary_condition="vacuum")

first_half = mcdc.Cell(region=+s1 & -midplane, fill=mat_number_density)
second_half = mcdc.Cell(region=+midplane & -s2, fill=mat_mass_density)
simulation.set_model([first_half, second_half])

# =============================================================================
# Set source
# =============================================================================
# Parallel beam of 1 MeV electrons entering at z=0

source = mcdc.Source(
    z=[z0 + TINY, z0 + TINY],
    particle_type="electron",
    energy=np.array([[ENERGY - 1, ENERGY + 1], [0.5, 0.5]]),
    direction=[math.sin(THETA), 0.0 + TINY, math.cos(THETA)],
)
simulation.set_sources([source])

# =============================================================================
# Set tally
# =============================================================================
# Energy deposition tally along z-axis
z_bins = np.linspace(0.0, L, N_LAYERS + 1)
mesh = mcdc.MeshStructured(z=z_bins)

edep_tally = mcdc.Tally(name="edep", mesh=mesh, scores=["energy_deposition"])
flux_tally = mcdc.Tally(name="flux", scores=["flux"], mesh=mesh)
s1_current = mcdc.Tally(name="s1_current", surface=s1, scores=["current-net"])
s2_current = mcdc.Tally(name="s2_current", surface=s2, scores=["current-net"])
simulation.set_tallies([edep_tally, flux_tally, s1_current, s2_current])

# =============================================================================
# Settings and run
# =============================================================================

simulation.settings.N_particle = N_PARTICLES
simulation.settings.active_bank_buffer = N_PARTICLES * 1000

simulation.compile()
np.testing.assert_allclose(
    mat_mass_density.element_densities,
    mat_number_density.element_densities,
    rtol=1e-12,
    atol=0.0,
)
assert not simulation.settings.neutron_transport.active
assert not simulation.settings.photon_transport.active
assert not simulation.nuclides
simulation.run()
library.cleanup()
