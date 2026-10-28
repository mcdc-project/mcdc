"""Coupled dose against collision kerma: a broad 6 MeV photon beam in water.

A monoenergetic beam enters a 40 cm water slab at normal incidence.
Reflective lateral faces make the beam infinitely broad; the front and back
faces are vacuum. On one depth mesh the deck scores the energy deposited,
D(z), and the photon track length binned in energy, phi(E, z).

Folding phi with mu_en gives the collision kerma
K_col(z) = sum_E phi(E, z) E mu_en(E). Past the build-up region (deeper than
the fastest secondary electron's range, about 3 cm here) D / K_col settles to
a constant beta slightly above 1. Every photon interaction, the electron and
positron transport, and the escape of bremsstrahlung and annihilation photons
all enter; the last is what makes D track the collision kerma rather than the
total kerma. At production statistics (32000 histories) beta = 1.0005 +/-
0.007 against NIST mu_en, constant with depth. This deck runs a handful of
histories for regression.
"""

import os

import numpy as np

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("Dose vs collision kerma, 6 MeV photons in water")

# ======================================================================================
# Set model
# ======================================================================================

ENERGY = 6.0e6  # eV
DEPTH = 40.0  # cm
HALF_WIDTH = 5.0  # cm

# Water at 1.0 g/cm^3:
#   n(H2O) = 1.0 g/cm^3 / 18.0153 g/mol * 6.02214076e23 /mol * 1e-24 = 0.033428
water = mcdc.Material(element_composition={"H": 0.066856, "O": 0.033428}, name="water")

x0 = mcdc.Surface.PlaneX(x=-HALF_WIDTH, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=HALF_WIDTH, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-HALF_WIDTH, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=HALF_WIDTH, boundary_condition="reflective")
z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
z1 = mcdc.Surface.PlaneZ(z=DEPTH, boundary_condition="vacuum")

slab = mcdc.Cell(region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=water, name="slab")
simulation.set_model([slab])

# ======================================================================================
# Set source
# ======================================================================================

# Broad parallel beam over the whole front face
source = mcdc.Source(
    x=[-HALF_WIDTH, HALF_WIDTH],
    y=[-HALF_WIDTH, HALF_WIDTH],
    z=1.0e-9,
    direction=[0.0, 0.0, 1.0],
    energy=ENERGY,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

# Depth mesh: fine through the build-up region, coarser beyond
z_mesh = np.r_[np.arange(0.0, 6.0, 0.2), np.arange(6.0, DEPTH + 1.0, 2.0)]

# Photon energy bins: logarithmic from 1 keV, with a narrow top bin holding the
# uncollided beam exactly at the source energy
energy_edges = np.r_[
    0.0, np.geomspace(1.0e3, ENERGY * (1.0 - 1.0e-6), 400), ENERGY * (1.0 + 1.0e-6)
]

mesh = mcdc.MeshStructured(z=z_mesh)
tallies = [
    mcdc.Tally(mesh=mesh, scores=["energy_deposition"], name="deposition"),
    mcdc.Tally(
        mesh=mesh,
        scores=["flux"],
        particle_type="photon",
        energy=energy_edges,
        name="photon_flux",
    ),
]
simulation.set_tallies(tallies)

simulation.settings.N_particle = 2
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 20261026
simulation.settings.photon_transport.active = True
simulation.settings.electron_transport.active = True

simulation.run()
