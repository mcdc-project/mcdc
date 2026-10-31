"""Coupled energy-deposition and leakage regression in a thin lead slab.

A 50 keV pencil beam exercises photon/electron energy transfer and outward
current tallies without the expensive MeV secondary cascade. The thin slab
allows leakage as well as deposition. Pair production and positron
annihilation are covered separately by the coupling unit tests.
"""

import os

import numpy as np

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("Energy ledger, 50 keV photons on lead")

# ======================================================================================
# Set model
# ======================================================================================

ENERGY = 5.0e4  # eV
THICKNESS = 0.001  # cm
HALF_WIDTH = 5.0  # cm

# Lead at 11.35 g/cm^3:
#   n = 11.35 g/cm^3 / 207.2 g/mol * 6.02214076e23 /mol * 1e-24 = 0.03299 atoms/barn-cm
lead = mcdc.Material(number_density={"Pb": 0.03299}, name="lead")

x0 = mcdc.Surface.PlaneX(x=-HALF_WIDTH, boundary_condition="vacuum")
x1 = mcdc.Surface.PlaneX(x=HALF_WIDTH, boundary_condition="vacuum")
y0 = mcdc.Surface.PlaneY(y=-HALF_WIDTH, boundary_condition="vacuum")
y1 = mcdc.Surface.PlaneY(y=HALF_WIDTH, boundary_condition="vacuum")
z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
z1 = mcdc.Surface.PlaneZ(z=THICKNESS, boundary_condition="vacuum")

slab = mcdc.Cell(region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=lead, name="slab")
simulation.set_model([slab])

# Resolve the slab's bounding surfaces for the cell current tallies
simulation.compile()

# ======================================================================================
# Set source
# ======================================================================================

# Pencil beam entering the front face along +z
source = mcdc.Source(
    position=[0.0, 0.0, 1.0e-9],
    direction=[0.0, 0.0, 1.0],
    energy=ENERGY,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

# 1 keV bins with edges at (k - 0.5) keV
n_bins = int(np.ceil(ENERGY / 1.0e3)) + 1
energy_edges = (np.arange(n_bins + 1) - 0.5) * 1.0e3
energy_edges[0] = 0.0

tallies = [mcdc.Tally(cell=slab, scores=["energy_deposition"], name="deposition")]
for particle_type in ["photon", "electron"]:
    tallies.append(
        mcdc.Tally(
            cell=slab,
            scores=["current-out"],
            particle_type=particle_type,
            energy=energy_edges,
            name=f"leakage_{particle_type}",
        )
    )
simulation.set_tallies(tallies)

# Enough histories to record both photon and electron leakage with this seed.
simulation.settings.N_particle = 32
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 20261026
simulation.settings.photon_transport.active = True
simulation.settings.electron_transport.active = True

simulation.run()
