"""10 MeV photon point source in lead, track-length flux by shell.

A monoenergetic isotropic 10 MeV photon point source sits at the origin of an
aluminium medium. A 40 cm sphere is divided into twenty 2 cm concentric
regions, each carrying its own track-length (MCNP F4-style) flux tally.

Migrated from
``photon_transport_code/MCNP_Verification_Tests/MCNP_test_problems/10mev_pb_spheres.py``.
The physics is unchanged; what changed is the API. Energies are eV rather than
MeV, the material is an element composition rather than a ``PhotonMaterial``,
and settings hang off the ``Simulation`` rather than the module.
"""

import numpy as np
import os

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("10 MeV photon source in lead")

# ======================================================================================
# Set model
# ======================================================================================

# Lead at 11.35 g/cm^3:
#   n = 11.35 g/cm^3 * 6.022e23 /mol / 207.2 g/mol * 1e-24 = 0.03299 atoms/barn-cm
lead = mcdc.Material(element_composition={"Pb": 0.03299}, name="lead")

RADII = [float(r) for r in range(2, 41, 2)]

spheres = []
for radius in RADII:
    boundary_condition = "vacuum" if radius == RADII[-1] else "none"
    spheres.append(
        mcdc.Surface.Sphere(
            center=[0.0, 0.0, 0.0],
            radius=radius,
            boundary_condition=boundary_condition,
        )
    )

cells = [mcdc.Cell(region=-spheres[0], fill=lead)]
for index in range(1, len(spheres)):
    cells.append(mcdc.Cell(region=+spheres[index - 1] & -spheres[index], fill=lead))

simulation.set_model(cells)

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(
    x=0.0,
    y=0.0,
    z=0.0,
    energy=10.0e6,
    isotropic=True,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

# One track-length flux tally per region, named so results read back in radial
# order. Divide by the cell volume to recover the scalar flux in 1/cm^2 per
# source photon.
tallies = [
    mcdc.Tally(cell=cell, scores=["flux"], name=f"region_{index}")
    for index, cell in enumerate(cells)
]
simulation.set_tallies(tallies)

simulation.settings.N_particle = 500
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 42

simulation.run()
