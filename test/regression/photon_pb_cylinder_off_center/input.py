"""1 MeV photon source off-centre in a finite lead cylinder.

The source sits at z = -15 cm, inside the lowest axial segment, so the
flux map is strongly asymmetric in z and most segments are reached only
by scattered photons. That makes it a sharper test of the scattering
treatment than a centred source would be.

Migrated from ``photon_transport_code/MCNP_Verification_Tests/Complex_M&G/lead_finite_cylinder_off_center.py``.
The physics is unchanged; the API is not. Energies are eV rather than MeV,
times are seconds rather than nanoseconds, materials are element compositions
rather than ``PhotonMaterial``, and settings hang off the ``Simulation``.
"""

import numpy as np
import os

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

simulation = mcdc.Simulation("Off-centre source in a lead cylinder")

# ======================================================================================
# Set model
# ======================================================================================
# Finite lead cylinder, four radial shells by four axial segments.

lead = mcdc.Material(element_composition={"Pb": 0.03299}, name="lead")

RADII = [5.0, 10.0, 15.0, 20.0]
Z_BOUNDS = [-20.0, -10.0, 0.0, 10.0, 20.0]
N_R = len(RADII)
N_Z = len(Z_BOUNDS) - 1

cylinders = [
    mcdc.Surface.CylinderZ(
        center=[0.0, 0.0],
        radius=radius,
        boundary_condition="vacuum" if index == N_R - 1 else "none",
    )
    for index, radius in enumerate(RADII)
]

z_planes = [
    mcdc.Surface.PlaneZ(
        z=z,
        boundary_condition=("vacuum" if index in (0, len(Z_BOUNDS) - 1) else "none"),
    )
    for index, z in enumerate(Z_BOUNDS)
]

cells = {}
for ir in range(N_R):
    radial = -cylinders[0] if ir == 0 else +cylinders[ir - 1] & -cylinders[ir]
    for iz in range(N_Z):
        axial = +z_planes[iz] & -z_planes[iz + 1]
        cells[(ir, iz)] = mcdc.Cell(
            region=radial & axial, fill=lead, name=f"r{ir}_z{iz}"
        )

simulation.set_model(list(cells.values()))

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(
    x=0.0,
    y=0.0,
    z=-15.0,
    energy=1.0e6,
    isotropic=True,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

simulation.set_tallies(
    [
        mcdc.Tally(cell=cells[(ir, iz)], scores=["flux"], name=f"r{ir}_z{iz}_flux")
        for ir in range(N_R)
        for iz in range(N_Z)
    ]
)

simulation.settings.N_particle = 1000
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 42

simulation.run()
