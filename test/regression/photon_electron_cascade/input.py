"""MeV photon/electron coupling in a narrow, vacuum-bounded lead needle.

The long axis gives photons an appreciable interaction probability, while
charged secondaries can escape through the narrow sides. This limits the
cascade cost without changing cross sections or transport cutoffs. The fixed
seed includes a low-energy positron that stops and annihilates, as well as
MeV charged secondaries. A separate bremsstrahlung regression checks the
transport of an emitted high-energy photon. Leakage bins isolate the
annihilation line and resolve the secondary photon/electron spectrum.

This is a transport regression, not a statistical Fano or dose/kerma benchmark.
"""

import os

import mcdc

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"
simulation = mcdc.Simulation("MeV coupling in a lead needle")
lead = mcdc.Material(number_density={"Pb": 0.03299}, name="lead")
width = 1.0e-4
surfaces = [
    mcdc.Surface.PlaneX(x=-width, boundary_condition="vacuum"),
    mcdc.Surface.PlaneX(x=width, boundary_condition="vacuum"),
    mcdc.Surface.PlaneY(y=-width, boundary_condition="vacuum"),
    mcdc.Surface.PlaneY(y=width, boundary_condition="vacuum"),
    mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum"),
    mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum"),
]
x0, x1, y0, y1, z0, z1 = surfaces
cell = mcdc.Cell(region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=lead)
simulation.set_model([cell])
simulation.compile()
simulation.set_sources(
    [
        mcdc.Source(
            position=[0.0, 0.0, 1.0e-9],
            direction=[0.0, 0.0, 1.0],
            energy=6.0e6,
            particle_type="photon",
        )
    ]
)
energy = [0.0, 1.0e3, 1.0e5, 510998.0, 511000.0, 1.0e6, 2.0e6, 6.0e6 + 1.0]
simulation.set_tallies(
    [
        mcdc.Tally(cell=cell, scores=["energy_deposition"], name="deposition"),
        *[
            mcdc.Tally(
                cell=cell,
                scores=["current-out"],
                energy=energy,
                particle_type=species,
                name=f"leakage_{species}",
            )
            for species in ["photon", "electron"]
        ],
    ]
)
simulation.settings.N_particle = 4
simulation.settings.N_batch = 2
# This seed reaches pair production and positron stopping/annihilation in
# eight histories. Preserve it when changing the workload.
simulation.settings.rng_seed = 7390
simulation.settings.photon_transport.active = True
simulation.settings.electron_transport.active = True
simulation.run()
