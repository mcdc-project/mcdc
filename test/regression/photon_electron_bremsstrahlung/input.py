"""Transport of MeV-electron bremsstrahlung in a vacuum-bounded lead needle.

A fixed seed gives a 6 MeV source electron a bremsstrahlung collision near
its starting point, emitting a 666 keV photon. The narrow transverse extent
lets charged particles escape quickly. Photon flux and leakage retain a
signal from the emitted photon's subsequent transport. Real library data
and normal transport cutoffs are used throughout.
"""

import os

import mcdc

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"
simulation = mcdc.Simulation("MeV electron bremsstrahlung in a lead needle")
lead = mcdc.Material(number_density={"Pb": 0.03299}, name="lead")
width = 1.0e-6
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
            particle_type="electron",
        )
    ]
)
energy = [0.0, 1.0e3, 1.0e5, 510998.0, 511000.0, 1.0e6, 2.0e6, 6.0e6 + 1.0]
simulation.set_tallies(
    [
        mcdc.Tally(
            cell=cell,
            scores=["flux"],
            particle_type="photon",
            energy=energy,
            name="photon_flux",
        ),
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
simulation.settings.N_particle = 1
simulation.settings.N_batch = 2
# This seed emits a 666 keV bremsstrahlung photon in the first electron
# collision. Preserve it when changing the workload.
simulation.settings.rng_seed = 4875
simulation.settings.photon_transport.active = True
simulation.settings.electron_transport.active = True
simulation.run()
