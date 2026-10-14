"""Collimated 1 MeV beam into a lead-water-concrete-air slab stack.

Mono-directional rather than isotropic, so every history starts with
the same direction and the profile is a clean uncollided-beam
attenuation with scatter buildup on top. One cell tally per slab.

Migrated from ``photon_transport_code/MCNP_Verification_Tests/Complex_M&G/multi_material_slabs_collimated_beam.py``.
The physics is unchanged; the API is not. Energies are eV rather than MeV,
times are seconds rather than nanoseconds, materials are element compositions
rather than ``PhotonMaterial``, and settings hang off the ``Simulation``.
"""

import numpy as np
import os

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

simulation = mcdc.Simulation("Collimated beam into a slab stack")

# ======================================================================================
# Material compositions
# ======================================================================================
# n_i = rho [g/cm^3] * w_i * (N_A * 1e-24) / A_i,  with N_A * 1e-24 = 0.602214,
# giving atom densities in 10^24 atoms/cm^3 -- the unit macro_xs assumes, so no
# barn-to-cm^2 conversion appears anywhere.

AVOGADRO_BARN = 0.602214

MOLAR_MASS = {
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "Ar": 39.948,
    "Ca": 40.078,
    "Fe": 55.845,
    "Pb": 207.200,
}


def composition(density, weight_fractions):
    """Convert a bulk density and weight fractions to an element composition."""
    return {
        symbol: density * fraction * AVOGADRO_BARN / MOLAR_MASS[symbol]
        for symbol, fraction in weight_fractions.items()
    }


lead = mcdc.Material(element_composition=composition(11.34, {"Pb": 1.0}), name="lead")
water = mcdc.Material(
    element_composition=composition(1.0, {"H": 0.111898, "O": 0.888102}), name="water"
)
concrete = mcdc.Material(
    element_composition=composition(
        2.3,
        {
            "O": 0.529107,
            "Na": 0.016,
            "Mg": 0.002,
            "Al": 0.033872,
            "Si": 0.337021,
            "Ca": 0.044,
            "Fe": 0.014,
            "H": 0.010,
        },
    ),
    name="concrete",
)
air = mcdc.Material(
    element_composition=composition(0.00129, {"N": 0.7818, "O": 0.2097, "Ar": 0.0085}),
    name="air",
)

# ======================================================================================
# Set model
# ======================================================================================
# Lead, water, concrete and air slabs, with a thin vacuum-bounded air backing
# behind the source face.

SLABS = [
    (lead, 0.0, 5.0),
    (water, 5.0, 25.0),
    (concrete, 25.0, 55.0),
    (air, 55.0, 200.0),
]
Z_BACK_BOUNDARY = -1.0
Z_SOURCE = 1.0e-6

materials = [material for material, _, _ in SLABS]
z_backs = [z_back for _, _, z_back in SLABS]

z_back_bound = mcdc.Surface.PlaneZ(z=Z_BACK_BOUNDARY, boundary_condition="vacuum")
z_planes = [mcdc.Surface.PlaneZ(z=SLABS[0][1])]
for index, z_back in enumerate(z_backs):
    z_planes.append(
        mcdc.Surface.PlaneZ(
            z=z_back,
            boundary_condition="vacuum" if index == len(z_backs) - 1 else "none",
        )
    )

backing = mcdc.Cell(region=+z_back_bound & -z_planes[0], fill=air, name="backing_gap")
cells = [
    mcdc.Cell(region=+z_planes[index] & -z_planes[index + 1], fill=materials[index])
    for index in range(len(SLABS))
]

simulation.set_model([backing] + cells)

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(
    x=0.0,
    y=0.0,
    z=Z_SOURCE,
    direction=[0.0, 0.0, 1.0],
    energy=1.0e6,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

simulation.set_tallies(
    [
        mcdc.Tally(cell=cell, scores=["flux"], name=f"slab_{index}")
        for index, cell in enumerate(cells)
    ]
)

simulation.settings.N_particle = 1000
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 42

simulation.run()
