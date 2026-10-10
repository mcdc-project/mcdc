"""Energy spectrum in concentric multi-material spheres.

A 1-10 MeV uniform source spectrum, tallied into 40 logarithmic energy
bins per region. The energy filter is what makes this deck a test of the
scattered spectrum rather than just the total flux, so it is sensitive
to the Klein-Nishina sampling in a way the integral decks are not.

Migrated from ``photon_transport_code/MCNP_Verification_Tests/Complex_M&G/multi_material_spheres_1to10mev_spectrum.py``.
The physics is unchanged; the API is not. Energies are eV rather than MeV,
times are seconds rather than nanoseconds, materials are element compositions
rather than ``PhotonMaterial``, and settings hang off the ``Simulation``.
"""

import numpy as np
import os

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

simulation = mcdc.Simulation("Multi-material sphere spectrum")

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

RADII = [5.0, 10.0, 15.0, 20.0, 25.0]
SHELL_MATERIALS = [lead, water, concrete, air, concrete]

spheres = [
    mcdc.Surface.Sphere(
        center=[0.0, 0.0, 0.0],
        radius=radius,
        boundary_condition="vacuum" if index == len(RADII) - 1 else "none",
    )
    for index, radius in enumerate(RADII)
]

cells = [mcdc.Cell(region=-spheres[0], fill=SHELL_MATERIALS[0])]
for index in range(1, len(spheres)):
    cells.append(
        mcdc.Cell(
            region=+spheres[index - 1] & -spheres[index],
            fill=SHELL_MATERIALS[index],
        )
    )

simulation.set_model(cells)

# ======================================================================================
# Set source
# ======================================================================================

ENERGY_MIN = 1.0e6
ENERGY_MAX = 1.0e7

source = mcdc.Source(
    x=0.0,
    y=0.0,
    z=0.0,
    energy=np.array([[ENERGY_MIN, ENERGY_MAX], [1.0, 1.0]]),
    isotropic=True,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

ENERGY_EDGES = np.logspace(np.log10(1.0e4), np.log10(ENERGY_MAX), 41)

simulation.set_tallies(
    [
        mcdc.Tally(
            cell=cell,
            scores=["flux"],
            energy=ENERGY_EDGES,
            name=f"region_{index}",
            particle_type="photon",
        )
        for index, cell in enumerate(cells)
    ]
)

simulation.settings.N_particle = 1000
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 42

simulation.run()
