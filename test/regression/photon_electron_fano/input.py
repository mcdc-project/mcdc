"""Photon Fano test: lead at two densities in an infinite reflective medium.

Three layers along z -- wall, cavity, wall -- hold lead, the cavity at 1% of
the wall density. Every face is reflective, so the layers repeat into an
infinite periodic medium. A uniform isotropic 100 keV photon source emits in
proportion to mass, the same number of photons per gram in every layer.

By Fano's theorem, with a uniform source per unit mass and cross sections
per unit mass that do not depend on density, the fluence of every species is
independent of density. So per source photon the dose in each layer is
E0 / M_total, the photon and electron fluences are the same in every layer,
and the total deposition is E0. Nothing escapes, so the regression exercises
photoelectric and Compton electron production, fluorescence, and electron
energy deposition.

The source energy is reduced to keep Python regression runs short; pair
production and annihilation are covered by the coupling unit tests.
"""

import os

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("Photon Fano test in lead")

# ======================================================================================
# Set model
# ======================================================================================

ENERGY = 1.0e5  # eV
HALF_WIDTH = 0.5  # cm; physically irrelevant, the lateral faces reflect

# Lead: n = rho / 207.2 g/mol * 6.02214076e23 /mol * 1e-24 atoms/barn-cm
WALL_DENSITY = 11.35  # g/cm^3
CAVITY_DENSITY = 0.1135  # g/cm^3
LAYERS = [
    # name, density (g/cm^3), z_min, z_max (cm)
    ("wall_front", WALL_DENSITY, 0.0, 0.3),
    ("cavity", CAVITY_DENSITY, 0.3, 4.3),
    ("wall_back", WALL_DENSITY, 4.3, 4.6),
]


def lead(density, name):
    n = density / 207.2 * 6.02214076e23 * 1.0e-24
    return mcdc.Material(number_density={"Pb": n}, name=name)


x0 = mcdc.Surface.PlaneX(x=-HALF_WIDTH, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=HALF_WIDTH, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-HALF_WIDTH, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=HALF_WIDTH, boundary_condition="reflective")
lateral = +x0 & -x1 & +y0 & -y1

z_bounds = [layer[2] for layer in LAYERS] + [LAYERS[-1][3]]
z_planes = [
    mcdc.Surface.PlaneZ(
        z=z, boundary_condition="reflective" if i in (0, len(LAYERS)) else "none"
    )
    for i, z in enumerate(z_bounds)
]

cells = []
for i, (name, density, _, _) in enumerate(LAYERS):
    cells.append(
        mcdc.Cell(
            region=lateral & +z_planes[i] & -z_planes[i + 1],
            fill=lead(density, name),
            name=name,
        )
    )
simulation.set_model(cells)

# ======================================================================================
# Set source
# ======================================================================================

# One uniform isotropic source per layer, weighted by the layer's mass
area = (2.0 * HALF_WIDTH) ** 2
sources = [
    mcdc.Source(
        x=[-HALF_WIDTH, HALF_WIDTH],
        y=[-HALF_WIDTH, HALF_WIDTH],
        z=[z_min, z_max],
        isotropic=True,
        energy=ENERGY,
        particle_type="photon",
        probability=density * area * (z_max - z_min),
    )
    for _, density, z_min, z_max in LAYERS
]
simulation.set_sources(sources)

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

tallies = []
for cell, (name, _, _, _) in zip(cells, LAYERS):
    tallies.append(
        mcdc.Tally(cell=cell, scores=["energy_deposition"], name=f"{name}_deposition")
    )
    for particle_type in ["photon", "electron"]:
        tallies.append(
            mcdc.Tally(
                cell=cell,
                scores=["flux"],
                particle_type=particle_type,
                name=f"{name}_{particle_type}_flux",
            )
        )
simulation.set_tallies(tallies)

simulation.settings.N_particle = 1
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 20261026
simulation.settings.photon_transport.active = True
simulation.settings.electron_transport.active = True

simulation.run()
