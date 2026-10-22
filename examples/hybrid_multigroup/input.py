import numpy as np

import mcdc

simulation = mcdc.Simulation("Hybrid multigroup sphere")

# Material with native H-1 data and a low-energy multigroup treatment
# MCDC_LIB must contain H1-293.6K.h5.
material = mcdc.Material(
    number_density={"H1": 5.0e-2},
    neutron_multigroup=mcdc.NeutronMultigroupData(
        capture=np.array([0.2]),
        scatter=np.array([[0.8]]),
        speed=np.array([1.383e6]),
        energy_grid=np.array([0.1, 10.0]),
        energy_representation="midpoint",
    ),
)

# Spherical domain
boundary = mcdc.Surface.Sphere(radius=5.0, boundary_condition="vacuum")
cell = mcdc.Cell(region=-boundary, fill=material)
simulation.set_model([cell])

# The low-energy source uses multigroup physics
multigroup_source = mcdc.Source(
    position=[0.0, 0.0, 1.0],
    isotropic=True,
    energy=1.0,
    probability=0.5,
)

# The high-energy source lies outside the multigroup grid and uses native physics
native_source = mcdc.Source(
    position=[0.0, 0.0, -1.0],
    isotropic=True,
    energy=1.0e6,
    probability=0.5,
)
simulation.set_sources([multigroup_source, native_source])

# Hybrid tallies use physical energy boundaries in eV
mesh = mcdc.MeshStructured(z=np.array([-5.0, 0.0, 5.0]))
tally = mcdc.Tally(
    name="hybrid_flux",
    mesh=mesh,
    particle_type="neutron",
    energy=np.array([0.0, 0.1, 10.0, 1.0e5, 20.0e6]),
    scores=["flux", "collision"],
)
simulation.set_tallies([tally])

simulation.settings.N_particle = 1_000
simulation.settings.N_batch = 5
simulation.settings.output_name = "hybrid_multigroup"

simulation.run()
