import numpy as np
import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Kornreich")

# ======================================================================================
# Set model
# ======================================================================================
# Based on Kornreich, ANE 2004, 31, 1477-1494,
# DOI: 10.1016/j.anucene.2004.03.012

# Set materials
m1 = mcdc.Material.multigroup(
    capture=np.array([0.0]),
    scatter=np.array([[0.9]]),
    fission=np.array([0.1]),
    nu_p=np.array([6.0]),
)
m2 = mcdc.Material.multigroup(
    capture=np.array([0.68]),
    scatter=np.array([[0.2]]),
    fission=np.array([0.12]),
    nu_p=np.array([2.5]),
)

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneX(x=1.5)
s3 = mcdc.Surface.PlaneX(x=2.5, boundary_condition="vacuum")

# Set cells
cell_1 = mcdc.Cell(region=+s1 & -s2, fill=m1)
cell_2 = mcdc.Cell(region=+s2 & -s3, fill=m2)
simulation.set_model([cell_1, cell_2])

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(x=[0.0, 2.5], isotropic=True, energy=0)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.array(
        [
            0.0,
            0.15,
            0.3,
            0.45,
            0.6,
            0.75,
            0.9,
            1.05,
            1.2,
            1.35,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
        ]
    )
)
tally = mcdc.Tally(mesh=mesh, scores=["flux"])
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 100
simulation.settings.N_batch = 2
simulation.settings.census_bank_buffer_ratio = 3.0
simulation.settings.source_bank_buffer_ratio = 3.0
simulation.settings.set_eigenmode(N_inactive=1, N_active=2, gyration_radius="only-x")

# Run
simulation.run()
