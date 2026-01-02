import numpy as np

import mcdc


# ======================================================================================
# Set model
# ======================================================================================

# Set materials
air = mcdc.MaterialMG(
    capture=np.array([0.002]),
    scatter=np.array([[0.008]]),
    speed=np.array([200000.0]),
)

# Set container cell surfaces
min_x = mcdc.Surface.PlaneX(x=-5.0, boundary_condition="vacuum")
max_x = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
min_y = mcdc.Surface.PlaneY(y=-5.0, boundary_condition="vacuum")
max_y = mcdc.Surface.PlaneY(y=5.0, boundary_condition="vacuum")
min_z = mcdc.Surface.PlaneZ(z=-10.0, boundary_condition="vacuum")
max_z = mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum")

# Make cells
mcdc.Cell(region=+min_x & -max_x & +min_y & -max_y & +min_z & -max_z, fill=air)

# ======================================================================================
# Set source
# ======================================================================================

src = mcdc.Source(
    x=[-4.0, -3.0],
    y=[-0.5, 0.5],
    z=[-0.5, 0.5],
    isotropic=True,
    energy_group=0,
    time=[0.0, 10.0],
)
src.move([[1.0, 0.0, 0.0], [-0.5, 2.0, 0.0], [0.0, -3.0, 0.0]], [7.0, 2.0, 1.0])

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.linspace(-5.0, 5.0, 21),
    y=np.linspace(-5.0, 5.0, 21),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"], time=np.linspace(0, 10, 11))


# Settings
mcdc.settings.N_particle = 1000
mcdc.settings.N_batch = 2

# Run
mcdc.run()
