import numpy as np

import mcdc

# ======================================================================================
# Set model
# ======================================================================================
# Finite homogeneous pure-absorbing slab

# Set materials
m = mcdc.MaterialMG(capture=np.array([1.0]))

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+s1 & -s2, fill=m)

# ======================================================================================
# Set source
# ======================================================================================
# Isotropic beam from left-end

mcdc.Source(
    position=(0.0, 0.0, 0.0),
    white_direction=(1.0, 0.0, 0.0),
    energy_group=0,
    time=[0.0, 5.0],
)

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshUniform(x=(0.0, 0.1, 50))
mcdc.Tally(
    mesh=mesh,
    scores=["flux"],
    time=np.linspace(0.0, 5.0, 51),
)

# Settings
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 2

# Run
mcdc.run()
