import numpy as np
import mcdc

# ======================================================================================
# Set model
# ======================================================================================
# Three slab layers with different purely-absorbing materials

# Set materials
m1 = mcdc.MaterialMG(capture=np.array([1.0]))
m2 = mcdc.MaterialMG(capture=np.array([1.5]))
m3 = mcdc.MaterialMG(capture=np.array([2.0]))

# Set surfaces
s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneZ(z=2.0)
s3 = mcdc.Surface.PlaneZ(z=4.0)
s4 = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+s1 & -s2, fill=m2)
mcdc.Cell(region=+s2 & -s3, fill=m3)
mcdc.Cell(region=+s3 & -s4, fill=m1)

# ======================================================================================
# Set source
# ======================================================================================
# Uniform isotropic source throughout the domain

mcdc.Source(z=[0.0, 6.0], isotropic=True, energy_group=0)

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

# Tallies
mcdc.Tally(surface=s4, scores=["net-current"])
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
mcdc.Tally(mesh=mesh, mu=np.linspace(-1.0, 1.0, 32 + 1), scores=["flux", "collision"])

# Settings
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 2

# Run
mcdc.run()
