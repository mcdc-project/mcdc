import numpy as np
import os
import mcdc

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# ======================================================================================
# Set model
# ======================================================================================
# Finite homogeneous pure-absorbing slab

# Set materials
# Set materials
generate_material = lambda atomdensity: mcdc.Material(
    nuclide_composition={"H1": atomdensity}
)
m1 = generate_material(0.0001)

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")

# Set cells
low_xs_cell = mcdc.Cell(region=+s1 & -s2, fill=m1)

# ======================================================================================
# Set source
# ======================================================================================
# Isotropic beam from left-end

mcdc.Source(position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
# energy deposition for actual VR against analog
mesh = mcdc.MeshUniform()
mcdc.Tally(
    mesh=mesh,
    scores=["energy_deposition"],
)
# flux to make sure tracklength is unbiased
mcdc.Tally(cell=low_xs_cell, scores=["flux"])
# net-current to make sure surface is unbiased
mcdc.Tally(surface=s2, scores=["net-current"])

# Settings
mcdc.settings.N_particle = 5000
mcdc.settings.N_batch = 2

mcdc.simulation.forced_collisions(cells=[low_xs_cell])

# Run
mcdc.run()
