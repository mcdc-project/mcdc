import numpy as np

import mcdc


# =============================================================================
# Materials
# =============================================================================

# Set materials
fuel = mcdc.MaterialMG(
    capture=np.array([0.45]),
    fission=np.array([0.55]),
    nu_p=np.array([2.5]),
)

cover = mcdc.MaterialMG(
    capture=np.array([0.05]),
    scatter=np.array([[0.95]]),
)

water = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.08]]),
)

# =============================================================================
# Set an assembly
# =============================================================================

# Surfaces
cylinder_z = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=1.0)
cylinder_x = mcdc.surface("cylinder-x", center=[0.0, 0.0], radius=1.0)

top_z = mcdc.surface("plane-z", z=2.5)
bot_z = mcdc.surface("plane-z", z=-2.5)
top_x = mcdc.surface("plane-x", x=2.5)
bot_x = mcdc.surface("plane-x", x=-2.5)

sphere = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=3.0)

# Cells
pellet_z = -cylinder_z & +bot_z & -top_z
pellet_x = -cylinder_x & +bot_x & -top_x
shooting_star = pellet_z | pellet_x
fuel_shooting_star = mcdc.cell(shooting_star, fuel)
cover_sphere = mcdc.cell(-sphere & ~shooting_star, cover)
water_tank = mcdc.cell(+sphere, water)

# =============================================================================
# Copy the assembly via universe cells
# =============================================================================

# Set the universe
assembly = mcdc.universe([fuel_shooting_star, cover_sphere, water_tank])

# Set container cell surfaces
min_x = mcdc.surface("plane-x", x=-10.0, bc="vacuum")
mid_x = mcdc.surface("plane-x", x=0.0)
max_x = mcdc.surface("plane-x", x=10.0, bc="vacuum")
min_y = mcdc.surface("plane-y", y=-5.0, bc="vacuum")
max_y = mcdc.surface("plane-y", y=5.0, bc="vacuum")
min_z = mcdc.surface("plane-z", z=-5.0, bc="vacuum")
max_z = mcdc.surface("plane-z", z=5.0, bc="vacuum")

# Make copies via universe cells
container_left = +min_y & -max_y & +min_z & -max_z & +min_x & -mid_x
container_right = +min_y & -max_y & +min_z & -max_z & +mid_x & -max_x
assembly_left = mcdc.cell(container_left, assembly, (-5, 0, 0))
assembly_right = mcdc.cell(container_right, assembly, (+5, 0, 0), (0, 10, 0))

# Root universe
mcdc.universe([assembly_left, assembly_right], root=True)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source()

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=100, active_bank_buffer=1000, N_batch=2)

mcdc.tally.mesh_tally(
    x=np.linspace(-10, 10, 201),
    z=np.linspace(-5, 5, 101),
    scores=["fission"],
)

mcdc.run()
"""
colors = {
    fuel: 'red',
    cover: 'gray',
    water: 'blue',
}
mcdc.visualize('xz', y=0.0, x=[-11., 11.], z=[-6, 6], pixel=(400, 400), colors=colors)
"""
