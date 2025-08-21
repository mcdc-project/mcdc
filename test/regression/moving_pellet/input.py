import numpy as np

import mcdc


# =============================================================================
# Materials
# =============================================================================

# Set materials
fuel = mcdc.MaterialMG(
    capture=np.array([0.5]),
    fission=np.array([0.5]),
    nu_p=np.array([2.5]),
    speed=np.array([200000.0]),
)

air = mcdc.MaterialMG(
    capture=np.array([0.002]),
    scatter=np.array([[0.008]]),
    speed=np.array([200000.0]),
)

# =============================================================================
# Set an assembly
# =============================================================================

# Surfaces
cylinder_z = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=1.0)
top_z = mcdc.surface("plane-z", z=9.0)
bot_z = mcdc.surface("plane-z", z=6.0)

# Move
cylinder_z.move([[-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], [2.0, 5.0, 1.0])
top_z.move([[0.0, 0.0, -2.0], [0.0, 0.0, 4.0], [0.0, 0.0, -10.0]], [5.0, 2.0, 1.0])
bot_z.move([[0.0, 0.0, -2.0], [0.0, 0.0, 4.0], [0.0, 0.0, -10.0]], [5.0, 2.0, 1.0])

# Set container cell surfaces
min_x = mcdc.surface("plane-x", x=-5.0, bc="vacuum")
max_x = mcdc.surface("plane-x", x=5.0, bc="vacuum")
min_y = mcdc.surface("plane-y", y=-5.0, bc="vacuum")
max_y = mcdc.surface("plane-y", y=5.0, bc="vacuum")
min_z = mcdc.surface("plane-z", z=-10.0, bc="vacuum")
max_z = mcdc.surface("plane-z", z=10.0, bc="vacuum")

# Make cells
fuel_pellet_region = +bot_z & -top_z & -cylinder_z
mcdc.cell(fuel_pellet_region, fuel)
mcdc.cell(
    ~fuel_pellet_region & +min_x & -max_x & +min_y & -max_y & +min_z & -max_z, air
)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source(x=[2.0, 3.0], y=[-0.5, 0.5], z=[-0.5, 0.5], time=[0.0, 9.0])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=5, active_bank_buffer=1000, N_batch=2)

mcdc.tally.mesh_tally(
    x=np.linspace(-5, 5, 201),
    z=np.linspace(-10, 10, 201),
    t=np.linspace(0, 9, 46),
    scores=["fission"],
)

mcdc.run()
"""
colors = {
    fuel: "red",
    air: "blue",
}
mcdc.visualize(
    "xz",
    y=0.0,
    x=[-5.0, 5.0],
    z=[-10, 10],
    pixel=(100, 100),
    colors=colors,
    time=np.linspace(0.0, 9.0, 19),
    save_as="figure",
)
"""
