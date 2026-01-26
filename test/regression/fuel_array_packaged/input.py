import numpy as np
import mcdc

# ======================================================================================
# Materials
# ======================================================================================

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

# ======================================================================================
# The assembly
# ======================================================================================

# Surfaces
cylinder_z = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=1.0)
cylinder_x = mcdc.Surface.CylinderX(center=[0.0, 0.0], radius=1.0)

top_z = mcdc.Surface.PlaneZ(z=2.5)
bot_z = mcdc.Surface.PlaneZ(z=-2.5)
top_x = mcdc.Surface.PlaneX(x=2.5)
bot_x = mcdc.Surface.PlaneX(x=-2.5)

sphere = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=3.0)

# Cells
pellet_z = -cylinder_z & +bot_z & -top_z
pellet_x = -cylinder_x & +bot_x & -top_x
shooting_star = pellet_z | pellet_x
fuel_shooting_star = mcdc.Cell(region=shooting_star, fill=fuel)
cover_sphere = mcdc.Cell(region=-sphere & ~shooting_star, fill=cover)
water_tank = mcdc.Cell(region=+sphere, fill=water)

# ======================================================================================
# Copy the assembly via universe cells
# ======================================================================================

# Set the universe
assembly = mcdc.Universe(cells=[fuel_shooting_star, cover_sphere, water_tank])

# Set container cell surfaces
min_x = mcdc.Surface.PlaneX(x=-10.0, boundary_condition="vacuum")
mid_x = mcdc.Surface.PlaneX(x=0.0)
max_x = mcdc.Surface.PlaneX(x=10.0, boundary_condition="vacuum")
min_y = mcdc.Surface.PlaneY(y=-5.0, boundary_condition="vacuum")
max_y = mcdc.Surface.PlaneY(y=5.0, boundary_condition="vacuum")
min_z = mcdc.Surface.PlaneZ(z=-5.0, boundary_condition="vacuum")
max_z = mcdc.Surface.PlaneZ(z=5.0, boundary_condition="vacuum")

# Make copies via universe cells
container_left = +min_y & -max_y & +min_z & -max_z & +min_x & -mid_x
container_right = +min_y & -max_y & +min_z & -max_z & +mid_x & -max_x
assembly_left = mcdc.Cell(region=container_left, fill=assembly, translation=[-5, 0, 0])
assembly_right = mcdc.Cell(
    region=container_right, fill=assembly, translation=[+5, 0, 0], rotation=[0, 10, 0]
)

# Root universe
mcdc.simulation.set_root_universe(cells=[assembly_left, assembly_right])

# ======================================================================================
# Set source
# ======================================================================================

mcdc.Source(x=[-0.1, 0.1], isotropic=True, energy_group=0)

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.linspace(-10, 10, 201),
    z=np.linspace(-5, 5, 101),
)
mcdc.TallyMesh(mesh=mesh, scores=["fission"])

# Settings
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000

# Run (or visualize)
visualize = False
if not visualize:
    mcdc.run()
else:
    colors = {
        fuel: "red",
        cover: "gray",
        water: "blue",
    }
    mcdc.visualize(
        "xz", y=0.0, x=[-11.0, 11.0], z=[-6, 6], pixels=(400, 400), colors=colors
    )
