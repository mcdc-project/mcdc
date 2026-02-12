import numpy as np

import mcdc

# ======================================================================================
# Set model
# ======================================================================================

# Set materials
fuel = mcdc.MaterialMG(
    capture=np.array([0.5]),
    fission=np.array([0.25]),
    nu_p=np.array([1.5]),
    speed=np.array([200000.0]),
)
air = mcdc.MaterialMG(
    capture=np.array([0.002]),
    scatter=np.array([[0.008]]),
    speed=np.array([200000.0]),
)

# Set surfaces
cylinder_z = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=1.0)
top_z = mcdc.Surface.PlaneZ(z=9.0)
bot_z = mcdc.Surface.PlaneZ(z=6.0)

# Move surfaces
cylinder_z.move([[-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], [2.0, 5.0, 1.0])
top_z.move([[0.0, 0.0, -2.0], [0.0, 0.0, 4.0], [0.0, 0.0, -10.0]], [5.0, 2.0, 1.0])
bot_z.move([[0.0, 0.0, -2.0], [0.0, 0.0, 4.0], [0.0, 0.0, -10.0]], [5.0, 2.0, 1.0])

# Set container cell surfaces
min_x = mcdc.Surface.PlaneX(x=-5.0, boundary_condition="vacuum")
max_x = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
min_y = mcdc.Surface.PlaneY(y=-5.0, boundary_condition="vacuum")
max_y = mcdc.Surface.PlaneY(y=5.0, boundary_condition="vacuum")
min_z = mcdc.Surface.PlaneZ(z=-10.0, boundary_condition="vacuum")
max_z = mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum")

# Make cells
fuel_pellet_region = +bot_z & -top_z & -cylinder_z
mcdc.Cell(region=fuel_pellet_region, fill=fuel)
mcdc.Cell(
    region=~fuel_pellet_region & +min_x & -max_x & +min_y & -max_y & +min_z & -max_z,
    fill=air,
)

# ======================================================================================
# Set source
# ======================================================================================

mcdc.Source(
    x=[2.0, 3.0],
    y=[-0.5, 0.5],
    z=[-0.5, 0.5],
    isotropic=True,
    energy_group=0,
    time=[0.0, 9.0],
)

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.linspace(-5, 5, 201),
    z=np.linspace(-10, 10, 201),
)
mcdc.TallyMesh(mesh=mesh, scores=["fission"], time=np.linspace(0, 9, 46))

# Settings
mcdc.settings.N_particle = 100000
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000

# Run (or visualize)
visualize = False
if not visualize:
    mcdc.run()
else:
    colors = {
        fuel: "red",
        air: "blue",
    }
    mcdc.visualize(
        "xz",
        y=0.0,
        x=[-5.0, 5.0],
        z=[-10, 10],
        pixels=(100, 100),
        colors=colors,
        time=np.linspace(0.0, 9.0, 19),
        save_as="figure",
    )
