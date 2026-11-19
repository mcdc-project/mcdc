import numpy as np

import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Moving pellet")

# ======================================================================================
# Set model
# ======================================================================================

# Set materials
fuel = mcdc.Material.multigroup(
    capture=np.array([0.5]),
    fission=np.array([0.25]),
    nu_p=np.array([1.0]),
    nu_d=np.array([[0.5]]),
    decay_rate=np.array([1.0]),
    speed=np.array([200000.0]),
)
air = mcdc.Material.multigroup(
    capture=np.array([0.002]),
    scatter=np.array([[0.008]]),
    speed=np.array([200000.0]),
)

# Set surfaces
cylinder_z = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=1.0)
top_z = mcdc.Surface.PlaneZ(z=9.0)
bot_z = mcdc.Surface.PlaneZ(z=6.0)

# Move the pellet surfaces with one shared translation schedule.
velocities = [[-0.5, 0.0, -2.0], [1.0, 0.0, -2.0], [1.0, 0.0, 4.0], [-2.0, 0.0, -10.0]]
durations = [2.0, 3.0, 2.0, 1.0]
for surface in (cylinder_z, top_z, bot_z):
    surface.move(velocities, durations)

# Set container cell surfaces
min_x = mcdc.Surface.PlaneX(x=-5.0, boundary_condition="vacuum")
max_x = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
min_y = mcdc.Surface.PlaneY(y=-5.0, boundary_condition="vacuum")
max_y = mcdc.Surface.PlaneY(y=5.0, boundary_condition="vacuum")
min_z = mcdc.Surface.PlaneZ(z=-10.0, boundary_condition="vacuum")
max_z = mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum")

# Make cells
fuel_pellet_region = +bot_z & -top_z & -cylinder_z
fuel_cell = mcdc.Cell(region=fuel_pellet_region, fill=fuel)
# Delayed-neutron precursors follow the fuel until their sampled emission time.
fuel_cell.move_product(velocities, durations)
air_cell = mcdc.Cell(
    region=~fuel_pellet_region & +min_x & -max_x & +min_y & -max_y & +min_z & -max_z,
    fill=air,
)
simulation.set_model([fuel_cell, air_cell])

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(
    x=[2.0, 3.0],
    y=[-0.5, 0.5],
    z=[-0.5, 0.5],
    isotropic=True,
    energy=0,
    time=[0.0, 9.0],
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.linspace(-5, 5, 101),
    z=np.linspace(-10, 10, 101),
)
tally = mcdc.Tally(mesh=mesh, scores=["fission"], time=np.linspace(0, 9, 46))
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 50
simulation.settings.N_batch = 2
simulation.settings.active_bank_buffer = 1000

# Run (or visualize)
visualize = False
if not visualize:
    simulation.run()
else:
    colors = {
        fuel: "red",
        air: "blue",
    }
    simulation.visualize_model(
        vis_plane="xz",
        y=0.0,
        x=[-5.0, 5.0],
        z=[-10, 10],
        pixels=(100, 100),
        colors=colors,
        time=np.linspace(0.0, 9.0, 19),
        save_as="figure",
    )
