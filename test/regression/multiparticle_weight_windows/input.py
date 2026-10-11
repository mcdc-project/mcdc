import mcdc
from mcdc.constant import INF
import numpy as np
import os

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("Basic weight windows")

# Material
mat = mcdc.Material(element_composition={"Al": 1.0})

# Geometry
pitch = 1.25984
x0 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")
#
cell = mcdc.Cell(+x0 & -x1 & +y0 & -y1, fill=mat)
simulation.set_model([cell])

# Source
nsource = mcdc.Source(
    particle_type="neutron",
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    time=0.0,
    energy=14.1e6,
)
esource = mcdc.Source(
    particle_type="electron",
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    time=0.0,
    energy=1e6,
)
simulation.set_sources([nsource, esource])

# Setting
simulation.settings.N_particle = 10
simulation.settings.active_bank_buffer = 1000

# Mesh
Nx, Ny = 20, 20
x0, y0 = -pitch / 2, -pitch / 2
dx, dy = pitch / Nx, pitch / Ny
mesh = mcdc.MeshUniform(x=(x0, dx, Nx), y=(y0, dy, Ny))

# Tally
tally = mcdc.Tally(mesh=mesh, scores=["flux"])
simulation.set_tallies([tally])

# Weight windows
ww_array = np.ones((20, 20, 3))
# Actual bounds are set to arbitrary numbers
ww_array[..., 0] = 0.55  # Forces roulette on split particles from 1.0
ww_array[..., 1] = 0.7  # arbitrary in the middle
ww_array[..., 2] = 0.9  # forces splitting on all particles born with w=1.0
simulation.technique.weight_windows(ww_array, particle_type="neutron", mesh=mesh)
simulation.technique.weight_windows(ww_array, particle_type="electron", mesh=mesh)


simulation.run()
