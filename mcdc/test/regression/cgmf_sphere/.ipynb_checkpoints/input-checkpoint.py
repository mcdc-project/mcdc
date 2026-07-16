## MCNP vs MCDC examples
# U-235 Sphere
import mcdc
import numpy as np

#===========
# Set Model
#===========

U_235 = mcdc.Material(name='U_235',nuclide_composition={'U235',1.0})

# Set Surfaces
# 
sphere = mcdc.Surface.Sphere(center=[0, 0, 0], radius=10, boundary_condition="vacuum")
inside_sphere = -sphere
sphere_cell = mcdc.Cell(region=inside_sphere, fill=U_235)

# Set Source
mcdc.Source(x=[0.0, 2.5], isotropic=True, energy=14e6) # energy in ev

# set mesh: circular slice in x-z plane
theta = np.linspace(0,pi,50)
phi = 0
r = np.linspace(0,10,50)

x = r * np.cos(phi)*np.sin(theta)
y = 0
z = r*np.cos(theta)

mesh = np.meshgrid(x,z)

E_1 = np.linspace(1e-4,1,100) # thermal energy axis
E_2 = np.linspace(200,1e5,1000)
E_3 = np.linspace(1.1e5,14e6,1000)
E_axis = []
E_axis.append(E_1,E_2,E_3)

# tallies

mcdc.Tally(
    mesh=mesh,
    scores=["flux"],
    energy=E_axis
)

# whole sphere
mcdc.Tally(cell=sphere_cell, scores=["fission"],energy=E_axis)

# Settings
mcdc.settings.N_batch = 1
mcdc.settings.N_particle = 10e7

mcdc.run()
