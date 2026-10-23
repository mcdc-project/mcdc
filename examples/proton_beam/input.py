import numpy as np
import mcdc

simulation = mcdc.Simulation("proton_beam_test")

# ======================================================================================
# Set model
# ======================================================================================

# Set materials
material_density = 2.33  # g/cm^3
molar_mass = 28.085
atom_density = material_density * 1e-24 * 1 / molar_mass * 6.022e23
silicon = mcdc.Material("Si", {"Si28": atom_density}, temperature=0.0)

# Set surfaces
sx1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
sx2 = mcdc.Surface.PlaneX(x=4.5, boundary_condition="vacuum")
sy1 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="vacuum")
sy2 = mcdc.Surface.PlaneY(y=1.0, boundary_condition="vacuum")
sz1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
sz2 = mcdc.Surface.PlaneZ(z=1.0, boundary_condition="vacuum")

# Set cells
si_cell = mcdc.Cell(region=+sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2, fill=silicon)
simulation.set_model([si_cell])

# ======================================================================================
# Set source
# ======================================================================================
E_mean = 1.0e8
E_sigma = 0.01 * E_mean

# Discretize the Gaussian over +/- 5 sigma
E_values = np.linspace(E_mean - 5 * E_sigma, E_mean + 5 * E_sigma, 200)
pdf_weights = np.exp(-0.5 * ((E_values - E_mean) / E_sigma) ** 2)
pdf_weights /= np.trapezoid(pdf_weights, E_values)

source = mcdc.Source(
    x=[0.0, 0.0],
    y=[0.5, 0.5],
    z=[0.5, 0.5],
    direction=[1.0, 0.0, 0.0],
    energy=[E_values, pdf_weights],
    particle_type="proton",
)

simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, techniques, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshUniform(x=(0.0, 0.045, 100))
tally = mcdc.Tally(mesh=mesh, name="proton_edep", scores=["energy_deposition"])
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 10000
simulation.settings.N_batch = 2
simulation.settings.condensed_interactions(max_fractional_energy_loss=0.01)

# Techniques
# simulation.technique.implicit_capture()

# Run
simulation.run()
