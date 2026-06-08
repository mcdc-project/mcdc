import numpy as np
import mcdc

# ======================================================================================
# Set model
# ======================================================================================
# Proton beam, incident on a slab

# Set materials (atom density in units of atoms/barn-cm)
silicon = mcdc.Material("silicon", {"C12": 0.05})

# Set surfaces
sx1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
sx2 = mcdc.Surface.PlaneX(x=0.002, boundary_condition="vacuum")
sy1 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="vacuum")
sy2 = mcdc.Surface.PlaneY(y=1.0, boundary_condition="vacuum")
sz1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
sz2 = mcdc.Surface.PlaneZ(z=1.0, boundary_condition="vacuum")

slab = +sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2

# Set cells
slab_cell = mcdc.Cell(name="silicon", region=slab, fill=silicon)

# ======================================================================================
# Set source
# ======================================================================================

mcdc.Source(
    x=[0.0, 0.0],
    y=[0.0, 1.0],
    z=[0.0, 1.0],
    direction=[1.0, 0.0, 0.0],
    energy=1e6,
    # energy_group=0,
    particle_type="proton",
    # time=[0.0, 0.0],
)

# ======================================================================================
# Set tallies, settings, techniques, and run MC/DC
# ======================================================================================

# Tallies
percent_of_range = np.array(
    [
        0.0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        85,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        107.5,
        110,
        115,
    ]
)
range = 16.45 * 1e-4  # cm

bin_edges = range * percent_of_range * 1e-2

mesh = mcdc.MeshStructured(x=(bin_edges))
mcdc.Tally(name=f"edep", mesh=mesh, scores=["energy_deposition"])

# Settings
mcdc.settings.set_transported_particles(["proton"])
mcdc.settings.N_particle = 1_000
mcdc.settings.N_batch = 1
mcdc.settings.csda = True
mcdc.settings.csda_max_fractional_e_loss = 0.01

# Techniques
mcdc.simulation.implicit_capture()

# Run
mcdc.run()
