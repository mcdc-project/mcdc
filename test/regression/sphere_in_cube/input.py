import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================
# Homogeneous pure-fission sphere inside a pure-scattering cube

# Set materials
pure_f = mcdc.MaterialMG(fission=np.array([1.0]), nu_p=np.array([1.2]))
pure_s = mcdc.MaterialMG(scatter=np.array([[1.0]]))

# Set surfaces
sx1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
sx2 = mcdc.surface("plane-x", x=4.0, bc="vacuum")
sy1 = mcdc.surface("plane-y", y=0.0, bc="vacuum")
sy2 = mcdc.surface("plane-y", y=4.0, bc="vacuum")
sz1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
sz2 = mcdc.surface("plane-z", z=4.0, bc="vacuum")
sphere = mcdc.surface("sphere", center=[2.0, 2.0, 2.0], radius=1.5)
inside_sphere = -sphere
inside_box = +sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2

# Set cells
# Source
mcdc.cell(inside_box & ~inside_sphere, pure_s)

# Sphere
sphere_cell = mcdc.cell(inside_sphere, pure_f)

# =============================================================================
# Set source
# =============================================================================
# The source pulses in t=[0,5]

mcdc.source(x=[0.0, 4.0], y=[0.0, 4.0], z=[0.0, 4.0], time=[0.0, 50.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=100, N_batch=2)
mcdc.implicit_capture()

mcdc.tally.mesh_tally(
    scores=["fission"],
    x=np.linspace(0.0, 4.0, 2),
    y=np.linspace(0.0, 4.0, 2),
    z=np.linspace(0.0, 4.0, 2),
    # t=np.linspace(0.0, 200.0, 2),
)

mcdc.tally.cell_tally(sphere_cell, scores=["fission"])

# Run
mcdc.run()
