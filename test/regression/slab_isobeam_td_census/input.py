from mcdc.input_ import source
import numpy as np

import mcdc


# =============================================================================
# Set model
# =============================================================================
# Finite homogeneous pure-absorbing slab

# Set materials
m = mcdc.MaterialMG(capture=np.array([1.0]))

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=5.0, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic beam from left-end

mcdc.source(point=[1e-10, 0.0, 0.0], time=[0.0, 5.0], white_direction=[1.0, 0.0, 0.0])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=100, source_bank_buffer_ratio=5, N_batch=2)
settings.set_time_census(np.linspace(0.0, 5.0, 6)[1:])
mcdc.population_control()

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 5.0, 51),
    t=np.linspace(0.0, 5.0, 51),
)

# Run
mcdc.run()
