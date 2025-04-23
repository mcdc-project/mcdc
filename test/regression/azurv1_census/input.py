import numpy as np
from pprint import pprint
import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Set materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Set cells
mcdc.cell(+s1 & -s2, m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic pulse at x=t=0

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True, time=[1e-10, 1e-10])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)

# Setting
mcdc.setting(N_particle=50, census_bank_buff=5, source_bank_buff=5, N_batch=2)
mcdc.time_census(np.linspace(0.0, 20.0, 21)[1:-1])
mcdc.population_control()

# Run
mcdc.run()
