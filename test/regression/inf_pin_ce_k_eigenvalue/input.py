import numpy as np
import os

import mcdc


# Set the XS library directory
os.environ["MCDC_XSLIB"] = os.path.dirname(os.getcwd())

# =============================================================================
# Set model
# =============================================================================

# Set materials

fuel = mcdc.Material(
    nuclide_composition={
        "dummy_nuclide_4": 0.0005581658948833916,
        "dummy_nuclide_5": 0.022404594715383263,
        "dummy_nuclide_3": 0.045831301393656466,
    }
)

water = mcdc.Material(
    nuclide_composition={
        "dummy_nuclide_2": 0.0001357003217727274,
        "dummy_nuclide_1": 0.0684556951587359,
        "dummy_nuclide_3": 0.032785655643293984,
    }
)

# Set surfaces
cy = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=0.45720)
pitch = 1.25984
x1 = mcdc.surface("plane-x", x=-pitch / 2, bc="reflective")
x2 = mcdc.surface("plane-x", x=pitch / 2, bc="reflective")
y1 = mcdc.surface("plane-y", y=-pitch / 2, bc="reflective")
y2 = mcdc.surface("plane-y", y=pitch / 2, bc="reflective")

# Set cells
mcdc.cell(-cy & +x1 & -x2 & +y1 & -y2, fuel)
mcdc.cell(+cy & +x1 & -x2 & +y1 & -y2, water)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(
    x=[-pitch / 2, pitch / 2],
    y=[-pitch / 2, pitch / 2],
    energy=np.array([[1e6 - 1, 1e6 + 1], [1.0, 1.0]]),
    isotropic=True,
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=10, census_bank_buffer_ratio=1000.0)
settings.set_eigenmode(N_inactive=1, N_active=2)
mcdc.population_control()

mcdc.tally.mesh_tally(
    scores=["flux", "density"],
    E=np.loadtxt("energy_grid.txt"),
    t=np.insert(np.logspace(-8, 2, 50), 0, 0.0),
)

mcdc.run()
