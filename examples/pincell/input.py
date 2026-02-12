import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
fuel = mcdc.Material(
    nuclide_composition={
        "U235": 0.0005581658948833916,
        "U238": 0.022404594715383263,
        "O16": 0.045831301393656466,
    }
)

water = mcdc.Material(
    nuclide_composition={
        "B10": 0.0001357003217727274,
        "H1": 0.0684556951587359,
        "O16": 0.032785655643293984,
    }
)

# Set surfaces
cy = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.45720)
pitch = 1.25984
x1 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
x2 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
y2 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")

# Set cells
mcdc.Cell(region=-cy & +x1 & -x2 & +y1 & -y2, fill=fuel)
mcdc.Cell(region=+cy & +x1 & -x2 & +y1 & -y2, fill=water)

# =============================================================================
# Set source
# =============================================================================

mcdc.Source(
    x=[-pitch / 2, pitch / 2],
    y=[-pitch / 2, pitch / 2],
    isotropic=True,
    energy=1e6,
)

# =============================================================================
# Set tallies, settings, and run MC/DC
# =============================================================================

# Tallies
mcdc.TallyGlobal(
    scores=["flux"],
    energy=np.loadtxt("energy_grid.txt"),
)

# Settings
mcdc.settings.N_particle = 100
mcdc.settings.set_eigenmode(N_inactive=10, N_active=50)

# Run
mcdc.run()
