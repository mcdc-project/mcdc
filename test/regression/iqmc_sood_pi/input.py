import numpy as np
import h5py
import mcdc

# =========================================================================
# Set model
# =========================================================================
# Based on Sood, PNE, Volume 42, Issue 1, 2003, Pages 55-106 2003,
# "Analytical Benchmark Test Set For Criticality Code Verification"

# 2G-U Slab data
m1 = mcdc.material(
    capture=np.array([0.01344, 0.00384]),
    scatter=np.array([[0.26304, 0.0720], [0.00000, 0.078240]]),
    fission=np.array([0.06912, 0.06192]),
    nu_p=np.array([2.5, 2.7]),
    chi_p=np.array([[0.425, 0.425], [0.575, 0.575]]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=6.01275, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, m1)

# =========================================================================
# iQMC Parameters
# =========================================================================
Nx = 5
N = 10
maxit = 5
tol = 1e-3
x = np.linspace(0.0, 6.01275, num=Nx + 1)
fixed_source = np.zeros(Nx)
phi0 = np.ones((Nx))

# =========================================================================
# Set tally, setting, and run mcdc
# =========================================================================

mcdc.iQMC(
    x=x,
    phi0=phi0,
    fixed_source=fixed_source,
    maxit=maxit,
    tol=tol,
)
# Setting
mcdc.setting(N_particle=N)
mcdc.eigenmode()

# Run
mcdc.run()
