import numpy as np
import h5py
import mcdc

# =========================================================================
# Set model
# =========================================================================
# Based on Kornreich, ANE 2004, 31, 1477-1494,
# DOI: 10.1016/j.anucene.2004.03.012

# Set materials
m1 = mcdc.material(
    capture=np.array([0.0]),
    scatter=np.array([[0.9]]),
    fission=np.array([0.1]),
    nu_p=np.array([6.0]),
)
m2 = mcdc.material(
    capture=np.array([0.68]),
    scatter=np.array([[0.2]]),
    fission=np.array([0.12]),
    nu_p=np.array([2.5]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=1.5)
s3 = mcdc.surface("plane-x", x=2.5, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, m1)
mcdc.cell(+s2 & -s3, m2)

# =========================================================================
# iQMC Parameters
# =========================================================================
N = 50
maxit = 5
tol = 1e-3
x = np.arange(0.0, 2.6, 0.1)
Nx = len(x) - 1
fixed_source = np.zeros(Nx)
phi0 = np.ones((Nx))

# =========================================================================
# Set tally, setting, and run mcdc
# =========================================================================

mcdc.iQMC(
    x=x,
    fixed_source=fixed_source,
    phi0=phi0,
    maxit=maxit,
    tol=tol,
    scores=["source-x"],
)
# Setting
mcdc.setting(N_particle=N)
mcdc.eigenmode()

# Run
mcdc.run()
