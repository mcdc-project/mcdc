import numpy as np

import mcdc
# Set materials
m1 = mcdc.MaterialMG(capture=np.array([1.0]))
m2 = mcdc.MaterialMG(capture=np.array([1.5]))
m3 = mcdc.MaterialMG(capture=np.array([2.0]))
# Set surfaces
s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneZ(z=2.0)
s3 = mcdc.Surface.PlaneZ(z=4.0)
s4 = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")
mcdc.Cell(region=+s1 & -s2, fill=m2)
mcdc.Cell(region=+s2 & -s3, fill=m3)
mcdc.Cell(region=+s3 & -s4, fill=m1)
mcdc.Source(z=[0.0, 6.0], isotropic=True, energy_group=0)
# Tally: cell-average fluxes and collisions
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
mcdc.Tally(
    mesh=mesh,
    scores=["flux", "collision"],
    mu=np.linspace(-1.0, 1.0, 32 + 1),
)

# Tally: current crossing a cell boundary
mcdc.Tally(
    cell=my_cell,
    scores=["current-net", "current-in", "current-out"],
)
mcdc.settings.N_particle = 1000
mcdc.run()
import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different purely-absorbing materials

# Set materials
m1 = mcdc.MaterialMG(capture=np.array([1.0]))
m2 = mcdc.MaterialMG(capture=np.array([1.5]))
m3 = mcdc.MaterialMG(capture=np.array([2.0]))

# Set surfaces
s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneZ(z=2.0)
s3 = mcdc.Surface.PlaneZ(z=4.0)
s4 = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+s1 & -s2, fill=m2)
mcdc.Cell(region=+s2 & -s3, fill=m3)
mcdc.Cell(region=+s3 & -s4, fill=m1)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.Source(z=[0.0, 6.0], isotropic=True, energy_group=0)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average fluxes and collisions
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
mcdc.Tally(
    mesh=mesh,
    scores=["flux", "collision"],
    mu=np.linspace(-1.0, 1.0, 32 + 1),
)

# Setting
mcdc.settings.N_particle = 1000

# Run
mcdc.run()
import h5py
import numpy as np
# Load results
with h5py.File("output.h5", "r") as f:
    # The tally name matches the auto-generated name (e.g., "mesh_tally_0")
    tally_name = list(f["tallies"].keys())[0]
    tally = f[f"tallies/{tally_name}"]

    z = tally["grid/z"][:]
    dz = z[1:] - z[:-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    mu = tally["grid/mu"][:]
    dmu = mu[1:] - mu[:-1]
    mu_mid = 0.5 * (mu[:-1] + mu[1:])

    psi = tally["flux/mean"][:]
    psi_sd = tally["flux/sdev"][:]
from reference import reference
import matplotlib.pyplot as plt
import numpy as np

I = len(z) - 1
N = len(mu) - 1

# Scalar flux
phi = np.zeros(I)
phi_sd = np.zeros(I)
for i in range(I):
    phi[i] += np.sum(psi[i, :])
    phi_sd[i] += np.linalg.norm(psi_sd[i, :])

# Normalize
phi /= dz
phi_sd /= dz
J /= dz
J_sd /= dz
for n in range(N):
    psi[:, n] = psi[:, n] / dz / dmu[n]
    psi_sd[:, n] = psi_sd[:, n] / dz / dmu[n]

# Reference solution
phi_ref, J_ref, psi_ref = reference(z, mu)

# Flux - spatial average
plt.plot(z_mid, phi, "-b", label="MC")
plt.fill_between(z_mid, phi - phi_sd, phi + phi_sd, alpha=0.2, color="b")
plt.plot(z_mid, phi_ref, "--r", label="Ref.")
plt.xlabel(r"$z$, cm")
plt.ylabel("Flux")
plt.ylim([0.06, 0.16])
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()

# Current - spatial average
plt.plot(z_mid, J, "-b", label="MC")
plt.fill_between(z_mid, J - J_sd, J + J_sd, alpha=0.2, color="b")
plt.plot(z_mid, J_ref, "--r", label="Ref.")
plt.xlabel(r"$z$, cm")
plt.ylabel("Current")
plt.ylim([-0.03, 0.045])
plt.grid()
plt.legend()
plt.title(r"$\bar{J}_i$")
plt.show()

# Angular flux - spatial average
vmin = min(np.min(psi_ref), np.min(psi))
vmax = max(np.max(psi_ref), np.max(psi))
fig, ax = plt.subplots(1, 2, sharey=True)
Z, MU = np.meshgrid(z_mid, mu_mid)
im = ax[0].pcolormesh(MU.T, Z.T, psi_ref, vmin=vmin, vmax=vmax)
ax[0].set_xlabel(r"Polar cosine, $\mu$")
ax[0].set_ylabel(r"$z$")
ax[0].set_title(r"\psi")
ax[0].set_title(r"$\bar{\psi}_i(\mu)$ [Ref.]")
ax[1].pcolormesh(MU.T, Z.T, psi, vmin=vmin, vmax=vmax)
ax[1].set_xlabel(r"Polar cosine, $\mu$")
ax[1].set_ylabel(r"$z$")
ax[1].set_title(r"$\bar{\psi}_i(\mu)$ [MC]")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Angular flux")
plt.show()
