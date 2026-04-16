import matplotlib.pyplot as plt
import h5py
import numpy as np

from reference import reference

# Load results
with h5py.File("output.h5", "r") as f:
    z = f["tallies/mesh_tally_0/grid/z"][:]
    dz = z[1:] - z[:-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    mu = f["tallies/mesh_tally_0/grid/mu"][:]
    dmu = mu[1:] - mu[:-1]
    mu_mid = 0.5 * (mu[:-1] + mu[1:])

    psi = f["tallies/mesh_tally_0/flux/mean"][:]
    psi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]
    psi = np.transpose(psi)
    psi_sd = np.transpose(psi_sd)

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
for n in range(N):
    psi[:, n] = psi[:, n] / dz / dmu[n]
    psi_sd[:, n] = psi_sd[:, n] / dz / dmu[n]

# Reference solution
phi_ref, _, psi_ref = reference(z, mu)

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
