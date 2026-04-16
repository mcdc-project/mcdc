import numpy as np
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# Plot results
# =============================================================================

# Load iqmc result
with h5py.File("output.h5", "r") as f:
    x = f["iqmc/grid/x"][:]
    y = f["iqmc/grid/y"][:]
    phi_avg = f["iqmc/tally/flux"][:]
    f.close()


dx = x[1] - x[0]
x_mid = 0.5 * (x[1:] + x[:-1])
y_mid = 0.5 * (y[1:] + y[:-1])
Y, X = np.meshgrid(x_mid, y_mid)

norm = np.sum(phi_avg)
phi_tot = phi_avg.sum(axis=0) / norm

phi_fast = phi_avg[:5, :, :].sum(axis=0)
norm = np.sum(phi_fast)
phi_fast /= norm

phi_slow = phi_avg[5:7, :, :].sum(axis=0)
norm = np.sum(phi_slow)
phi_slow /= norm


plt.figure(dpi=300, figsize=(8, 4))
plt.pcolormesh(X, Y, phi_tot, shading="nearest")
plt.colorbar().set_label(r"Normalized Scalar Flux", rotation=270, labelpad=15)
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Total Neutron Flux")
plt.show()
plt.tight_layout()


plt.figure(dpi=300)
plt.pcolormesh(X, Y, phi_fast, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast Neutron Flux")
plt.show()


plt.figure(dpi=300)
plt.pcolormesh(X, Y, phi_slow, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal Neutron Flux")
plt.show()
