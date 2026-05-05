import matplotlib.pyplot as plt
import h5py, sys
import numpy as np

# Load result
with h5py.File("output.h5", "r") as f:
    x = f["tallies/tracklength_tally_0/grid/x"][:]
    z = f["tallies/tracklength_tally_0/grid/z"][:]
    dx = [x[1:] - x[:-1]][-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dz = [z[1:] - z[:-1]][-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    phi = f["tallies/tracklength_tally_0/fission/mean"][:]
    phi_sd = f["tallies/tracklength_tally_0/fission/sdev"][:]


# Plot result
fig, ax = plt.subplots(2, 1, figsize=(4, 9))
X, Y = np.meshgrid(z_mid, x_mid)
Z = phi
flux_plot = ax[0].pcolormesh(Y, X, Z)
ax[0].set_aspect("equal")
ax[0].set_ylabel(r"$x$ [cm]")
ax[0].set_xlabel(r"$z$ [cm]")
fig.colorbar(flux_plot, ax=ax[0], orientation="horizontal")
ax[0].set_title("Flux")
#
Z = phi_sd / phi
sdev_plot = ax[1].pcolormesh(Y, X, Z)
ax[1].set_aspect("equal")
ax[1].set_ylabel(r"$x$ [cm]")
ax[1].set_xlabel(r"$z$ [cm]")
fig.colorbar(sdev_plot, ax=ax[1], orientation="horizontal")
ax[1].set_title("Standard Deviation [%]")

plt.show()
