import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py

# Load result and grid
with h5py.File("output.h5", "r") as f:
    x = f["tallies/mesh_tally_0/grid/x"][:]
    y = f["tallies/mesh_tally_0/grid/y"][:]
    z = f["tallies/mesh_tally_0/grid/z"][:]

    phi = f["tallies/mesh_tally_0/flux/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]

# The 2D grid for Z-scan plots
X, Y = np.meshgrid(y, x)

# Normalization over all Z slices
norm_mean = colors.Normalize(vmin=phi.min(), vmax=phi.max())
norm_sdev = colors.Normalize(vmin=phi_sd.min(), vmax=phi_sd.max())

# Z-scan loop
Nz = len(z) - 1
for i in range(Nz):
    # Get the mean and sdev values for current Z-slice
    #   The indexing is [x,y,z]
    mean = phi[:, :, i]
    sdev = phi_sd[:, :, i]

    # Plot mean and sdev side-by-side
    fig, ax = plt.subplots(1, 2)
    plot_mean = ax[0].pcolormesh(X, Y, mean, norm=norm_mean)
    plot_sdev = ax[1].pcolormesh(X, Y, sdev, norm=norm_sdev)

    # Colorbar
    fig.colorbar(plot_mean, ax=ax[0])
    fig.colorbar(plot_sdev, ax=ax[1])

    # Formats
    fig.suptitle(f"Flux within Z = [{z[i]}, {z[i+1]}]")
    ax[0].set_title("Mean")
    ax[0].set_xlabel("$y$ [cm]")
    ax[0].set_ylabel("$x$ [cm]")
    ax[1].set_title("Std. Dev.")
    ax[1].set_xlabel("$y$ [cm]")
    ax[1].set_ylabel("$x$ [cm]")

    plt.show()
