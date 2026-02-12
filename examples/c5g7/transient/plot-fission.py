import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import shutil

# Get results
with h5py.File("output.h5", "r") as f:
    fissions = f["tallies/mesh_tally_0/fission/mean"][()]
    x = f["tallies/mesh_tally_0/grid/x"][()]
    y = f["tallies/mesh_tally_0/grid/y"][()]
    z = f["tallies/mesh_tally_0/grid/z"][()]
    t = f["tallies/mesh_tally_0/grid/time"][()]

# Total fission
fission_total = np.average(fissions, axis=(1, 2, 3))
fission_total /= fission_total[0]

# The grids
t_mid = 0.5 * (t[:-1] + t[1:])
XY_X, XY_Y = np.meshgrid(x, y, indexing="ij")
XZ_X, XZ_Z = np.meshgrid(x, z, indexing="ij")
YZ_Y, YZ_Z = np.meshgrid(y, z, indexing="ij")

# Create clean folder for output figures
# Check if the folder exists
if os.path.exists("fission"):
    shutil.rmtree("fission")  # Remove the existing folder
os.makedirs("fission")  # Create a new folder

# Iterate over time step and create figures
N = len(fissions)
for i in range(N):
    fission = fissions[i]

    # Calculate fission averages
    fission_x = np.average(fission, axis=0)
    fission_y = np.average(fission, axis=1)
    fission_z = np.average(fission, axis=2)

    # Plot
    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(
        2, 3, width_ratios=[0.7, 1, 1], height_ratios=[1, 1], hspace=0.5
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left
    ax3 = fig.add_subplot(gs[:, 1])  # Entire second column
    ax4 = fig.add_subplot(gs[:, 2])  # Entire third column

    # Total fission curve
    ax1.plot(t_mid, fission_total, "b")
    ax1.set_yscale("log")
    ax1.set_ylabel("Total fission rate")
    ax1.set_xlabel("Time")
    ax1.set_title("Total fission rate")
    # Total fission point
    ax1.plot(t_mid[i], fission_total[i], "ro", fillstyle="none")

    # XY fission
    ax2.pcolormesh(XY_X, XY_Y, fission_z)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title("Fission-XY")

    # XZ fission
    ax3.pcolormesh(XZ_X, XZ_Z, fission_y)
    ax3.set_aspect("equal")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$z$")
    ax3.set_title("Fission-XZ")
    pos = ax3.get_position()
    ax3.set_position(
        [pos.x0 + 0.02, pos.y0, pos.width, pos.height]
    )  # shift right by 0.02

    # YZ fission
    ax4.pcolormesh(YZ_Y, YZ_Z, fission_x)
    ax4.set_aspect("equal")
    ax4.set_xlabel(r"$y$")
    ax4.set_ylabel(r"$z$")
    ax4.set_title("Fission-YZ")

    plt.suptitle("MC/DC result - Fission Rate")
    plt.savefig(f"fission/figure_{i:03}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
