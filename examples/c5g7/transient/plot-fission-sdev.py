import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import shutil

# Get fission rates
with h5py.File("output.h5", "r") as f:
    fissions = f["tallies/mesh_tally_0/fission/mean"][()]
    fissions_sd = f["tallies/mesh_tally_0/fission/sdev"][()]

    x = f["tallies/mesh_tally_0/grid/x"][()]
    y = f["tallies/mesh_tally_0/grid/y"][()]
    z = f["tallies/mesh_tally_0/grid/z"][()]
    t = f["tallies/mesh_tally_0/grid/time"][()]

# The grids
t_mid = 0.5 * (t[:-1] + t[1:])
XY_X, XY_Y = np.meshgrid(x, y, indexing="ij")
XZ_X, XZ_Z = np.meshgrid(x, z, indexing="ij")
YZ_Y, YZ_Z = np.meshgrid(y, z, indexing="ij")

# Relative stdevs
fissions_sd[fissions == 0.0] = 0.0
non_zeros = fissions != 0.0
fissions_sd[non_zeros] /= fissions[non_zeros]

# Average relative stdev (in %)
fission_sd_avg = np.average(fissions_sd, axis=(1, 2, 3)) * 100.0

# Create clean folder for output figures
# Check if the folder exists
if os.path.exists("fission-sdev"):
    shutil.rmtree("fission-sdev")  # Remove the existing folder
os.makedirs("fission-sdev")  # Create a new folder

# Iterate over time step and create figures
N = len(fissions)
for i in range(N):
    fission_sd = fissions_sd[i]

    # Calculate fission averages
    fission_x_sd = np.average(fission_sd, axis=0)
    fission_y_sd = np.average(fission_sd, axis=1)
    fission_z_sd = np.average(fission_sd, axis=2)

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
    ax1.plot(t_mid, fission_sd_avg, "b")
    ax1.set_yscale("log")
    ax1.set_ylabel("Average relative sdev (%)")
    ax1.set_xlabel("Time")
    ax1.set_title("Average relative sdev")
    # Total fission point
    ax1.plot(t_mid[i], fission_sd_avg[i], "ro", fillstyle="none")

    # XY fission
    ax2.pcolormesh(XY_X, XY_Y, fission_z_sd)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title("Fission-XY")

    # XZ fission
    ax3.pcolormesh(XZ_X, XZ_Z, fission_y_sd)
    ax3.set_aspect("equal")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$z$")
    ax3.set_title("Fission-XZ")
    pos = ax3.get_position()
    ax3.set_position(
        [pos.x0 + 0.02, pos.y0, pos.width, pos.height]
    )  # shift right by 0.02

    # YZ fission
    ax4.pcolormesh(YZ_Y, YZ_Z, fission_x_sd)
    ax4.set_aspect("equal")
    ax4.set_xlabel(r"$y$")
    ax4.set_ylabel(r"$z$")
    ax4.set_title("Fission-YZ")

    plt.suptitle("MC/DC result - Fission Rate Relative Sdev.")
    plt.savefig(
        f"fission-sdev/figure_{i:03}.png", dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.close()
