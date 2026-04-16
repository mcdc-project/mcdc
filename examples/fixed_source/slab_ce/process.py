import matplotlib.pyplot as plt
import h5py, sys
import numpy as np

# Load results
with h5py.File("output.h5", "r") as f:
    # Grid
    x = f["tallies/mesh_tally_0/grid/x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    # Fluks
    phi_fast = f["tallies/mesh_tally_0/flux/mean"][1, :] / dx
    phi_fast_sd = f["tallies/mesh_tally_0/flux/sdev"][1, :] / dx
    phi_thermal = f["tallies/mesh_tally_0/flux/mean"][0, :] / dx
    phi_thermal_sd = f["tallies/mesh_tally_0/flux/sdev"][0, :] / dx

# Plot results
fig, ax1 = plt.subplots(figsize=(5, 3))
ax2 = ax1.twinx()

y = phi_fast
y_sd = phi_fast_sd
p1 = ax1.step(x_mid, y, "-b", where="mid", label="Fast flux")
ax1.fill_between(x_mid, y - y_sd, y + y_sd, alpha=0.2, color="b", step="mid")

y = phi_thermal
y_sd = phi_thermal_sd
p2 = ax2.step(x_mid, y, "--r", where="mid", label="Thermal flux")
ax2.fill_between(x_mid, y - y_sd, y + y_sd, alpha=0.2, color="r", step="mid")

ax1.set_xlabel(r"$x$ [cm]")
ax1.set_ylabel("Fast flux [/cm-s]", color="b")
ax1.tick_params(axis="y", colors="b")
ax2.set_ylabel("Thermal flux [/cm-s]", color="r")
ax2.tick_params(axis="y", colors="r")

"""
lim1 = ax1.get_ylim()
lim2 = ax2.get_ylim()
max_range = max(lim1[1]-lim1[0], lim2[1]-lim2[0])
ax1.set_ylim(top=lim1[0]+max_range)
ax2.set_ylim(top=lim2[0]+max_range)
"""

ax1.legend(handles=p1 + p2, loc="lower left")
plt.show()
