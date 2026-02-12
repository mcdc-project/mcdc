import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation

# Load result
with h5py.File("output.h5", "r") as f:
    x = f["tallies/mesh_tally_0/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = f["tallies/mesh_tally_0/grid/y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    t = f["tallies/mesh_tally_0/grid/time"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(y, x)

    phi = f["tallies/mesh_tally_0/flux/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]

    phi_total = f["tallies/global_tally_0/density/mean"][:]
    phi_total_sd = f["tallies/global_tally_0/density/sdev"][:]

# Animate result
fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.0, 2]})
#
cax = ax[1].pcolormesh(X, Y, phi[0], vmin=phi[0].min(), vmax=phi[0].max())
ax[1].set_aspect("equal", "box")
ax[1].set_xlabel("$y$ [cm]")
ax[1].set_ylabel("$x$ [cm]")
#
ax[0].plot(t_mid, phi_total)
ax[0].set_xlabel("$t$ [s]")
ax[0].set_ylabel("Neutron density")
ax[0].set_yscale("log")
ax[0].plot(t_mid, phi_total, "b-")
ax[0].fill_between(
    t_mid, phi_total - phi_total_sd, phi_total + phi_total_sd, alpha=0.2, color="b"
)
ax[0].grid()
ax[0].set_box_aspect(1)
(line,) = ax[0].plot([], [], "ok", fillstyle="none")


#
def animate(i):
    n = np.zeros_like(t_mid)
    n[i] = phi_total[i]
    line.set_data(t_mid, n)
    cax.set_array(phi[i])
    cax.set_clim(phi[i].min(), phi[i].max())


#
K = len(t) - 1
anim = animation.FuncAnimation(fig, animate, frames=K)
plt.show()
