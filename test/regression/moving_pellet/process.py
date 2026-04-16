import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import matplotlib.animation as animation

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("output.h5", "r") as f:
    x = f["tallies/mesh_tally_0/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    z = f["tallies/mesh_tally_0/grid/z"][:]
    z_mid = 0.5 * (z[:-1] + z[1:])
    t = f["tallies/mesh_tally_0/grid/t"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(z, x)

    phi = f["tallies/mesh_tally_0/fission/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/fission/sdev"][:]

print(np.max(phi))

fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, phi[0])
text = ax.text(0.02, 1.02, "", transform=ax.transAxes)
ax.set_aspect("equal", "box")
ax.set_xlabel("$y$ [cm]")
ax.set_ylabel("$x$ [cm]")


def animate(i):
    cax.set_array(phi[i])
    cax.set_clim(phi[i].min(), phi[i].max())
    text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[i], t[i + 1]))


anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
plt.show()
