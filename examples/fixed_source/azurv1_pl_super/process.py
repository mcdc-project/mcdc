import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# Reference solution
data = np.load("reference.npz")
phi_ref = data["phi"]

# Get results
with h5py.File("output.h5", "r") as f:
    tallies = f["tallies/mesh_tally_0"]
    flux = tallies["flux"]
    grid = tallies["grid"]
    x = grid["x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    t = grid["t"][:]
    dt = t[1:] - t[:-1]
    K = len(t) - 1

    phi = flux["mean"][:]
    phi_sd = flux["sdev"][:]

    # Normalize
    for k in range(K):
        phi[k] /= dx * dt[k]
        phi_sd[k] /= dx * dt[k]

# Flux - average
fig = plt.figure()
ax = plt.axes(
    xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992)
)
ax.grid()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Flux")
ax.set_title(r"$\bar{\phi}_{k,j}$")
(line1,) = ax.plot([], [], "-b", label="MC")
(line2,) = ax.plot([], [], "--r", label="Ref.")
fb = ax.fill_between([], [], [], [], alpha=0.2, color="b")
text = ax.text(0.02, 0.9, "", transform=ax.transAxes)
ax.legend()


def animate(k):
    global fb
    fb.remove()
    line1.set_data(x_mid, phi[k, :])
    fb = ax.fill_between(
        x_mid, phi[k, :] - phi_sd[k, :], phi[k, :] + phi_sd[k, :], alpha=0.2, color="b"
    )
    line2.set_data(x_mid, phi_ref[k, :])
    text.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[k], t[k + 1]))
    return line1, line2, text


simulation = animation.FuncAnimation(fig, animate, frames=K)
simulation.save(
    "azurv1.gif",
    fps=4,
    writer="imagemagick",
    savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0},
)
plt.show()
