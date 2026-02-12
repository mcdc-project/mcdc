import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import matplotlib.animation as animation

# Load results
E = np.loadtxt("energy_grid.txt")
G = len(E) - 1
E_mid = 0.5 * (E[1:] + E[:-1])
dE = E[1:] - E[:-1]

with h5py.File("output.h5", "r") as f:
    t = f["tallies/mesh_tally_0/grid/t"][:]
    dt = t[1:] - t[:-1]
    t_mid = 0.5 * (t[1:] + t[:-1])
    K = len(t) - 1

    phi = f["tallies/mesh_tally_0/flux/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]

    density = f["tallies/mesh_tally_0/density/mean"][:]
    density_sd = f["tallies/mesh_tally_0/density/sdev"][:]

# Neutron density
n = np.zeros(K)
n_sd = np.zeros(K)
for k in range(K):
    n[k] = np.sum(density[:, k]) / dt[k]
    n_sd[k] = np.linalg.norm(density_sd[:, k]) / dt[k]

# Normalize
for k in range(K):
    phi[:, k] *= E_mid / dE / dt[k]
    phi_sd[:, k] *= E_mid / dE / dt[k]

# Flux - t
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout(pad=3.0)

ax1.plot(t_mid, n, "-b", label="MC")
no = np.zeros_like(n)
(line,) = ax1.plot(t[1:], no, "ko", fillstyle="none")
ax1.fill_between(t[1:], n - n_sd, n + n_sd, alpha=0.2, color="b")
ax1.set_xlabel(r"$t$, s")
ax1.set_ylabel("Density")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_title(r"$n_g(t)$")

ax2.grid()
ax2.set_xlabel(r"$E$, MeV")
ax2.set_ylabel(r"$E\phi(E)$")
ax2.set_title(r"$\phi(E,t)$")
ax2.set_xscale("log")
(line1,) = ax2.plot([], [], "-b", label="MC")
fb = ax2.fill_between([], [], [], [], alpha=0.2, color="b")
ax2.legend()


def animate(k):
    global fb
    fb.remove()
    no[k - 1] = 0.0
    no[k] = n[k]
    line.set_data(t[1:], no)
    line1.set_data(E_mid, phi[:, k])
    fb = ax2.fill_between(
        E_mid,
        phi[:, k] - phi_sd[:, k],
        phi[:, k] + phi_sd[:, k],
        alpha=0.2,
        color="b",
    )
    ax2.legend()
    ax2.set_ylim([(phi[:, k]).min(), (phi[:, k]).max()])
    return line1


simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
