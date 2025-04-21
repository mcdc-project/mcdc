import matplotlib.pyplot as plt
import h5py
import numpy as np

# Load result
with h5py.File("output.h5", "r") as f:
    tally = f["tallies/mesh_tally_0"]
    x = tally["grid/x"][:]
    dx = [x[1:] - x[:-1]][-1]
    x_mid = 0.5 * (x[:-1] + x[1:])

    phi = tally["flux/mean"][:]
    phi_sd = tally["flux/sdev"][:]


# Plot result
X, Y = np.meshgrid(x_mid, x_mid)
Z = phi
Z = np.log10(np.abs(Z))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Y, X, Z, edgecolor="k", color="white")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("Log10 of scalar flux", rotation=90)
ax.view_init(elev=18, azim=38)
plt.show()

Z = phi_sd / phi
Z = np.log10(np.abs(Z))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Y, X, Z, edgecolor="k", color="white")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("Log10 of scalar flux rel. stdev.", rotation=90)
ax.view_init(elev=18, azim=38)
plt.show()
