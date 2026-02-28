import matplotlib.pyplot as plt
import h5py, sys
import numpy as np

# Load result
with h5py.File(sys.argv[1], "r") as f:
    x = f["tallies/mesh_tally_0/grid/x"][:]
    z = f["tallies/mesh_tally_0/grid/z"][:]
    dx = [x[1:] - x[:-1]][-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dz = [z[1:] - z[:-1]][-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    phi = f["tallies/mesh_tally_0/fission/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/fission/sdev"][:]


# Plot result
X, Y = np.meshgrid(z_mid, x_mid)
Z = phi
plt.pcolormesh(X, Y, Z)
plt.gca().set_aspect("equal")
plt.show()

X, Y = np.meshgrid(z_mid, x_mid)
Z = phi_sd
plt.pcolormesh(X, Y, Z)
plt.gca().set_aspect("equal")
plt.show()
