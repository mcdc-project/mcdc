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
    tallies = f["tallies/mesh_tally_0"]
    flux = tallies["flux"]
    grid = tallies["grid"]
    x = grid["x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = grid["y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    t = grid["t"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(y, x)

    phi = flux["mean"][:]
    phi_sd = flux["sdev"][:]

    print(len(x))

    plt.imshow(phi[:, :, 0])
    plt.show()
