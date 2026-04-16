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
    cs_recon = f["tallies/cs_tally_0/flux/reconstruction"][:]
    plt.imshow(cs_recon)
    plt.show()

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

    for i in range(len(f["input_deck"]["cell_tallies"])):
        flux_score = f[f"tallies/cell_tally_{i}/flux"]
        print(
            f'cell {i+1} mean = {flux_score["mean"][()]}, sdev = {flux_score["sdev"][()]}'
        )
