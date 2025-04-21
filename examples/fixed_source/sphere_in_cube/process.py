import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("output.h5", "r") as f:
    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    plt.imshow(mesh_results, extent=[0, 4, 0, 4])
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title(f"Fission Results")
    plt.colorbar()
    plt.show()
