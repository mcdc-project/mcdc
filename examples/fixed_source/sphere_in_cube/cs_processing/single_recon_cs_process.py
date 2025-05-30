import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import time
from mcdc.main import cs_reconstruct, construct_cs_sampling_matrix_S

try:
    import cvxpy as cp
except ImportError:
    print("CVXPY has not been installed. Please install it with 'pip install cvxpy'")


# This script just provides a single reconstruction, based on the dimensions of
# x and y (from Nx and Ny), and the lambda_ value. A search for an optimal lambda_
# may be necessary, in which case, cs_process.py may be of use.

Nx = 40
Ny = 40
lambda_ = 1e-3

# Don't touch anything below this
with h5py.File("output.h5", "r") as f:
    # Obtaining the sampling matrix from the problem parameters
    S, N_dims, bin_size_pixels = construct_cs_sampling_matrix_S(f, Nx, Ny)
    N_cs_bins = f["tallies"]["cs_tally_0"]["N_cs_bins"][()]

    # The measurement vector
    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]

# Performing the reconstructions
recon = cs_reconstruct(S, cs_results, lambda_, Nx, Ny)

plt.imshow(recon)
plt.show()
