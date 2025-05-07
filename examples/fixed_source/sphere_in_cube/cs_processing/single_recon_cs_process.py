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

Nx = 40
Ny = 40
lambda_ = 1e-4
with h5py.File("output.h5", "r") as f:
    S, N_dims, bin_size_pixels = construct_cs_sampling_matrix_S(f, Nx, Ny)
    N_cs_bins = f["tallies"]["cs_tally_0"]["N_cs_bins"][()]
    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]
    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    mesh_sdev = f["tallies"]["mesh_tally_0"]["fission"]["sdev"][:]

recon, recon_time = cs_reconstruct(S, cs_results, lambda_, Nx, Ny)
