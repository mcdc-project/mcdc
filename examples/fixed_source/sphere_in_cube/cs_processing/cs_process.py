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


def rel_l2_error(recon, real):
    recon = recon.flatten()
    real = real.flatten()

    return np.linalg.norm(real - recon, ord=2) / np.linalg.norm(real, ord=2)


Nx = 20
Ny = 20
Nz = 20
with h5py.File("../output.h5", "r") as f:
    S, N_dims, bin_size_pixels = construct_cs_sampling_matrix_S(f, Nx, Ny, Nz)
    N_cs_bins = f["tallies"]["cs_tally_0"]["N_cs_bins"][()]
    cs_results = f["tallies"]["cs_tally_0"]["fission"]["mean"][:]
    mesh_results = f["tallies"]["mesh_tally_0"]["fission"]["mean"][:]
    mesh_sdev = f["tallies"]["mesh_tally_0"]["fission"]["sdev"][:]

lambda_array = [
    "mesh",
    0,
    5e-7,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
    2e-1,
    3e-1,
]
recon_array = [
    mesh_results if lam == "mesh" else cs_reconstruct(S, cs_results, lam, Nx, Ny, Nz)
    for lam in lambda_array
]
rel_errors = [rel_l2_error(recon, mesh_results) for recon in recon_array]


# Plotting the reconstructions for different lambda values
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    l = lambda_array[i]
    reconstruction = recon_array[i]
    if N_dims == 2:
        im = ax.imshow(reconstruction, origin="lower", extent=[0, 4, 0, 4])
    elif N_dims == 3:
        im = ax.imshow(
            reconstruction[:, :, Nz // 2], origin="lower", extent=[0, 4, 0, 4]
        )

    ax.set_title(f"MC/DC Solution" if l == "mesh" else f"$\lambda$ = {l:.5g}")
    ax.set_ylabel("y [cm]" if i % 4 == 0 else "")
    ax.set_xlabel("x [cm]" if i >= 12 else "")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=1)
    cbar.formatter.set_powerlimits((0, 0))

plt.suptitle(
    "Basis Pursuit Denoising Reconstructions with Different Values of $\lambda$",
    fontsize=16,
)
plt.tight_layout()
if N_dims == 2:
    plt.savefig(
        f"{N_dims}D_sphere_recons_{N_cs_bins}_{bin_size_pixels}{bin_size_pixels}.png"
    )
elif N_dims == 3:
    plt.savefig(
        f"{N_dims}D_sphere_recons_{N_cs_bins}_{bin_size_pixels}{bin_size_pixels}{bin_size_pixels}.png"
    )
plt.show()


# Plotting the relative errors
plt.plot(lambda_array[1:], rel_errors[1:], label="Reconstruction Errors")
plt.hlines(
    np.linalg.norm(mesh_sdev.flatten(), ord=2)
    / np.linalg.norm(mesh_results.flatten(), ord=2),
    plt.xlim()[0],
    plt.xlim()[1],
    color="black",
    linestyle="--",
    label="Reference Std Dev",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("Relative L$^2$ Error")
plt.title(f"Relative Error vs $\lambda$ - {N_dims}D Sphere Reconstructions")
plt.legend()
plt.tight_layout()
if N_dims == 2:
    plt.savefig(
        f"{N_dims}D_sphere_errors_{N_cs_bins}_{bin_size_pixels}{bin_size_pixels}.png"
    )
elif N_dims == 3:
    plt.savefig(
        f"{N_dims}D_sphere_errors_{N_cs_bins}_{bin_size_pixels}{bin_size_pixels}{bin_size_pixels}.png"
    )

plt.show()
