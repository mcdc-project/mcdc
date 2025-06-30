import argparse, os, sys
import importlib.metadata
import mcdc.config as config
import matplotlib.pyplot as plt
import numba as nb
from matplotlib import colors as mpl_colors
import scipy.fft as spfft
from scipy.stats.qmc import Halton
import cvxpy as cp

from mcdc.card import UniverseCard
from mcdc.print_ import (
    print_banner,
    print_msg,
    print_runtime,
    print_header_eigenvalue,
    print_warning,
    print_error,
)

import h5py
import numpy as np

from mpi4py import MPI

import mcdc.kernel as kernel
import mcdc.type_ as type_

import mcdc.adapt as adapt
from mcdc.constant import *
from mcdc.loop import (
    loop_fixed_source,
    loop_eigenvalue,
    build_gpu_progs,
)
from mcdc.iqmc.iqmc_loop import iqmc_simulation, iqmc_validate_inputs
import mcdc.src.geometry as geometry

import mcdc.loop as loop
from mcdc.print_ import print_banner, print_msg, print_runtime, print_header_eigenvalue

# Get input_deck
import mcdc.global_ as mcdc_

input_deck = mcdc_.input_deck


def run():
    # Override input deck with command-line argument, if given
    if config.args.N_particle is not None:
        input_deck.setting["N_particle"] = config.args.N_particle
    if config.args.output is not None:
        input_deck.setting["output_name"] = config.args.output
    if config.args.progress_bar is not None:
        input_deck.setting["progress_bar"] = config.args.progress_bar
    if config.args.delta_tracking is not None:
        input_deck.technique["delta_tracking"] = config.args.delta_tracking
    # Start timer
    total_start = MPI.Wtime()

    # Preparation
    #   Set up and get the global variable container `mcdc` based on
    #   input deck
    preparation_start = MPI.Wtime()
    if input_deck.technique["iQMC"]:
        iqmc_validate_inputs(input_deck)

    data_tally, mcdc_arr = prepare()
    mcdc = mcdc_arr[0]
    mcdc["runtime_preparation"] = MPI.Wtime() - preparation_start

    # Print banner, hardware configuration, and header
    print_banner(mcdc, config.args)

    print_msg(" Now running TNT...")
    if mcdc["setting"]["mode_eigenvalue"]:
        print_header_eigenvalue(mcdc)

    # Run simulation
    simulation_start = MPI.Wtime()
    if mcdc["technique"]["iQMC"]:
        iqmc_simulation(mcdc_arr)
    elif mcdc["setting"]["mode_eigenvalue"]:
        loop_eigenvalue(data_tally, mcdc_arr)
    else:
        loop_fixed_source(data_tally, mcdc_arr)
    mcdc["runtime_simulation"] = MPI.Wtime() - simulation_start

    # Compressed sensing reconstruction
    N_cs_bins = mcdc["cs_tallies"]["filter"]["N_cs_bins"][0]
    if N_cs_bins != 0:
        cs_reconstruct(data_tally, mcdc)

    # Output: generate hdf5 output files
    output_start = MPI.Wtime()
    generate_hdf5(data_tally, mcdc)
    mcdc["runtime_output"] = MPI.Wtime() - output_start

    # Stop timer
    MPI.COMM_WORLD.Barrier()
    mcdc["runtime_total"] = MPI.Wtime() - total_start

    # Closout
    closeout(mcdc)


def calculate_cs_A(data, mcdc):
    x_grid = mcdc["mesh_tallies"]["filter"]["x"][0]
    y_grid = mcdc["mesh_tallies"]["filter"]["y"][0]
    Nx = len(x_grid) - 1
    Ny = len(y_grid) - 1

    N_cs_bins = mcdc["cs_tallies"]["filter"]["N_cs_bins"][0]
    cs_bin_size = mcdc["cs_tallies"]["filter"]["cs_bin_size"][0]

    S = [[] for _ in range(N_cs_bins)]

    [x_centers, y_centers] = mcdc["cs_tallies"]["filter"]["cs_centers"][0]
    x_centers[-1] = (x_grid[-1] + x_grid[0]) / 2
    y_centers[-1] = (y_grid[-1] + y_grid[0]) / 2

    # Calculate the overlap grid for each bin, and flatten into a row of S
    for ibin in range(N_cs_bins):
        if ibin == N_cs_bins - 1:
            # could just change to -INF, INF
            cs_bin_size = np.array([x_grid[-1] + x_grid[0], y_grid[-1] + y_grid[0]])

        bin_x_min = x_centers[ibin] - cs_bin_size[0] / 2
        bin_x_max = x_centers[ibin] + cs_bin_size[0] / 2
        bin_y_min = y_centers[ibin] - cs_bin_size[1] / 2
        bin_y_max = y_centers[ibin] + cs_bin_size[1] / 2

        overlap = np.zeros((len(y_grid) - 1, len(x_grid) - 1))

        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                cell_x_min = x_grid[j]
                cell_x_max = x_grid[j + 1]
                cell_y_min = y_grid[i]
                cell_y_max = y_grid[i + 1]

                # Calculate overlap in x and y directions
                overlap_x = np.maximum(
                    0,
                    np.minimum(bin_x_max, cell_x_max)
                    - np.maximum(bin_x_min, cell_x_min),
                )
                overlap_y = np.maximum(
                    0,
                    np.minimum(bin_y_max, cell_y_max)
                    - np.maximum(bin_y_min, cell_y_min),
                )

                # Calculate fractional overlap
                cell_area = (cell_x_max - cell_x_min) * (cell_y_max - cell_y_min)
                overlap[i, j] = (overlap_x * overlap_y) / cell_area

        S[ibin] = overlap.flatten()
    S = np.array(S)
    mcdc["cs_tallies"]["filter"]["cs_S"] = S

    assert np.allclose(S[-1], np.ones(Nx * Ny)), "Last row of S must be all ones"
    assert S.shape[1] == Nx * Ny, "Size of S must match Nx * Ny."
    assert (
        S.shape[1] == mcdc["cs_tallies"]["N_bin"][0]
    ), "Size of S must match number of cells in desired mesh tally"

    # TODO: can this be done in a different way? idk
    # Construct the DCT matrix T
    idct_basis_x = spfft.idct(np.identity(Nx), axis=0)
    idct_basis_y = spfft.idct(np.identity(Ny), axis=0)

    T_inv = np.kron(idct_basis_y, idct_basis_x)
    A = S @ T_inv
    return A, T_inv


def calculate_cs_sparse_solution(data, mcdc, A, b):
    N_fine_cells = mcdc["cs_tallies"]["N_bin"][0]

    # setting up the problem with CVXPY
    vx = cp.Variable(N_fine_cells)

    # Basis pursuit denoising
    l = 0.5
    objective = cp.Minimize(0.5 * cp.norm(A @ vx - b, 2) + l * cp.norm(vx, 1))
    prob = cp.Problem(objective)
    result = prob.solve(verbose=False)

    # # Basis pursuit
    # objective = cp.Minimize(cp.norm(vx, 1))
    # constraints = [A @ vx == b]
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve(verbose=True)
    # print(f'vx.value = {vx.value}')

    # formatting the sparse solution
    sparse_solution = np.array(vx.value).squeeze()

    return sparse_solution


def cs_reconstruct(data, mcdc):
    tally_bin = data[TALLY]
    tally = mcdc["cs_tallies"][0]
    stride = tally["stride"]
    bin_idx = stride["tally"]
    N_cs_bins = tally["filter"]["N_cs_bins"]
    Nx = len(mcdc["mesh_tallies"]["filter"]["x"][0]) - 1
    Ny = len(mcdc["mesh_tallies"]["filter"]["y"][0]) - 1

    b = tally_bin[TALLY_SUM, bin_idx : bin_idx + N_cs_bins]

    A, T_inv = calculate_cs_A(data, mcdc)
    x = calculate_cs_sparse_solution(data, mcdc, A, b)

    recon = T_inv @ x
    recon_reshaped = recon.reshape(Ny, Nx)

    tally["filter"]["cs_reconstruction"] = recon_reshaped


# =============================================================================
# utilities for handling discrepancies between input and program types
# =============================================================================


def copy_field(dst, src, name):
    if "padding" in name:
        return

    if isinstance(src, dict):
        data = src[name]
    else:
        data = getattr(src, name)

    if isinstance(dst[name], np.ndarray):
        if isinstance(data, np.ndarray) and dst[name].shape != data.shape:
            for dim in data.shape:
                if dim == 0:
                    return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {src[name].shape}"
            )
        elif isinstance(data, list) and dst[name].shape[0] != len(data):
            if len(src[name]) == 0:
                return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {len(src[name])}"
            )

    dst[name] = data


# =============================================================================
# prepare domain decomposition
# =============================================================================
def get_d_idx(i, j, k, ni, nj):
    N = i + j * ni + k * ni * nj
    return N


def get_indexes(N, nx, ny):
    k = int(N / (nx * ny))
    j = int((N - nx * ny * k) / nx)
    i = int(N - nx * ny * k - nx * j)
    return i, j, k


def get_neighbors(N, nx, ny, nz):
    i, j, k = get_indexes(N, nx, ny)
    if i > 0:
        xn = get_d_idx(i - 1, j, k, nx, ny)
    else:
        xn = None
    if i < (nx - 1):
        xp = get_d_idx(i + 1, j, k, nx, ny)
    else:
        xp = None
    if j > 0:
        yn = get_d_idx(i, j - 1, k, nx, ny)
    else:
        yn = None
    if j < (ny - 1):
        yp = get_d_idx(i, j + 1, k, nx, ny)
    else:
        yp = None
    if k > 0:
        zn = get_d_idx(i, j, k - 1, nx, ny)
    else:
        zn = None
    if k < (nz - 1):
        zp = get_d_idx(i, j, k + 1, nx, ny)
    else:
        zp = None
    return xn, xp, yn, yp, zn, zp


def prepare_domain_decomposition():
    # Key parameters
    work_ratio = input_deck.technique["dd_work_ratio"]
    N_proc = MPI.COMM_WORLD.Get_size()
    # Decomposition mesh sizes
    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1

    # Default parameters
    if input_deck.technique["dd_exchange_rate"] == None:
        input_deck.technique["dd_exchange_rate"] = 100
    if work_ratio is None:
        work_ratio = np.ones(d_Nx * d_Ny * d_Nz, dtype=int)
        input_deck.technique["dd_work_ratio"] = work_ratio
    if input_deck.technique["dd_exchange_rate_padding"] == None:
        if config.args.target == "gpu":
            padding = config.args.gpu_block_count * 64 * 16
        else:
            padding = 0
        input_deck.technique["dd_exchange_rate_padding"] = padding

    # Check if the combination of work_ratio and MPI rank size is acceptable
    if (
        input_deck.technique["domain_decomposition"]
        and N_proc % np.sum(work_ratio) != 0
    ):
        print_msg(
            "Number of MPI processes (%i) should be a multiple of the sum of the decomposed domain work ratio (%i)"
            % (N_proc, np.sum(work_ratio))
        )
        exit()
    N_ratio = int(N_proc / np.sum(work_ratio))
    work_ratio *= N_ratio

    # Assign domain index and processors' numbers in each domain
    if input_deck.technique["domain_decomposition"]:
        i = 0
        rank_info = []
        for n in range(d_Nx * d_Ny * d_Nz):
            ranks = []
            for r in range(work_ratio[n]):
                ranks.append(i)
                if MPI.COMM_WORLD.Get_rank() == i:
                    d_idx = n
                    local_rank = r
                i += 1
            rank_info.append(ranks)
        input_deck.technique["dd_idx"] = d_idx
        input_deck.technique["dd_local_rank"] = local_rank
        xn, xp, yn, yp, zn, zp = get_neighbors(d_idx, d_Nx, d_Ny, d_Nz)
    else:
        input_deck.technique["dd_idx"] = 0
        input_deck.technique["dd_local_rank"] = 0
        input_deck.technique["dd_xp_neigh"] = []
        input_deck.technique["dd_xn_neigh"] = []
        input_deck.technique["dd_yp_neigh"] = []
        input_deck.technique["dd_yn_neigh"] = []
        input_deck.technique["dd_zp_neigh"] = []
        input_deck.technique["dd_zn_neigh"] = []
        return

    # Assign neighbor processor numbers in all 3x2 sides
    if xp is not None:
        input_deck.technique["dd_xp_neigh"] = rank_info[xp]
    else:
        input_deck.technique["dd_xp_neigh"] = []
    if xn is not None:
        input_deck.technique["dd_xn_neigh"] = rank_info[xn]
    else:
        input_deck.technique["dd_xn_neigh"] = []

    if yp is not None:
        input_deck.technique["dd_yp_neigh"] = rank_info[yp]
    else:
        input_deck.technique["dd_yp_neigh"] = []
    if yn is not None:
        input_deck.technique["dd_yn_neigh"] = rank_info[yn]
    else:
        input_deck.technique["dd_yn_neigh"] = []

    if zp is not None:
        input_deck.technique["dd_zp_neigh"] = rank_info[zp]
    else:
        input_deck.technique["dd_zp_neigh"] = []
    if zn is not None:
        input_deck.technique["dd_zn_neigh"] = rank_info[zn]
    else:
        input_deck.technique["dd_zn_neigh"] = []


def dd_mesh_bounds(idx):
    """
    Defining mesh tally boundaries for domain decomposition.
    Used in prepare() when domain decomposition is active.
    """
    # find DD mesh index of subdomain
    d_idx = input_deck.technique["dd_idx"]  # subdomain index
    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1
    zmesh_idx = d_idx // (d_Nx * d_Ny)
    ymesh_idx = (d_idx % (d_Nx * d_Ny)) // d_Nx
    xmesh_idx = d_idx % d_Nx

    # find spatial boundaries of subdomain
    xn = input_deck.technique["dd_mesh"]["x"][xmesh_idx]
    xp = input_deck.technique["dd_mesh"]["x"][xmesh_idx + 1]
    yn = input_deck.technique["dd_mesh"]["y"][ymesh_idx]
    yp = input_deck.technique["dd_mesh"]["y"][ymesh_idx + 1]
    zn = input_deck.technique["dd_mesh"]["z"][zmesh_idx]
    zp = input_deck.technique["dd_mesh"]["z"][zmesh_idx + 1]

    # find boundary indices in tally mesh
    mesh_xn = int(np.where(input_deck.mesh_tallies[idx].x == xn)[0])
    mesh_xp = int(np.where(input_deck.mesh_tallies[idx].x == xp)[0]) + 1
    mesh_yn = int(np.where(input_deck.mesh_tallies[idx].y == yn)[0])
    mesh_yp = int(np.where(input_deck.mesh_tallies[idx].y == yp)[0]) + 1
    mesh_zn = int(np.where(input_deck.mesh_tallies[idx].z == zn)[0])
    mesh_zp = int(np.where(input_deck.mesh_tallies[idx].z == zp)[0]) + 1

    return mesh_xn, mesh_xp, mesh_yn, mesh_yp, mesh_zn, mesh_zp


def generate_cs_centers(mcdc, N_dim=3, seed=123456789):
    N_cs_bins = int(mcdc["cs_tallies"]["filter"]["N_cs_bins"])
    x_lims = (
        mcdc["cs_tallies"]["filter"]["x"][0][-1],
        mcdc["cs_tallies"]["filter"]["x"][0][0],
    )
    y_lims = (
        mcdc["cs_tallies"]["filter"]["y"][0][-1],
        mcdc["cs_tallies"]["filter"]["y"][0][0],
    )

    # Generate Halton sequence according to the seed
    halton_seq = Halton(d=N_dim, seed=seed)
    points = halton_seq.random(n=N_cs_bins)

    # Extract x and y coordinates as tuples separately, scaled to the problem dimensions
    x_coords = tuple(points[:, 0] * (x_lims[1] - x_lims[0]) + x_lims[0])
    y_coords = tuple(points[:, 1] * (y_lims[1] - y_lims[0]) + y_lims[0])

    return (x_coords, y_coords)


def prepare():
    """
    Preparing the MC transport simulation:
      (1) Adapt kernels
      (2) Make types
      (3) Create and set up global variable container `mcdc`
    """

    prepare_domain_decomposition()

    # =========================================================================
    # Create root universe if not defined
    # =========================================================================

    if input_deck.universes[0] == None:
        N_cell = len(input_deck.cells)
        root_universe = UniverseCard()
        root_universe.ID = 0
        root_universe.cell_IDs = np.zeros(N_cell, int)
        for i, cell in enumerate(input_deck.cells):
            root_universe.cell_IDs[i] = cell.ID
        input_deck.universes[0] = root_universe

    # =========================================================================
    # Prepare cell region RPN (Reverse Polish Notation)
    #   - Replace halfspace region ID with its surface and insert
    #     complement operator if the sense is negative.
    # =========================================================================

    for cell in input_deck.cells:
        i = 0
        while i < len(cell._region_RPN):
            token = cell._region_RPN[i]
            if token >= 0:
                surface_ID = input_deck.regions[token].A
                sense = input_deck.regions[token].B
                cell._region_RPN[i] = surface_ID
                if sense < 0:
                    cell._region_RPN.insert(i + 1, BOOL_NOT)
            i += 1

    # =========================================================================
    # Time census-based tally
    # =========================================================================
    # Reset time grid size of all tallies if census-based tally is desired

    if input_deck.setting["census_based_tally"]:
        N_bin = input_deck.setting["census_tally_frequency"]
        for tally in input_deck.mesh_tallies:
            tally.N_bin *= N_bin / (len(tally.t) - 1)
            tally.t = np.zeros(N_bin + 1)

    # =========================================================================
    # Adapt kernels
    # =========================================================================

    kernel.adapt_rng(nb.config.DISABLE_JIT)

    # =========================================================================
    # Make types
    # =========================================================================

    type_.make_type_particle(input_deck)
    type_.make_type_particle_record(input_deck)
    type_.make_type_nuclide(input_deck)
    type_.make_type_material(input_deck)
    type_.make_type_surface(input_deck)
    type_.make_type_cell(input_deck)
    type_.make_type_lattice(input_deck)
    type_.make_type_source(input_deck)
    type_.make_type_mesh_tally(input_deck)
    type_.make_type_surface_tally(input_deck)
    type_.make_type_cell_tally(input_deck)
    type_.make_type_cs_tally(input_deck)
    type_.make_type_setting(input_deck)
    type_.make_type_uq(input_deck)
    type_.make_type_domain_decomp(input_deck)
    type_.make_type_dd_turnstile_event(input_deck)
    type_.make_type_technique(input_deck)
    type_.make_type_global(input_deck)
    type_.make_size_rpn(input_deck)
    kernel.adapt_rng(nb.config.DISABLE_JIT)

    input_deck.setting["target"] = config.target

    # =========================================================================
    # Create the global variable container
    #   TODO: Better alternative?
    # =========================================================================

    mcdc_arr = np.zeros(1, dtype=type_.global_)
    mcdc = mcdc_arr[0]

    # Now, set up the global variable container

    # Get modes
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    # =========================================================================
    # Nuclides
    # =========================================================================

    N_nuclide = len(input_deck.nuclides)
    for i in range(N_nuclide):
        # General data
        for name in ["ID", "fissionable"]:
            copy_field(mcdc["nuclides"][i], input_deck.nuclides[i], name)

        # MG data
        if mode_MG:
            for name in [
                "G",
                "J",
                "speed",
                "decay",
                "total",
                "capture",
                "scatter",
                "fission",
                "nu_s",
                "nu_f",
                "nu_p",
                "nu_d",
                "chi_s",
                "chi_p",
                "chi_d",
            ]:
                copy_field(mcdc["nuclides"][i], input_deck.nuclides[i], name)

        # CE data (load data from XS library)
        dir_name = os.getenv("MCDC_XSLIB")
        if mode_CE:
            nuc_name = input_deck.nuclides[i].name
            with h5py.File(dir_name + "/" + nuc_name + ".h5", "r") as f:
                # Atomic weight ratio
                mcdc["nuclides"][i]["A"] = f["A"][()]
                # Energy grids
                for name in [
                    "E_xs",
                    "E_nu_p",
                    "E_nu_d",
                    "E_chi_p",
                    "E_chi_d1",
                    "E_chi_d2",
                    "E_chi_d3",
                    "E_chi_d4",
                    "E_chi_d5",
                    "E_chi_d6",
                ]:
                    mcdc["nuclides"][i]["N" + name] = len(f[name][:])
                    mcdc["nuclides"][i][name][: len(f[name][:])] = f[name][:]

                # XS
                for name in ["capture", "scatter", "fission"]:
                    mcdc["nuclides"][i]["ce_" + name][: len(f[name][:])] = f[name][:]
                    mcdc["nuclides"][i]["ce_total"][: len(f[name][:])] += f[name][:]

                # Fission production
                mcdc["nuclides"][i]["ce_nu_p"][: len(f["nu_p"][:])] = f["nu_p"][:]
                for j in range(6):
                    mcdc["nuclides"][i]["ce_nu_d"][j][: len(f["nu_d"][j, :])] = f[
                        "nu_d"
                    ][j, :]

                # Fission spectrum
                mcdc["nuclides"][i]["ce_chi_p"][: len(f["chi_p"][:])] = f["chi_p"][:]
                for j in range(6):
                    mcdc["nuclides"][i]["ce_chi_d%i" % (j + 1)][
                        : len(f["chi_d%i" % (j + 1)][:])
                    ] = f["chi_d%i" % (j + 1)][:]

                # Decay
                mcdc["nuclides"][i]["ce_decay"][: len(f["decay_rate"][:])] = f[
                    "decay_rate"
                ][:]

    # =========================================================================
    # Materials
    # =========================================================================

    N_material = len(input_deck.materials)
    for i in range(N_material):
        for name in type_.material.names:
            if name in ["nuclide_IDs", "nuclide_densities"]:
                mcdc["materials"][i][name][: mcdc["materials"][i]["N_nuclide"]] = (
                    getattr(input_deck.materials[i], name)
                )
            else:
                copy_field(mcdc["materials"][i], input_deck.materials[i], name)

    # =========================================================================
    # Surfaces
    # =========================================================================

    N_surface = len(input_deck.surfaces)
    j = 0  # indexing reflective surfaces
    for i in range(N_surface):

        surface = mcdc["surfaces"][i]
        surface_input = input_deck.surfaces[i]

        # Direct assignment
        for name in type_.surface.names:
            if name not in [
                "type",
                "BC",
                "tally_IDs",
                "move_velocities",
                "move_translations",
                "move_time_grid",
            ]:
                copy_field(surface, surface_input, name)

        # Type
        if surface_input.type == "plane-x":
            surface["type"] = SURFACE_LINEAR
            surface["type"] += SURFACE_PLANE_X
        elif surface_input.type == "plane-y":
            surface["type"] = SURFACE_LINEAR
            surface["type"] += SURFACE_PLANE_Y
        elif surface_input.type == "plane-z":
            surface["type"] = SURFACE_LINEAR
            surface["type"] += SURFACE_PLANE_Z
        elif surface_input.type == "plane":
            surface["type"] = SURFACE_LINEAR
            surface["type"] += SURFACE_PLANE
        elif surface_input.type == "cylinder-x":
            surface["type"] = SURFACE_QUADRATIC
            surface["type"] += SURFACE_CYLINDER_X
        elif surface_input.type == "cylinder-y":
            surface["type"] = SURFACE_QUADRATIC
            surface["type"] += SURFACE_CYLINDER_Y
        elif surface_input.type == "cylinder-z":
            surface["type"] = SURFACE_QUADRATIC
            surface["type"] += SURFACE_CYLINDER_Z
        elif surface_input.type == "sphere":
            surface["type"] = SURFACE_QUADRATIC
            surface["type"] += SURFACE_SPHERE
        elif surface_input.type == "quadric":
            surface["type"] = SURFACE_QUADRATIC
            surface["type"] += SURFACE_SPHERE

        # Boundary condition
        if input_deck.surfaces[i].boundary_type == "interface":
            mcdc["surfaces"][i]["BC"] = BC_NONE
        elif input_deck.surfaces[i].boundary_type == "vacuum":
            mcdc["surfaces"][i]["BC"] = BC_VACUUM
            # mcdc["boundary_surface_IDs"][j] = i
            # j += 1  # increasing the reflecting surface indexer
        elif input_deck.surfaces[i].boundary_type == "reflective":
            mcdc["surfaces"][i]["BC"] = BC_REFLECTIVE
            mcdc["boundary_surface_IDs"][j] = i
            j += 1  # increasing the reflecting surface indexer

        # Variables with possible different sizes
        for name in ["tally_IDs"]:
            N = len(getattr(input_deck.surfaces[i], name))
            mcdc["surfaces"][i][name][:N] = getattr(input_deck.surfaces[i], name)

        # Moves
        if surface["moving"]:
            for n in range(surface["N_move"]):
                duration = surface_input.move_durations[n]
                velocity = surface_input.move_velocities[n]

                surface["move_velocities"][n] = velocity

                t_start = surface["move_time_grid"][n]
                surface["move_time_grid"][n + 1] = t_start + duration

                trans_start = surface["move_translations"][n]
                surface["move_translations"][n + 1] = trans_start + velocity * duration

    # =========================================================================
    # Set cells
    # =========================================================================

    N_cell = len(input_deck.cells)
    surface_data_idx = 0
    region_data_idx = 0
    for i in range(N_cell):
        cell = mcdc["cells"][i]
        cell_input = input_deck.cells[i]

        # Directly transferables
        for name in ["ID", "fill_ID", "translation", "rotation", "N_tally"]:
            copy_field(cell, cell_input, name)

        # Fill type
        if cell_input.fill_type == "material":
            cell["fill_type"] = FILL_MATERIAL
        elif cell_input.fill_type == "universe":
            cell["fill_type"] = FILL_UNIVERSE
        elif cell_input.fill_type == "lattice":
            cell["fill_type"] = FILL_LATTICE

        # Fill translation
        if np.max(np.abs(cell["translation"])) > 0.0:
            cell["fill_translated"] = True

        # Fill rotation
        if np.max(np.abs(cell["rotation"])) > 0.0:
            cell["fill_rotated"] = True

            # Convert rotation
            cell["rotation"][0] *= PI / 180.0
            cell["rotation"][1] *= PI / 180.0
            cell["rotation"][2] *= PI / 180.0

        # Surface IDs
        cell["surface_data_idx"] = surface_data_idx
        cell["N_surface"] = len(cell_input.surface_IDs)
        # The data
        start = surface_data_idx
        end = start + cell["N_surface"]
        mcdc["cells_data_surface"][start:end] = cell_input.surface_IDs
        surface_data_idx += cell["N_surface"]

        # Region RPN tokens
        cell["region_data_idx"] = region_data_idx
        cell["N_region"] = len(cell_input._region_RPN)
        # The data
        start = region_data_idx
        end = start + cell["N_region"]
        mcdc["cells_data_region"][start:end] = cell_input._region_RPN
        region_data_idx += cell["N_region"]

        # Variables with possible different sizes
        for name in ["tally_IDs"]:
            N = len(getattr(input_deck.cells[i], name))
            mcdc["cells"][i][name][:N] = getattr(input_deck.cells[i], name)

    # =========================================================================
    # Set universes
    # =========================================================================

    N_universe = len(input_deck.universes)
    cell_data_idx = 0
    for i in range(N_universe):
        universe = mcdc["universes"][i]
        universe_input = input_deck.universes[i]

        # Directly transferables
        for name in ["ID"]:
            copy_field(universe, universe_input, name)

        # Cells IDs
        universe["cell_data_idx"] = cell_data_idx
        universe["N_cell"] = len(universe_input.cell_IDs)
        # Cell ID data
        start = cell_data_idx
        end = start + universe["N_cell"]
        mcdc["universes_data_cell"][start:end] = universe_input.cell_IDs
        cell_data_idx += universe["N_cell"]

    # =========================================================================
    # Lattices
    # =========================================================================

    N_lattice = len(input_deck.lattices)
    for i in range(N_lattice):
        for name in type_.lattice.names:
            if name not in ["universe_IDs", "t0", "dt", "Nt"]:
                mcdc["lattices"][i][name] = getattr(input_deck.lattices[i], name)

        # Universe IDs
        Nx = mcdc["lattices"][i]["Nx"]
        Ny = mcdc["lattices"][i]["Ny"]
        Nz = mcdc["lattices"][i]["Nz"]
        mcdc["lattices"][i]["universe_IDs"][:Nx, :Ny, :Nz] = input_deck.lattices[
            i
        ].universe_IDs

        # Default for time grid
        mcdc["lattices"][i]["t0"] = 0.0
        mcdc["lattices"][i]["dt"] = INF
        mcdc["lattices"][i]["Nt"] = 1

    # =========================================================================
    # Source
    # =========================================================================

    N_source = len(input_deck.sources)
    for i in range(N_source):
        for name in type_.source.names:
            copy_field(mcdc["sources"][i], input_deck.sources[i], name)

    # Normalize source probabilities
    tot = 1e-16
    for S in mcdc["sources"]:
        tot += S["prob"]
    for S in mcdc["sources"]:
        S["prob"] /= tot

    # =========================================================================
    # Tally
    # =========================================================================

    N_mesh_tally = len(input_deck.mesh_tallies)
    N_surface_tally = len(input_deck.surface_tallies)
    N_cell_tally = len(input_deck.cell_tallies)
    N_cs_tally = len(input_deck.cs_tallies)
    tally_bin_size = 0

    # Mesh tallies
    for i in range(N_mesh_tally):
        # Direct assignment
        copy_field(mcdc["mesh_tallies"][i], input_deck.mesh_tallies[i], "N_bin")

        # Filters (variables with possible different sizes)
        if not input_deck.technique["domain_decomposition"]:
            for name in ["x", "y", "z", "t", "mu", "azi", "g"]:
                N = len(getattr(input_deck.mesh_tallies[i], name))
                mcdc["mesh_tallies"][i]["filter"][name][:N] = getattr(
                    input_deck.mesh_tallies[i], name
                )

        else:  # decomposed mesh filters
            mcdc["technique"]["dd_xsum"] = len(input_deck.mesh_tallies[i].x) - 1
            mcdc["technique"]["dd_ysum"] = len(input_deck.mesh_tallies[i].y) - 1
            mcdc["technique"]["dd_zsum"] = len(input_deck.mesh_tallies[i].z) - 1

            mxn, mxp, myn, myp, mzn, mzp = dd_mesh_bounds(i)

            # Filters
            new_x = input_deck.mesh_tallies[i].x[mxn:mxp]
            new_y = input_deck.mesh_tallies[i].y[myn:myp]
            new_z = input_deck.mesh_tallies[i].z[mzn:mzp]
            xlen = len(new_x)
            ylen = len(new_y)
            zlen = len(new_z)
            mcdc["mesh_tallies"][i]["filter"]["x"][:xlen] = new_x
            mcdc["mesh_tallies"][i]["filter"]["y"][:ylen] = new_y
            mcdc["mesh_tallies"][i]["filter"]["z"][:zlen] = new_z
            mcdc["technique"]["dd_xlen"] = xlen - 1
            mcdc["technique"]["dd_ylen"] = ylen - 1
            mcdc["technique"]["dd_zlen"] = zlen - 1
            for name in ["t", "mu", "azi", "g"]:
                N = len(getattr(input_deck.mesh_tallies[i], name))
                mcdc["mesh_tallies"][i]["filter"][name][:N] = getattr(
                    input_deck.mesh_tallies[i], name
                )

        # Set tally scores
        N_score = len(input_deck.mesh_tallies[i].scores)
        mcdc["mesh_tallies"][i]["N_score"] = N_score
        for j in range(N_score):
            score_name = input_deck.mesh_tallies[i].scores[j]
            score_type = None
            if score_name == "flux":
                score_type = SCORE_FLUX
            elif score_name == "density":
                score_type = SCORE_DENSITY
            elif score_name == "total":
                score_type = SCORE_TOTAL
            elif score_name == "fission":
                score_type = SCORE_FISSION
            elif score_name == "net-current":
                score_type = SCORE_NET_CURRENT
            elif score_name == "mu-sq":
                score_type = SCORE_MU_SQ
            elif score_name == "time-moment-flux":
                score_type = SCORE_TIME_MOMENT_FLUX
            elif score_name == "space-moment-flux":
                score_type = SCORE_SPACE_MOMENT_FLUX
            elif score_name == "time-moment-current":
                score_type = SCORE_TIME_MOMENT_CURRENT
            elif score_name == "space-moment-current":
                score_type = SCORE_SPACE_MOMENT_CURRENT
            elif score_name == "time-moment-mu-sq":
                score_type = SCORE_TIME_MOMENT_MU_SQ
            elif score_name == "space-moment-mu-sq":
                score_type = SCORE_SPACE_MOMENT_MU_SQ
            mcdc["mesh_tallies"][i]["scores"][j] = score_type

        # Filter grid sizes
        Nmu = len(input_deck.mesh_tallies[i].mu) - 1
        N_azi = len(input_deck.mesh_tallies[i].azi) - 1
        Ng = len(input_deck.mesh_tallies[i].g) - 1
        Nx = len(input_deck.mesh_tallies[i].x) - 1
        Ny = len(input_deck.mesh_tallies[i].y) - 1
        Nz = len(input_deck.mesh_tallies[i].z) - 1
        Nt = len(input_deck.mesh_tallies[i].t) - 1
        mcdc["mesh_tallies"][i]["filter"]["Nmu"] = Nmu
        mcdc["mesh_tallies"][i]["filter"]["N_azi"] = N_azi
        mcdc["mesh_tallies"][i]["filter"]["Ng"] = Ng
        mcdc["mesh_tallies"][i]["filter"]["Nx"] = Nx
        mcdc["mesh_tallies"][i]["filter"]["Ny"] = Ny
        mcdc["mesh_tallies"][i]["filter"]["Nz"] = Nz
        mcdc["mesh_tallies"][i]["filter"]["Nt"] = Nt

        # Decompose mesh tallies
        if input_deck.technique["domain_decomposition"]:
            Nmu = len(input_deck.mesh_tallies[i].mu) - 1
            N_azi = len(input_deck.mesh_tallies[i].azi) - 1
            Ng = len(input_deck.mesh_tallies[i].g) - 1
            Nx = len(input_deck.mesh_tallies[i].x[mxn:mxp]) - 1
            Ny = len(input_deck.mesh_tallies[i].y[myn:myp]) - 1
            Nz = len(input_deck.mesh_tallies[i].z[mzn:mzp]) - 1
            Nt = len(input_deck.mesh_tallies[i].t) - 1
            mcdc["mesh_tallies"][i]["N_bin"] = Nx * Ny * Nz * Nt * Nmu * N_azi * Ng
            mcdc["mesh_tallies"][i]["filter"]["Nx"] = Nx
            mcdc["mesh_tallies"][i]["filter"]["Ny"] = Ny
            mcdc["mesh_tallies"][i]["filter"]["Nz"] = Nz

        # Update N_bin
        mcdc["mesh_tallies"][i]["N_bin"] *= N_score

        # Filter strides
        stride = N_score
        if Nz > 1:
            mcdc["mesh_tallies"][i]["stride"]["z"] = stride
            stride *= Nz
        if Ny > 1:
            mcdc["mesh_tallies"][i]["stride"]["y"] = stride
            stride *= Ny
        if Nx > 1:
            mcdc["mesh_tallies"][i]["stride"]["x"] = stride
            stride *= Nx
        if Nt > 1:
            mcdc["mesh_tallies"][i]["stride"]["t"] = stride
            stride *= Nt
        if Ng > 1:
            mcdc["mesh_tallies"][i]["stride"]["g"] = stride
            stride *= Ng
        if N_azi > 1:
            mcdc["mesh_tallies"][i]["stride"]["azi"] = stride
            stride *= N_azi
        if Nmu > 1:
            mcdc["mesh_tallies"][i]["stride"]["mu"] = stride
            stride *= Nmu

        # Set tally stride and accumulate total tally size
        mcdc["mesh_tallies"][i]["stride"]["tally"] = tally_bin_size
        tally_bin_size += mcdc["mesh_tallies"][i]["N_bin"]

    # Surface tallies
    for i in range(N_surface_tally):
        # Direct assignment
        copy_field(mcdc["surface_tallies"][i], input_deck.surface_tallies[i], "N_bin")

        # Filters (variables with possible different sizes)
        for name in ["t", "mu", "azi", "g"]:
            N = len(getattr(input_deck.surface_tallies[i], name))
            mcdc["surface_tallies"][i]["filter"][name][:N] = getattr(
                input_deck.surface_tallies[i], name
            )

        # Set tally scores and their strides
        N_score = len(input_deck.surface_tallies[i].scores)
        mcdc["surface_tallies"][i]["N_score"] = N_score
        for j in range(N_score):
            score_name = input_deck.surface_tallies[i].scores[j]
            mcdc["surface_tallies"][i]["scores"][j] = SCORE_NET_CURRENT

        # Filter grid sizes
        Nmu = len(input_deck.surface_tallies[i].mu) - 1
        N_azi = len(input_deck.surface_tallies[i].azi) - 1
        Ng = len(input_deck.surface_tallies[i].g) - 1
        Nt = len(input_deck.surface_tallies[i].t) - 1

        # Update N_bin
        mcdc["surface_tallies"][i]["N_bin"] *= N_score

        # Filter strides
        stride = N_score
        if Nt > 1:
            mcdc["surface_tallies"][i]["stride"]["t"] = stride
            stride *= Nt
        if Ng > 1:
            mcdc["surface_tallies"][i]["stride"]["g"] = stride
            stride *= Ng
        if N_azi > 1:
            mcdc["surface_tallies"][i]["stride"]["azi"] = stride
            stride *= N_azi
        if Nmu > 1:
            mcdc["surface_tallies"][i]["stride"]["mu"] = stride
            stride *= Nmu

        # Set tally stride and accumulate total tally size
        mcdc["surface_tallies"][i]["stride"]["tally"] = tally_bin_size
        tally_bin_size += mcdc["surface_tallies"][i]["N_bin"]

    # Cell tallies
    for i in range(N_cell_tally):
        copy_field(mcdc["cell_tallies"][i], input_deck.cell_tallies[i], "N_bin")

        # Filters (variables with possible different sizes)
        for name in ["t", "mu", "azi", "g"]:
            N = len(getattr(input_deck.cell_tallies[i], name))
            mcdc["cell_tallies"][i]["filter"][name][:N] = getattr(
                input_deck.cell_tallies[i], name
            )

        # Differentiating the tallies by cell_ID
        mcdc["cell_tallies"][i]["filter"]["cell_ID"] = getattr(
            input_deck.cell_tallies[i], "cell_ID"
        )

        # Set tally scores and their strides
        N_score = len(input_deck.cell_tallies[i].scores)
        mcdc["cell_tallies"][i]["N_score"] = N_score
        for j in range(N_score):
            score_name = input_deck.cell_tallies[i].scores[j]
            score_type = None
            if score_name == "flux":
                score_type = SCORE_FLUX
            elif score_name == "fission":
                score_type = SCORE_FISSION
            elif score_name == "net-current":
                score_type = SCORE_NET_CURRENT
            mcdc["cell_tallies"][i]["scores"][j] = score_type

        # Filter grid sizes
        Nmu = len(input_deck.cell_tallies[i].mu) - 1
        N_azi = len(input_deck.cell_tallies[i].azi) - 1
        Ng = len(input_deck.cell_tallies[i].g) - 1
        Nt = len(input_deck.cell_tallies[i].t) - 1
        mcdc["cell_tallies"][i]["filter"]["Ng"] = Ng
        mcdc["cell_tallies"][i]["filter"]["Nt"] = Nt

        # Update N_bin
        mcdc["cell_tallies"][i]["N_bin"] *= N_score

        # Filter strides
        stride = N_score
        if Nt > 1:
            mcdc["cell_tallies"][i]["stride"]["t"] = stride
            stride *= Nt
        if Ng > 1:
            mcdc["cell_tallies"][i]["stride"]["g"] = stride
            stride *= Ng
        if N_azi > 1:
            mcdc["cell_tallies"][i]["stride"]["azi"] = stride
            stride *= N_azi
        if Nmu > 1:
            mcdc["cell_tallies"][i]["stride"]["mu"] = stride
            stride *= Nmu

        # Set tally stride and accumulate total tally size
        mcdc["cell_tallies"][i]["stride"]["tally"] = tally_bin_size
        tally_bin_size += mcdc["cell_tallies"][i]["N_bin"]

    # CS tallies
    for i in range(N_cs_tally):
        # Direct assignment
        copy_field(mcdc["cs_tallies"][i], input_deck.cs_tallies[i], "N_bin")

        mcdc["cs_tallies"][i]["filter"]["N_cs_bins"] = input_deck.cs_tallies[
            i
        ].N_cs_bins[0]
        mcdc["cs_tallies"][i]["filter"]["cs_bin_size"] = input_deck.cs_tallies[
            i
        ].cs_bin_size[0]

        # Filters (variables with possible different sizes)
        if not input_deck.technique["domain_decomposition"]:
            for name in ["x", "y", "z", "t", "mu", "azi", "g"]:
                N = len(getattr(input_deck.cs_tallies[i], name))
                mcdc["cs_tallies"][i]["filter"][name][:N] = getattr(
                    input_deck.cs_tallies[i], name
                )

        mcdc["cs_tallies"][i]["filter"]["cs_centers"] = generate_cs_centers(mcdc)

        # Set tally scores
        N_score = len(input_deck.cs_tallies[i].scores)
        mcdc["cs_tallies"][i]["N_score"] = N_score
        for j in range(N_score):
            score_name = input_deck.cs_tallies[i].scores[j]
            score_type = None
            if score_name == "flux":
                score_type = SCORE_FLUX
            elif score_name == "density":
                score_type = SCORE_DENSITY
            elif score_name == "total":
                score_type = SCORE_TOTAL
            elif score_name == "fission":
                score_type = SCORE_FISSION
            elif score_name == "net-current":
                score_type = SCORE_NET_CURRENT
            mcdc["cs_tallies"][i]["scores"][j] = score_type

        # Update N_bin
        mcdc["cs_tallies"][i]["N_bin"] *= N_score

        # Set tally stride and accumulate total tally size
        mcdc["cs_tallies"][i]["stride"]["tally"] = tally_bin_size
        tally_bin_size += mcdc["cs_tallies"][i]["filter"]["N_cs_bins"]

    # Set tally data
    if not input_deck.technique["uq"]:
        tally_bin_N_copies = 3
    else:
        tally_bin_N_copies = 5
    data_tally = np.zeros((tally_bin_N_copies, tally_bin_size), dtype=type_.float64)

    # =========================================================================
    # Platform Targeting, Adapters, Toggles, etc
    # =========================================================================

    if config.target == "gpu":
        if MPI.COMM_WORLD.Get_rank() != 0:
            adapt.harm.config.should_compile(adapt.harm.config.ShouldCompile.NEVER)
        elif config.caching == False:
            adapt.harm.config.should_compile(adapt.harm.config.ShouldCompile.ALWAYS)
        if not adapt.HAS_HARMONIZE:
            print_error(
                "No module named 'harmonize' - GPU functionality not available. "
            )
        adapt.gpu_forward_declare(config.args)

    adapt.set_toggle("iQMC", input_deck.technique["iQMC"])
    adapt.set_toggle("domain_decomp", input_deck.technique["domain_decomposition"])
    adapt.eval_toggle()
    adapt.target_for(config.target)
    if config.target == "gpu":
        build_gpu_progs(input_deck, config.args)
    adapt.nopython_mode((config.mode == "numba") or (config.mode == "numba_debug"))

    # =========================================================================
    # Setting
    # =========================================================================

    for name in type_.setting.names:
        copy_field(mcdc["setting"], input_deck.setting, name)

    t_limit = max(
        [
            tally["filter"]["t"][-1]
            for tally in list(mcdc["mesh_tallies"])
            + list(mcdc["surface_tallies"])
            + list(mcdc["cell_tallies"])
        ]
    )

    if (
        len(input_deck.mesh_tallies)
        + len(input_deck.surface_tallies)
        + len(input_deck.cell_tallies)
        == 0
    ):
        t_limit = INF

    # Replace the time limit if time census-based tally is used
    if mcdc["setting"]["census_based_tally"]:
        t_limit = mcdc["setting"]["census_time"][-2]

    # Set appropriate time boundary
    if mcdc["setting"]["time_boundary"] > t_limit:
        mcdc["setting"]["time_boundary"] = t_limit

    if input_deck.technique["iQMC"]:
        if len(mcdc["technique"]["iqmc"]["mesh"]["t"]) - 1 > 1:
            if (
                mcdc["setting"]["time_boundary"]
                > input_deck.technique["iqmc"]["mesh"]["t"][-1]
            ):
                mcdc["setting"]["time_boundary"] = input_deck.technique["iqmc"]["mesh"][
                    "t"
                ][-1]

    # =========================================================================
    # Technique
    # =========================================================================

    # Flags
    for name in [
        "weighted_emission",
        "implicit_capture",
        "population_control",
        "weight_window",
        "domain_decomposition",
        "weight_roulette",
        "iQMC",
        "IC_generator",
        "branchless_collision",
        "uq",
        "delta_tracking",
        "collision_estimator",
        "dt_cutoff_energy",
        "dt_material_exclusion",
    ]:
        copy_field(mcdc["technique"], input_deck.technique, name)

    # =========================================================================
    # Population control
    # =========================================================================

    # Population control technique (PCT)
    pct = input_deck.technique["pct"]
    if pct == "combing":
        mcdc["technique"]["pct"] = PCT_COMBING
    elif pct == "combing-weight":
        mcdc["technique"]["pct"] = PCT_COMBING_WEIGHT
    elif pct == "splitting-roulette":
        mcdc["technique"]["pct"] = PCT_SPLITTING_ROULETTE
    elif pct == "splitting-roulette-weight":
        mcdc["technique"]["pct"] = PCT_SPLITTING_ROULETTE_WEIGHT
    mcdc["technique"]["pc_factor"] = input_deck.technique["pc_factor"]

    # =========================================================================
    # IC generator
    # =========================================================================

    for name in [
        "IC_N_neutron",
        "IC_N_precursor",
        "IC_neutron_density",
        "IC_neutron_density_max",
        "IC_precursor_density",
        "IC_precursor_density_max",
    ]:
        copy_field(mcdc["technique"], input_deck.technique, name)

    # =========================================================================
    # Weight window (WW)
    # =========================================================================

    # WW mesh
    for name in type_.mesh_names[:-1]:
        copy_field(
            mcdc["technique"]["ww"]["mesh"], input_deck.technique["ww"]["mesh"], name
        )

    mcdc["technique"]["ww"]["mesh"]["Nx"] = (
        len(input_deck.technique["ww"]["mesh"]["x"]) - 1
    )
    mcdc["technique"]["ww"]["mesh"]["Ny"] = (
        len(input_deck.technique["ww"]["mesh"]["y"]) - 1
    )
    mcdc["technique"]["ww"]["mesh"]["Nz"] = (
        len(input_deck.technique["ww"]["mesh"]["z"]) - 1
    )
    mcdc["technique"]["ww"]["mesh"]["Nt"] = (
        len(input_deck.technique["ww"]["mesh"]["t"]) - 1
    )

    # WW parameters
    mcdc["technique"]["ww"]["width"] = input_deck.technique["ww"]["width"]
    mcdc["technique"]["ww"]["auto"] = input_deck.technique["ww"]["auto"]
    mcdc["technique"]["ww"]["epsilon"] = input_deck.technique["ww"]["epsilon"]
    mcdc["technique"]["ww"]["center"] = input_deck.technique["ww"]["center"]
    mcdc["technique"]["ww"]["save"] = input_deck.technique["ww"]["save"]
    mcdc["technique"]["ww"]["tally_idx"] = input_deck.technique["ww"]["tally_idx"]
    # =========================================================================
    # Weight roulette
    # =========================================================================

    # Threshold
    mcdc["technique"]["wr_threshold"] = input_deck.technique["wr_threshold"]

    # Survival probability
    mcdc["technique"]["wr_survive"] = input_deck.technique["wr_survive"]

    # =========================================================================
    # Domain Decomposition
    # =========================================================================

    # Set domain mesh
    if input_deck.technique["domain_decomposition"]:
        for name in ["x", "y", "z", "t", "mu", "azi"]:
            copy_field(
                mcdc["technique"]["dd_mesh"], input_deck.technique["dd_mesh"], name
            )
        mcdc["technique"]["dd_mesh"]["Nx"] = (
            input_deck.technique["dd_mesh"]["x"].size - 1
        )
        mcdc["technique"]["dd_mesh"]["Ny"] = (
            input_deck.technique["dd_mesh"]["y"].size - 1
        )
        mcdc["technique"]["dd_mesh"]["Nz"] = (
            input_deck.technique["dd_mesh"]["z"].size - 1
        )
        mcdc["technique"]["dd_mesh"]["Nt"] = (
            input_deck.technique["dd_mesh"]["t"].size - 1
        )
        mcdc["technique"]["dd_mesh"]["Nmu"] = (
            input_deck.technique["dd_mesh"]["mu"].size - 1
        )
        mcdc["technique"]["dd_mesh"]["N_azi"] = (
            input_deck.technique["dd_mesh"]["azi"].size - 1
        )
        # Set exchange rate
        for name in ["dd_exchange_rate"]:
            copy_field(mcdc["technique"], input_deck.technique, name)
        # Set domain index
        copy_field(mcdc, input_deck.technique, "dd_idx")
        copy_field(mcdc, input_deck.technique, "dd_local_rank")
        for name in ["xp", "xn", "yp", "yn", "zp", "zn"]:
            copy_field(mcdc["technique"], input_deck.technique, f"dd_{name}_neigh")
        copy_field(mcdc["technique"], input_deck.technique, "dd_work_ratio")

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================

    for name in type_.technique["iqmc"].names:
        if name not in [
            "mesh",
            "residual",
            "samples",
            "sweep_count",
            "total_source",
            "material_idx",
            "w_min",
            "score_list",
            "score",
        ]:
            copy_field(mcdc["technique"]["iqmc"], input_deck.technique["iqmc"], name)

    if input_deck.technique["iQMC"]:
        # pass in mesh
        iqmc = mcdc["technique"]["iqmc"]
        for name in ["x", "y", "z", "t"]:
            copy_field(iqmc["mesh"], input_deck.technique["iqmc"]["mesh"], name)
        Nx = len(input_deck.technique["iqmc"]["mesh"]["x"]) - 1
        Ny = len(input_deck.technique["iqmc"]["mesh"]["y"]) - 1
        Nz = len(input_deck.technique["iqmc"]["mesh"]["z"]) - 1
        Nt = len(input_deck.technique["iqmc"]["mesh"]["t"]) - 1
        iqmc["mesh"]["Nx"] = Nx
        iqmc["mesh"]["Ny"] = Ny
        iqmc["mesh"]["Nz"] = Nz
        iqmc["mesh"]["Nt"] = Nt
        # pass in score list
        for name, value in input_deck.technique["iqmc"]["score_list"].items():
            copy_field(
                iqmc["score_list"], input_deck.technique["iqmc"]["score_list"], name
            )
        # pass in initial tallies
        for name, value in input_deck.technique["iqmc"]["score"].items():
            mcdc["technique"]["iqmc"]["score"][name]["bin"] = value
        # minimum particle weight
        iqmc["w_min"] = 1e-13

    # =========================================================================
    # Variance Deconvolution - UQ
    # =========================================================================
    if mcdc["technique"]["uq"]:
        M = len(input_deck.uq_deltas["materials"])
        for i in range(M):
            idm = input_deck.uq_deltas["materials"][i].ID
            mcdc["technique"]["uq_"]["materials"][i]["info"]["ID"] = idm
            mcdc["technique"]["uq_"]["materials"][i]["info"]["distribution"] = (
                input_deck.uq_deltas["materials"][i].distribution
            )
            for name in input_deck.uq_deltas["materials"][i].flags:
                mcdc["technique"]["uq_"]["materials"][i]["flags"][name] = True
                mcdc["technique"]["uq_"]["materials"][i]["delta"][name] = getattr(
                    input_deck.uq_deltas["materials"][i], name
                )
            flags = mcdc["technique"]["uq_"]["materials"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
                flags["speed"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True
            if mcdc["materials"][idm]["N_nuclide"] > 1:
                for name in type_.uq_mat.names:
                    mcdc["technique"]["uq_"]["materials"][i]["mean"][name] = (
                        input_deck.materials[idm][name]
                    )

        N = len(input_deck.uq_deltas["nuclides"])
        for i in range(N):
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["distribution"] = (
                input_deck.uq_deltas["nuclides"][i].distribution
            )
            idn = input_deck.uq_deltas["nuclides"][i].ID
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["ID"] = idn
            for name in type_.uq_nuc.names:
                if name == "scatter":
                    G = input_deck.nuclides[idn].G
                    chi_s = input_deck.nuclides[idn].chi_s
                    scatter = input_deck.nuclides[idn].scatter
                    scatter_matrix = np.zeros((G, G))
                    for g in range(G):
                        scatter_matrix[g, :] = chi_s[g, :] * scatter[g]

                    mcdc["technique"]["uq_"]["nuclides"][i]["mean"][
                        name
                    ] = scatter_matrix
                else:
                    copy_field(
                        mcdc["technique"]["uq_"]["nuclides"][i]["mean"],
                        input_deck.nuclides[idn],
                        name,
                    )

            for name in input_deck.uq_deltas["nuclides"][i].flags:
                if "padding" in name:
                    continue
                mcdc["technique"]["uq_"]["nuclides"][i]["flags"][name] = True
                copy_field(
                    mcdc["technique"]["uq_"]["nuclides"][i]["delta"],
                    input_deck.uq_deltas["nuclides"][i],
                    name,
                )
            flags = mcdc["technique"]["uq_"]["nuclides"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True

    # =========================================================================
    # Delta tracking (majorant copmutation)
    # =========================================================================
    # TODO: Make seperate function that can be called without run

    micro_maj = False

    if mcdc["technique"]["delta_tracking"]:

        if input_deck.technique["plot_majorant"]:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(11, 8.5))

        if mode_CE:

            # TODO: Use numpy functions (currently can't as data is disorginized and requires extrapolation)
            try:
                import scipy
            except:
                print_error(
                    "SciPy not found!\n"
                    "Contininous energy delta tracking requires\n\
                     interpolation functions from Scipy to construct the marjoant,\n \
                     try `pip install scipy`"
                )

            # unify the energy grids from all nuclides
            majorant_energy_grid = np.array([])
            for n in range(N_nuclide):
                nuclide = mcdc["nuclides"][n]
                majorant_energy_grid = np.append(majorant_energy_grid, nuclide["E_xs"])

            # sort energy grid and eliminate duplicate points
            majorant_energy_grid = np.unique(majorant_energy_grid)
            majorant_xsec = np.zeros_like(majorant_energy_grid)

            # macro maj
            if micro_maj:

                # find the majorant at every point on the energy grid
                for n in range(N_nuclide):
                    nuclide = mcdc["nuclides"][n]

                    # builds an interploation funciton for a given nuclide
                    total_xsec_unified = scipy.interpolate.interp1d(
                        nuclide["E_xs"], nuclide["ce_total"], bounds_error=False
                    )

                    # evaluates nuclide interploation function at all unified energy grid points
                    total_xsec_unified = total_xsec_unified(majorant_energy_grid)

                    # compares old majorant xsec and the currently evaluated unified xsec and picks the larger xsecs
                    majorant_xsec = np.max((majorant_xsec, total_xsec_unified), axis=0)

                import matplotlib.pyplot as plt

                if input_deck.technique["plot_majorant"]:
                    plt.plot(
                        majorant_energy_grid[::5],
                        majorant_xsec[::5],
                        "k2",
                        label="majorant",
                        linewidth=0.75,
                    )
                    for n in range(N_nuclide):
                        nuclide = mcdc["nuclides"][n]
                        plt.plot(
                            nuclide["E_xs"],
                            nuclide["ce_total"],
                            label=input_deck.nuclides[n].name,
                            linewidth=0.75,
                        )
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel("E [ev]")
                    plt.ylabel("σ")
                    plt.legend()
                    plt.savefig("micro_majorant.pdf")

            else:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(11, 8.5))

                for m in range(N_material):

                    material = mcdc["materials"][m]

                    material_energy_grid = np.array([])

                    # copmute a unified energy grid across all nuclides of a given material
                    for n in range(material["N_nuclide"]):
                        nuclide = mcdc["nuclides"][n]
                        material_energy_grid = np.append(
                            material_energy_grid, nuclide["E_xs"]
                        )
                    material_energy_grid = np.unique(material_energy_grid)
                    MacroXS = np.zeros_like(material_energy_grid)

                    # compute the macroscopic total cross section of a material on its unified energy grid
                    for n in range(material["N_nuclide"]):
                        ID_nuclide = material["nuclide_IDs"][n]
                        nuclide = mcdc["nuclides"][ID_nuclide]

                        # Get nuclide density
                        N = material["nuclide_densities"][n]

                        # putting the microscopic cross-sections on the unifed material energy grid
                        total_micro_xsec_unified = scipy.interpolate.interp1d(
                            nuclide["E_xs"], nuclide["ce_total"], bounds_error=False
                        )
                        total_micro_xsec_unified = total_micro_xsec_unified(
                            material_energy_grid
                        )

                        # Accumulate
                        MacroXS += N * total_micro_xsec_unified

                    if input_deck.technique["plot_majorant"]:
                        plt.plot(
                            material_energy_grid,
                            MacroXS,
                            linewidth=0.5,
                            label="Mat {}".format(m),
                        )  # label=input_deck.materials[m].name,

                    # puting the total macroscopic cross sections on on the majorant energy grid
                    total_xsec_unified = scipy.interpolate.interp1d(
                        material_energy_grid, MacroXS, bounds_error=False
                    )
                    total_xsec_unified = total_xsec_unified(majorant_energy_grid)

                    # compares old majorant xsec and the currently evaluated unified xsec and picks the larger xsecs
                    majorant_xsec = np.max((majorant_xsec, total_xsec_unified), axis=0)

                if input_deck.technique["plot_majorant"]:
                    plt.plot(
                        majorant_energy_grid[::5],
                        majorant_xsec[::5],
                        "k2",
                        label="majorant",
                        linewidth=0.5,
                    )
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel("E [kev]")
                    plt.ylabel("σ")
                    plt.legend()
                    plt.savefig("macro_majorant.pdf")

        elif mode_MG:
            N_groups = mcdc["nuclides"][0]["G"]
            majorant_energy_grid = np.zeros(N_groups)
            majorant_xsec = np.zeros(N_groups)

            majorant_energy_grid = mcdc["nuclides"][0]["speed"][:]

            for i in range(N_groups):
                # print("g {}  ".format(i), mcdc["nuclides"][:]["total"][:, i])
                majorant_xsec[i] = np.max(mcdc["nuclides"][:]["total"][:, i])

            # print("here!")
            import matplotlib.pyplot as plt

            if input_deck.technique["plot_majorant"]:
                plt.figure(figsize=(11, 8.5))
                plt.plot(majorant_energy_grid, majorant_xsec, "k2", label="majorant")
                for n in range(N_nuclide):
                    nuclide = mcdc["nuclides"][n]
                    plt.plot(
                        nuclide["speed"][:],
                        nuclide["total"],
                        "-*",
                        label=input_deck.nuclides[n].name,
                    )
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("E [kev]")
                plt.ylabel("σ")
                plt.legend()
                plt.savefig("majorant.pdf")

        # This might need to use copy_feild but not sure why
        mcdc["technique"]["micro_majorant_xsec"] = majorant_xsec[:]
        mcdc["technique"]["majorant_energy"] = majorant_energy_grid[:]
        mcdc["technique"]["N_majorant"] = np.size(majorant_xsec)

    # =========================================================================
    # MPI
    # =========================================================================

    # MPI parameters
    mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
    mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
    mcdc["mpi_master"] = mcdc["mpi_rank"] == 0

    # Distribute work to MPI ranks
    if mcdc["technique"]["domain_decomposition"]:
        kernel.distribute_work_dd(mcdc["setting"]["N_particle"], mcdc)
    else:
        kernel.distribute_work(mcdc["setting"]["N_particle"], mcdc)

    # =========================================================================
    # Particle banks
    # =========================================================================

    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"
    mcdc["bank_future"]["tag"] = "future"

    # IC generator banks
    if mcdc["technique"]["IC_generator"]:
        mcdc["technique"]["IC_bank_neutron_local"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor_local"]["tag"] = "precursor"
        mcdc["technique"]["IC_bank_neutron"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor"]["tag"] = "precursor"

    # =========================================================================
    # Eigenvalue (or fixed-source)
    # =========================================================================

    # Initial guess
    mcdc["k_eff"] = mcdc["setting"]["k_init"]

    # Activate tally scoring for fixed-source
    if not mcdc["setting"]["mode_eigenvalue"]:
        mcdc["cycle_active"] = True

    # All active eigenvalue cycle?
    elif mcdc["setting"]["N_inactive"] == 0:
        mcdc["cycle_active"] = True

    # =========================================================================
    # Source file
    #   TODO: Use parallel h5py
    # =========================================================================

    # All ranks, take turn
    for i in range(mcdc["mpi_size"]):
        if mcdc["mpi_rank"] == i:
            if mcdc["setting"]["source_file"]:
                with h5py.File(mcdc["setting"]["source_file_name"], "r") as f:
                    # Get source particle size
                    N_particle = f["particles_size"][()]

                    # Redistribute work
                    kernel.distribute_work(N_particle, mcdc)
                    N_local = mcdc["mpi_work_size"]
                    start = mcdc["mpi_work_start"]
                    end = start + N_local

                    # Add particles to source bank
                    mcdc["bank_source"]["particles"][:N_local] = f["particles"][
                        start:end
                    ]
                    mcdc["bank_source"]["size"] = N_local
        MPI.COMM_WORLD.Barrier()

    # =========================================================================
    # IC file
    # =========================================================================

    if mcdc["setting"]["IC_file"]:
        with h5py.File(mcdc["setting"]["IC_file_name"], "r") as f:
            # =================================================================
            # Set neutron source
            # =================================================================

            # Get source particle size
            N_particle = f["IC/neutrons_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["IC/neutrons"][start:end]
            mcdc["bank_source"]["size"] = N_local

            # =================================================================
            # Set precursor source
            # =================================================================

            # Get source particle size
            N_precursor = f["IC/precursors_size"][()]

            # Redistribute work
            kernel.distribute_work(N_precursor, mcdc, True)  # precursor = True
            N_local = mcdc["mpi_work_size_precursor"]
            start = mcdc["mpi_work_start_precursor"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_precursor"]["precursors"][:N_local] = f["IC/precursors"][
                start:end
            ]
            mcdc["bank_precursor"]["size"] = N_local

            # Set precursor strength
            if N_precursor > 0 and N_particle > 0:
                mcdc["precursor_strength"] = mcdc["bank_precursor"]["precursors"][0][
                    "w"
                ]

    loop.setup_gpu(mcdc)

    # =========================================================================
    # Finalize data: wrapping into a tuple
    # =========================================================================

    return data_tally, mcdc_arr


def cardlist_to_h5group(dictlist, input_group, name):
    if name[-1] != "s":
        main_group = input_group.create_group(name + "s")
    else:
        main_group = input_group.create_group(name)
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % getattr(item, "ID"))
        card_to_h5group(item, group)


def card_to_h5group(card, group):
    for name in [
        a
        for a in dir(card)
        if not a.startswith("__") and not callable(getattr(card, a)) and a != "tag"
    ]:
        value = getattr(card, name)
        if type(value) == dict:
            dict_to_h5group(value, group.create_group(name))
        elif value is None:
            next
        else:
            if name not in ["region"]:
                group[name] = value

            elif name == "region":
                group[name] = str(value)


def dictlist_to_h5group(dictlist, input_group, name):
    if name[-1] != "s":
        main_group = input_group.create_group(name + "s")
    else:
        main_group = input_group.create_group(name)
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % item["ID"])
        dict_to_h5group(item, group)


def dict_to_h5group(dict_, group):
    for k, v in dict_.items():
        if type(v) == dict:
            dict_to_h5group(dict_[k], group.create_group(k))
        elif v is None:
            next
        else:
            group[k] = v


def dd_mergetally(mcdc, data_tally):
    """
    Performs tally recombination on domain-decomposed mesh tallies.
    Gathers and re-organizes tally data into a single array as it
      would appear in a non-decomposed simulation.
    """

    # create bin for recomposed tallies
    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1

    # capture tally lengths for reorganizing later
    xlen = mcdc["technique"]["dd_xlen"]
    ylen = mcdc["technique"]["dd_ylen"]
    zlen = mcdc["technique"]["dd_zlen"]

    # MPI gather
    if (d_Nx * d_Ny * d_Nz) == MPI.COMM_WORLD.Get_size():
        sendcounts = np.array(MPI.COMM_WORLD.gather(len(data_tally[0]), root=0))
        if mcdc["mpi_master"]:
            dd_tally = np.zeros((data_tally.shape[0], sum(sendcounts)))
        else:
            dd_tally = np.empty(data_tally.shape[0])  # dummy tally
        # gather tallies
        for i, t in enumerate(data_tally):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=data_tally[i], recvbuf=(dd_tally[i], sendcounts), root=0
            )
        # gather tally lengths for proper recombination
        xlens = MPI.COMM_WORLD.gather(xlen, root=0)
        ylens = MPI.COMM_WORLD.gather(ylen, root=0)
        zlens = MPI.COMM_WORLD.gather(zlen, root=0)

    # MPI gather for multiprocessor subdomains
    else:
        i = 0
        dd_ranks = []
        # find nonzero tally processor IDs
        for n in range(d_Nx * d_Ny * d_Nz):
            dd_ranks.append(i)
            i += int(mcdc["technique"]["dd_work_ratio"][n])
        # create MPI comm group for nonzero tallies
        dd_group = MPI.COMM_WORLD.group.Incl(dd_ranks)
        dd_comm = MPI.COMM_WORLD.Create(dd_group)
        dd_tally = np.empty(data_tally.shape[0])  # dummy tally

        if MPI.COMM_NULL != dd_comm:
            sendcounts = np.array(dd_comm.gather(len(data_tally[0]), root=0))
            if mcdc["mpi_master"]:
                dd_tally = np.zeros((data_tally.shape[0], sum(sendcounts)))
            # gather tallies
            for i, t in enumerate(data_tally):
                dd_comm.Gatherv(data_tally[i], (dd_tally[i], sendcounts), root=0)
            # gather tally lengths for proper recombination
            xlens = dd_comm.gather(xlen, root=0)
            ylens = dd_comm.gather(ylen, root=0)
            zlens = dd_comm.gather(zlen, root=0)
        dd_group.Free()
        if MPI.COMM_NULL != dd_comm:
            dd_comm.Free()

    if mcdc["mpi_master"]:
        buff = np.zeros_like(dd_tally)
        # reorganize tally data
        # TODO: find/develop a more efficient algorithm for this
        tally_idx = 0
        offset = 0
        ysum = mcdc["technique"]["dd_ysum"]
        zsum = mcdc["technique"]["dd_zsum"]
        for di in range(0, d_Nx * d_Ny * d_Nz):
            dz = di // (d_Nx * d_Ny)
            dy = (di % (d_Nx * d_Ny)) // d_Nx
            dx = di % d_Nx

            offset = 0
            # calculate subdomain offset
            for i in range(0, dx):
                offset += xlens[i] * ysum * zsum

            for i in range(0, dy):
                y_ind = i * d_Nx
                offset += ylens[y_ind] * zsum

            for i in range(0, dz):
                z_ind = i * d_Nx * d_Ny
                offset += zlens[z_ind]

            # calculate index within subdomain
            xlen = xlens[di]
            ylen = ylens[di]
            zlen = zlens[di]
            for xi in range(0, xlen):
                for yi in range(0, ylen):
                    for zi in range(0, zlen):
                        # calculate reorganized index
                        ind_x = xi * ysum * zsum
                        ind_y = yi * zsum
                        ind_z = zi
                        buff_idx = offset + ind_x + ind_y + ind_z
                        # place tally value in correct position
                        buff[:, buff_idx] = dd_tally[:, tally_idx]
                        tally_idx += 1
        # replace old tally with reorganized tally
        dd_tally = buff

    return dd_tally


def dd_mergemesh(mcdc, data_tally):
    """
    Performs mesh recombination on domain-decomposed mesh tallies.
    Gathers and re-organizes mesh data into a single array as it
      would appear in a non-decomposed simulation.
    """
    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1
    # gather mesh filter
    if d_Nx > 1:
        sendcounts = np.array(
            MPI.COMM_WORLD.gather(len(mcdc["mesh_tallies"][0]["filter"]["x"]), root=0)
        )
        if mcdc["mpi_master"]:
            x_filter = np.zeros((mcdc["mesh_tallies"].shape[0], sum(sendcounts)))
        else:
            x_filter = np.empty((mcdc["mesh_tallies"].shape[0]))  # dummy tally
        # gather mesh
        for i in range(mcdc["mesh_tallies"].shape[0]):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=mcdc["mesh_tallies"][i]["filter"]["x"],
                recvbuf=(x_filter[i], sendcounts),
                root=0,
            )
        if mcdc["mpi_master"]:
            x_final = np.zeros((mcdc["mesh_tallies"].shape[0], x_filter.shape[1] + 1))
            x_final[:, 0] = mcdc["mesh_tallies"][:]["filter"]["x"][0][0]
            x_final[:, 1:] = x_filter

    if d_Ny > 1:
        sendcounts = np.array(
            MPI.COMM_WORLD.gather(len(mcdc["mesh_tallies"][0]["filter"]["y"]), root=0)
        )
        if mcdc["mpi_master"]:
            y_filter = np.zeros((mcdc["mesh_tallies"].shape[0], sum(sendcounts)))
        else:
            y_filter = np.empty((mcdc["mesh_tallies"].shape[0]))  # dummy tally
        # gather mesh
        for i in range(mcdc["mesh_tallies"].shape[0]):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=mcdc["mesh_tallies"][i]["filter"]["y"],
                recvbuf=(y_filter[i], sendcounts),
                root=0,
            )
        if mcdc["mpi_master"]:
            y_final = np.zeros((mcdc["mesh_tallies"].shape[0], y_filter.shape[1] + 1))
            y_final[:, 0] = mcdc["mesh_tallies"][:]["filter"]["y"][0][0]
            y_final[:, 1:] = y_filter

    if d_Nz > 1:
        sendcounts = np.array(
            MPI.COMM_WORLD.gather(
                len(mcdc["mesh_tallies"][0]["filter"]["z"]) - 1, root=0
            )
        )
        if mcdc["mpi_master"]:
            z_filter = np.zeros((mcdc["mesh_tallies"].shape[0], sum(sendcounts)))
        else:
            z_filter = np.empty((mcdc["mesh_tallies"].shape[0]))  # dummy tally
        # gather mesh
        for i in range(mcdc["mesh_tallies"].shape[0]):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=mcdc["mesh_tallies"][i]["filter"]["z"][1:],
                recvbuf=(z_filter[i], sendcounts),
                root=0,
            )
        if mcdc["mpi_master"]:
            z_final = np.zeros((mcdc["mesh_tallies"].shape[0], z_filter.shape[1] + 1))
            z_final[:, 0] = mcdc["mesh_tallies"][:]["filter"]["z"][0][0]
            z_final[:, 1:] = z_filter

    dd_mesh = []
    if mcdc["mpi_master"]:
        if d_Nx > 1:
            dd_mesh.append(x_final)
        else:
            dd_mesh.append(mcdc["mesh_tallies"][:]["filter"]["x"])
        if d_Ny > 1:
            dd_mesh.append(y_final)
        else:
            dd_mesh.append(mcdc["mesh_tallies"][:]["filter"]["y"])
        if d_Nz > 1:
            dd_mesh.append(z_final)
        else:
            dd_mesh.append(mcdc["mesh_tallies"][:]["filter"]["z"])
    return dd_mesh


def generate_hdf5(data_tally, mcdc):

    if mcdc["technique"]["domain_decomposition"]:
        dd_tally = dd_mergetally(mcdc, data_tally)
        dd_mesh = dd_mergemesh(mcdc, data_tally)

    if mcdc["mpi_master"]:
        if mcdc["setting"]["progress_bar"]:
            print_msg("")
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "w") as f:
            # Version
            version = importlib.metadata.version("mcdc")
            f["version"] = version

            # Input deck
            if mcdc["setting"]["save_input_deck"]:
                input_group = f.create_group("input_deck")
                cardlist_to_h5group(input_deck.nuclides, input_group, "nuclide")
                cardlist_to_h5group(input_deck.materials, input_group, "material")
                cardlist_to_h5group(input_deck.surfaces, input_group, "surface")
                cardlist_to_h5group(input_deck.cells, input_group, "cell")
                cardlist_to_h5group(input_deck.universes, input_group, "universe")
                cardlist_to_h5group(input_deck.lattices, input_group, "lattice")
                cardlist_to_h5group(input_deck.sources, input_group, "source")
                cardlist_to_h5group(
                    input_deck.mesh_tallies, input_group, "mesh_tallies"
                )
                cardlist_to_h5group(
                    input_deck.surface_tallies, input_group, "surface_tallies"
                )
                cardlist_to_h5group(
                    input_deck.cell_tallies, input_group, "cell_tallies"
                )
                cardlist_to_h5group(input_deck.cs_tallies, input_group, "cs_tallies")
                dict_to_h5group(input_deck.setting, input_group.create_group("setting"))
                dict_to_h5group(
                    input_deck.technique, input_group.create_group("technique")
                )

            # No need to output tally if time census-based tally is used
            if input_deck.setting["census_based_tally"]:
                return

            # Mesh tallies
            for ID, tally in enumerate(mcdc["mesh_tallies"]):
                if mcdc["technique"]["iQMC"]:
                    break

                mesh = tally["filter"]

                # Get grid
                Nx = mesh["Nx"]
                Ny = mesh["Ny"]
                Nz = mesh["Nz"]
                Nt = mesh["Nt"]
                Nmu = mesh["Nmu"]
                N_azi = mesh["N_azi"]
                Ng = mesh["Ng"]
                #
                grid_x = mesh["x"][: Nx + 1]
                grid_y = mesh["y"][: Ny + 1]
                grid_z = mesh["z"][: Nz + 1]
                grid_t = mesh["t"][: Nt + 1]
                grid_mu = mesh["mu"][: Nmu + 1]
                grid_azi = mesh["azi"][: N_azi + 1]
                grid_g = mesh["g"][: Ng + 1]
                # Domain decomposed?
                if mcdc["technique"]["domain_decomposition"]:
                    grid_x = dd_mesh[0][ID]
                    grid_y = dd_mesh[1][ID]
                    grid_z = dd_mesh[2][ID]

                # Save to dataset
                f.create_dataset("tallies/mesh_tally_%i/grid/x" % ID, data=grid_x)
                f.create_dataset("tallies/mesh_tally_%i/grid/y" % ID, data=grid_y)
                f.create_dataset("tallies/mesh_tally_%i/grid/z" % ID, data=grid_z)
                f.create_dataset("tallies/mesh_tally_%i/grid/t" % ID, data=grid_t)
                f.create_dataset("tallies/mesh_tally_%i/grid/mu" % ID, data=grid_mu)
                f.create_dataset("tallies/mesh_tally_%i/grid/azi" % ID, data=grid_azi)
                f.create_dataset("tallies/mesh_tally_%i/grid/g" % ID, data=grid_g)

                # Set tally shape
                N_score = tally["N_score"]
                if mcdc["technique"]["domain_decomposition"]:
                    Nx = mcdc["technique"]["dd_xsum"]
                    Ny = mcdc["technique"]["dd_ysum"]
                    Nz = mcdc["technique"]["dd_zsum"]
                if not mcdc["technique"]["uq"]:
                    shape = (3, Nmu, N_azi, Ng, Nt, Nx, Ny, Nz, N_score)
                else:
                    shape = (5, Nmu, N_azi, Ng, Nt, Nx, Ny, Nz, N_score)

                # Reshape tally
                N_bin = tally["N_bin"]
                if mcdc["technique"]["domain_decomposition"]:
                    # use recomposed N_bin
                    N_bin = 1
                    for elem in shape:
                        N_bin *= elem
                start = tally["stride"]["tally"]
                tally_bin = data_tally[:, start : start + N_bin]
                if mcdc["technique"]["domain_decomposition"]:
                    # substitute recomposed tally
                    tally_bin = dd_tally[:, start : start + N_bin]
                tally_bin = tally_bin.reshape(shape)

                # Roll tally so that score is in the front
                tally_bin = np.rollaxis(tally_bin, 8, 0)

                # Iterate over scores
                for i in range(N_score):
                    score_type = tally["scores"][i]
                    score_tally_bin = np.squeeze(tally_bin[i])
                    if score_type == SCORE_FLUX:
                        score_name = "flux"
                    elif score_type == SCORE_DENSITY:
                        score_name = "density"
                    elif score_type == SCORE_TOTAL:
                        score_name = "total"
                    elif score_type == SCORE_FISSION:
                        score_name = "fission"
                    elif score_type == SCORE_NET_CURRENT:
                        score_name = "current"
                    elif score_type == SCORE_MU_SQ:
                        score_name = "mu-sq"
                    elif score_type == SCORE_TIME_MOMENT_FLUX:
                        score_name = "time-moment-flux"
                    elif score_type == SCORE_SPACE_MOMENT_FLUX:
                        score_name = "space-moment-flux"
                    elif score_type == SCORE_TIME_MOMENT_CURRENT:
                        score_name = "time-moment-current"
                    elif score_type == SCORE_SPACE_MOMENT_CURRENT:
                        score_name = "space-moment-current"
                    elif score_type == SCORE_TIME_MOMENT_MU_SQ:
                        score_name = "time-moment-mu-sq"
                    elif score_type == SCORE_SPACE_MOMENT_MU_SQ:
                        score_name = "space-moment-mu-sq"
                    group_name = "tallies/mesh_tally_%i/%s/" % (ID, score_name)

                    mean = score_tally_bin[TALLY_SUM]
                    sdev = score_tally_bin[TALLY_SUM_SQ]

                    f.create_dataset(group_name + "mean", data=mean)
                    f.create_dataset(group_name + "sdev", data=sdev)
                    if mcdc["technique"]["uq"]:
                        mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                        tot_var = score_tally_bin[TALLY_UQ_BATCH]
                        uq_var = tot_var - mc_var
                        f.create_dataset(group_name + "uq_var", data=uq_var)

            # Surface tallies
            for ID, tally in enumerate(mcdc["surface_tallies"]):
                if mcdc["technique"]["iQMC"]:
                    break

                # Shape
                N_score = tally["N_score"]

                if not mcdc["technique"]["uq"]:
                    shape = (3, N_score)
                else:
                    shape = (5, N_score)

                # Reshape tally
                N_bin = tally["N_bin"]
                start = tally["stride"]["tally"]
                tally_bin = data_tally[:, start : start + N_bin]
                tally_bin = tally_bin.reshape(shape)

                # Roll tally so that score is in the front
                tally_bin = np.rollaxis(tally_bin, 1, 0)

                # Iterate over scores
                for i in range(N_score):
                    score_type = tally["scores"][i]
                    score_tally_bin = np.squeeze(tally_bin[i])
                    score_name = "net-current"
                    group_name = "tallies/surface_tally_%i/%s/" % (ID, score_name)

                    mean = score_tally_bin[TALLY_SUM]
                    sdev = score_tally_bin[TALLY_SUM_SQ]

                    f.create_dataset(group_name + "mean", data=mean)
                    f.create_dataset(group_name + "sdev", data=sdev)
                    if mcdc["technique"]["uq"]:
                        mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                        tot_var = score_tally_bin[TALLY_UQ_BATCH]
                        uq_var = tot_var - mc_var
                        f.create_dataset(group_name + "uq_var", data=uq_var)

            # Cell tallies
            for ID, tally in enumerate(mcdc["cell_tallies"]):
                if mcdc["technique"]["iQMC"]:
                    break

                mesh = tally["filter"]

                # Get grid
                Nt = mesh["Nt"]
                Ng = mesh["Ng"]
                #
                grid_t = mesh["t"][: Nt + 1]
                grid_g = mesh["g"][: Ng + 1]

                # Save to dataset
                f.create_dataset("tallies/cell_tally_%i/grid/t" % ID, data=grid_t)
                f.create_dataset("tallies/cell_tally_%i/grid/g" % ID, data=grid_g)

                # Shape
                N_score = tally["N_score"]

                if not mcdc["technique"]["uq"]:
                    shape = (3, Ng, Nt, N_score)
                else:
                    shape = (5, Ng, Nt, N_score)

                # Reshape tally
                N_bin = tally["N_bin"]
                start = tally["stride"]["tally"]
                tally_bin = data_tally[:, start : start + N_bin]
                tally_bin = tally_bin.reshape(shape)

                # Roll tally so that score is in the front
                tally_bin = np.rollaxis(tally_bin, 3, 0)

                # Iterate over scores
                for i in range(N_score):
                    score_type = tally["scores"][i]
                    score_tally_bin = np.squeeze(tally_bin[i])
                    if score_type == SCORE_FLUX:
                        score_name = "flux"
                    elif score_type == SCORE_NET_CURRENT:
                        score_name = "net-current"
                    elif score_type == SCORE_FISSION:
                        score_name = "fission"
                    group_name = "tallies/cell_tally_%i/%s/" % (ID, score_name)

                    mean = score_tally_bin[TALLY_SUM]
                    sdev = score_tally_bin[TALLY_SUM_SQ]

                    f.create_dataset(group_name + "mean", data=mean)
                    f.create_dataset(group_name + "sdev", data=sdev)
                    if mcdc["technique"]["uq"]:
                        mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                        tot_var = score_tally_bin[TALLY_UQ_BATCH]
                        uq_var = tot_var - mc_var
                        f.create_dataset(group_name + "uq_var", data=uq_var)

            # CS tallies
            for ID, tally in enumerate(mcdc["cs_tallies"]):
                if mcdc["technique"]["iQMC"]:
                    break
                N_cs_bins = tally["filter"]["N_cs_bins"]

                # Shape
                N_score = tally["N_score"]

                if not mcdc["technique"]["uq"]:
                    shape = (3, N_cs_bins, N_score)
                else:
                    shape = (5, N_cs_bins, N_score)

                # Reshape tally
                start = tally["stride"]["tally"]
                tally_bin = data_tally[:, start : start + N_cs_bins]
                tally_bin = tally_bin.reshape(shape)

                # Roll tally so that score is in the front
                tally_bin = np.rollaxis(tally_bin, 2, 0)

                # Iterate over scores
                for i in range(N_score):
                    score_type = tally["scores"][i]
                    score_tally_bin = np.squeeze(tally_bin[i])
                    if score_type == SCORE_FLUX:
                        score_name = "flux"
                    elif score_type == SCORE_NET_CURRENT:
                        score_name = "net-current"
                    elif score_type == SCORE_FISSION:
                        score_name = "fission"
                    group_name = "tallies/cs_tally_%i/%s/" % (ID, score_name)

                    center_points = tally["filter"]["cs_centers"]
                    S = tally["filter"]["cs_S"]
                    reconstruction = tally["filter"]["cs_reconstruction"]

                    f.create_dataset(
                        "tallies/cs_tally_%i/center_points" % (ID), data=center_points
                    )
                    f.create_dataset("tallies/cs_tally_%i/S" % (ID), data=S)
                    f.create_dataset(group_name + "reconstruction", data=reconstruction)

                    mean = score_tally_bin[TALLY_SUM]
                    sdev = score_tally_bin[TALLY_SUM_SQ]

                    f.create_dataset(group_name + "mean", data=mean)
                    f.create_dataset(group_name + "sdev", data=sdev)
                    if mcdc["technique"]["uq"]:
                        mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                        tot_var = score_tally_bin[TALLY_UQ_BATCH]
                        uq_var = tot_var - mc_var
                        f.create_dataset(group_name + "uq_var", data=uq_var)

            # Eigenvalues
            if mcdc["setting"]["mode_eigenvalue"]:
                if mcdc["technique"]["iQMC"]:
                    f.create_dataset("k_eff", data=mcdc["k_eff"])
                    if mcdc["technique"]["iqmc"]["mode"] == "batched":
                        N_cycle = mcdc["setting"]["N_cycle"]
                        f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                        f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                        f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                else:
                    N_cycle = mcdc["setting"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
                    f.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
                    f.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
                    f.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
                    f.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
                    f.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
                    if mcdc["setting"]["gyration_radius"]:
                        f.create_dataset(
                            "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                        )

            # iQMC
            if mcdc["technique"]["iQMC"]:
                # iQMC mesh
                T = mcdc["technique"]
                f.create_dataset("iqmc/grid/t", data=T["iqmc"]["mesh"]["t"])
                f.create_dataset("iqmc/grid/x", data=T["iqmc"]["mesh"]["x"])
                f.create_dataset("iqmc/grid/y", data=T["iqmc"]["mesh"]["y"])
                f.create_dataset("iqmc/grid/z", data=T["iqmc"]["mesh"]["z"])
                # Scores
                for name in [
                    "flux",
                    "source-x",
                    "source-y",
                    "source-z",
                    "fission-power",
                ]:
                    if T["iqmc"]["score_list"][name]:
                        name_h5 = name.replace("-", "_")
                        f.create_dataset(
                            f"iqmc/tally/{name_h5}/mean",
                            data=np.squeeze(T["iqmc"]["score"][name]["mean"]),
                        )
                        f.create_dataset(
                            f"iqmc/tally/{name_h5}/sdev",
                            data=np.squeeze(T["iqmc"]["score"][name]["sdev"]),
                        )
                # iQMC source strength
                f.create_dataset(
                    "iqmc/tally/source_constant/mean",
                    data=np.squeeze(T["iqmc"]["source"]),
                )
                # Iteration data
                f.create_dataset(
                    "iqmc/iteration_count", data=T["iqmc"]["iteration_count"]
                )
                f.create_dataset("iqmc/sweep_count", data=T["iqmc"]["sweep_count"])
                f.create_dataset("iqmc/final_residual", data=T["iqmc"]["residual"])

            # IC generator
            if mcdc["technique"]["IC_generator"]:
                Nn = mcdc["technique"]["IC_bank_neutron"]["size"][0]
                Np = mcdc["technique"]["IC_bank_precursor"]["size"][0]
                f.create_dataset(
                    "IC/neutrons",
                    data=mcdc["technique"]["IC_bank_neutron"]["particles"][:Nn],
                )
                f.create_dataset(
                    "IC/precursors",
                    data=mcdc["technique"]["IC_bank_precursor"]["precursors"][:Np],
                )
                f.create_dataset("IC/neutrons_size", data=Nn)
                f.create_dataset("IC/precursors_size", data=Np)
                f.create_dataset(
                    "IC/fission", data=mcdc["technique"]["IC_fission"] / Nn
                )

    # Save particle?
    if mcdc["setting"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"][0]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Master saves the particle
        if mcdc["mpi_master"]:
            # Remove unwanted particle fields
            neutrons = np.concatenate(neutrons[:])

            # Create dataset
            with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
                f.create_dataset("particles", data=neutrons[:])
                f.create_dataset("particles_size", data=len(neutrons[:]))


def replace_dataset(file_, field, data):
    if field in file_:
        del file_[field]
    file_.create_dataset(field, data=data)


def recombine_tallies(file="output.h5"):
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Load main output file and read input params
        with h5py.File(file, "r") as f:
            output_name = str(f["input_deck/setting/output_name"][()])[2:-1]
            N_particle = f["input_deck/setting/N_particle"][()]
            N_census = f["input_deck/setting/N_census"][()] - 1
            N_batch = f["input_deck/setting/N_batch"][()]
            N_frequency = f["input_deck/setting/census_tally_frequency"][()]
        Nt = N_census * N_frequency
        # Combine the tally output into a single file

        collected_tallies = []
        collected_tally_names = []
        # Collecting info on number and types of tallies
        for i_census in range(N_census):
            for i_batch in range(N_batch):
                with h5py.File(
                    output_name + "-batch_%i-census_%i.h5" % (i_batch, i_census), "r"
                ) as f:
                    tallies = f["tallies"]
                    for tally in tallies:
                        if tally not in collected_tally_names:
                            grid = tallies[tally]["grid"]
                            tally_list = [tally]
                            for tally_type in tallies[tally]:
                                if tally_type != "grid":
                                    tally_list.append(tally_type)
                            collected_tallies.append(tally_list)
                            collected_tally_names.append(tally)

        for i, tally_info in enumerate(collected_tallies):
            print(tally_info)
            tally_type = tally_info[0].split("_")[0]
            tally_number = tally_info[0].split("_")[-1]
            with h5py.File(output_name + ".h5", "a") as f:
                grid = f[
                    "input_deck/"
                    + tally_type
                    + "_tallies/"
                    + tally_type
                    + "_tallies_"
                    + tally_number
                ]
                t_final = f["input_deck/setting/census_time"][()][-2]
                t = np.linspace(0, t_final, N_census * N_frequency + 1)
                Nx = len(grid["x"][()]) - 1
                Ny = len(grid["y"][()]) - 1
                Nz = len(grid["z"][()]) - 1
                Nmu = len(grid["mu"][()]) - 1
                N_azi = len(grid["azi"][()]) - 1
                Ng = len(grid["g"][()]) - 1

                # Creating structure of correct size to hold combined tally
                for tally_type in tally_info[1:]:
                    print("  ", tally_type)
                    tally_score = np.zeros((Nt, Nmu, N_azi, Ng, Nx, Ny, Nz))
                    print("  ", tally_score.shape)
                    tally_score = np.squeeze(tally_score)
                    tally_score_sq = np.zeros_like(tally_score)

                    # Number of shift of time index
                    N_shift = 0
                    if Nmu > 1:
                        N_shift += 1
                    if N_azi > 1:
                        N_shift += 1
                    if Ng > 1:
                        N_shift += 1

                    for i_census in range(N_census):
                        print("    ", i_census)
                        idx_start = i_census * N_frequency
                        idx_end = idx_start + N_frequency
                        for i_batch in range(N_batch):
                            with h5py.File(
                                output_name
                                + "-batch_%i-census_%i.h5" % (i_batch, i_census),
                                "r",
                            ) as f1:
                                score = f1[
                                    "tallies/"
                                    + tally_info[0]
                                    + "/"
                                    + tally_type
                                    + "/score"
                                ][:]
                                if N_shift > 0:
                                    score = np.rollaxis(score, N_shift, 0)
                                tally_score[idx_start:idx_end] += score
                                tally_score_sq[idx_start:idx_end] += score * score
                    tally_score /= N_batch
                    if N_batch > 0:
                        tally_score_sq = np.sqrt(
                            (tally_score_sq / N_batch - np.square(tally_score))
                            / (N_batch - 1)
                        )

                    field_base = "tallies/" + tally_info[0] + "/" + tally_type
                    replace_dataset(f, field_base + "/mean", tally_score)
                    replace_dataset(f, field_base + "/sdev", tally_score_sq)

                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/x", grid["x"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/y", grid["y"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/z", grid["z"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/t", grid["t"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/mu", grid["mu"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/azi", grid["azi"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/g", grid["g"][()]
                )
        """
        for i_census in range(N_census):
            for i_batch in range(N_batch):
                file_name = (
                    output_name
                    + "-batch_"
                    + str(i_batch)
                    + "-census_"
                    + str(i_census)
                    + ".h5"
                )
                os.system("rm " + file_name)
        """


def closeout(mcdc):

    loop.teardown_gpu(mcdc)

    # Runtime
    if mcdc["mpi_master"]:
        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
            for name in [
                "total",
                "preparation",
                "simulation",
                "output",
                "bank_management",
            ]:
                f.create_dataset(
                    "runtime/" + name, data=np.array([mcdc["runtime_" + name]])
                )

        if config.args.runtime_output:
            with h5py.File(mcdc["setting"]["output_name"] + "-runtime.h5", "w") as f:
                for name in [
                    "total",
                    "preparation",
                    "simulation",
                    "output",
                    "bank_management",
                ]:
                    f.create_dataset(name, data=np.array([mcdc["runtime_" + name]]))

    print_runtime(mcdc)
    input_deck.reset()


# ======================================================================================
# Visualize geometry
# ======================================================================================


def visualize(
    vis_type,
    x=0.0,
    y=0.0,
    z=0.0,
    pixels=(100, 100),
    colors=None,
    labels=None,
    legend=False,
    time=np.array([0.0]),
    movie=False,
    save_as=None,
):
    """
    2D visualization of the created model

    Parameters
    ----------
    vis_plane : {'xy', 'yz', 'xz', 'zx', 'yz', 'zy'}
        Axis plane to visualize
    x : float or array_like
        Plane x-position (float) for 'yz' plot. Range of x-axis for 'xy' or 'xz' plot.
    y : float or array_like
        Plane y-position (float) for 'xz' plot. Range of y-axis for 'xy' or 'yz' plot.
    z : float or array_like
        Plane z-position (float) for 'xy' plot. Range of z-axis for 'xz' or 'yz' plot.
    time : array_like
        Times at which the geometry snapshots are taken
    pixels : array_like
        Number of respective pixels in the two axes in vis_plane
    colors : array_like
        List of pairs of material and its color
    labels : array_like
        List of pairs of matieral and its label on plot, sets legend to True
    legend : bool
        Display legend or not, if labels not supplied uses materialIDs
    save_as : str
        If supplied outputs saved as a time stampped .png instead of displayed
    movie : bool
        Save output as a .gif, save_as must be supplied
    """
    # TODO: add input error checkers

    if movie:
        try:
            assert save_as is not None
        except:
            print_error("To save a move supply name of file")

        try:
            import imageio
        except:
            print_error(
                "Imageio required to save visualization results as a git\ntry: pip install imageio"
            )

    _, mcdc_container = prepare()
    mcdc = mcdc_container[0]

    # Color assignment for materials (by material ID)
    if colors is not None:
        new_colors = {}
        for item in colors.items():
            new_colors[item[0].ID] = mpl_colors.to_rgb(item[1])
        colors = new_colors
    else:
        colors = {}
        for i in range(len(mcdc["materials"])):
            colors[i] = plt.cm.Set1(i)[:-1]
    WHITE = mpl_colors.to_rgb("white")

    # legend label assigment (by material ID)
    import matplotlib.patches as mpatches

    legend_hands = [0] * len(mcdc["materials"])
    if labels is not None:
        legend = True
        for item in labels.items():
            legend_hands[item[0].ID] = mpatches.Patch(
                color=colors[item[0].ID], label=item[1]
            )
    else:  # it gives it the mat ID
        for i in range(len(mcdc["materials"])):
            legend_hands[i] = mpatches.Patch(color=colors[i], label="{}".format(i))

    # Set reference axis
    for axis in ["x", "y", "z"]:
        if axis not in vis_type:
            reference_key = axis

    if reference_key == "x":
        reference = x
    elif reference_key == "y":
        reference = y
    elif reference_key == "z":
        reference = z

    # Set first and second axes
    first_key = vis_type[0]
    second_key = vis_type[1]

    if first_key == "x":
        first = x
    elif first_key == "y":
        first = y
    elif first_key == "z":
        first = z

    if second_key == "x":
        second = x
    elif second_key == "y":
        second = y
    elif second_key == "z":
        second = z

    # Axis pixels sizes
    d_first = (first[1] - first[0]) / pixels[0]
    d_second = (second[1] - second[0]) / pixels[1]

    # Axis pixels grids and midpoints
    first_grid = np.linspace(first[0], first[1], pixels[0] + 1)
    first_midpoint = 0.5 * (first_grid[1:] + first_grid[:-1])

    second_grid = np.linspace(second[0], second[1], pixels[1] + 1)
    second_midpoint = 0.5 * (second_grid[1:] + second_grid[:-1])

    # Set dummy particle
    particle_container = adapt.local_array(1, type_.particle)
    particle = particle_container[0]
    particle[reference_key] = reference
    particle["g"] = 0
    particle["E"] = 1e6

    frames = []

    for t in time:
        # Set time
        particle["t"] = t

        # Random direction
        particle["ux"], particle["uy"], particle["uz"] = (
            kernel.sample_isotropic_direction(particle_container)
        )

        # RGB color data for each pixels
        data = np.zeros(pixels + (3,))

        # Loop over the two axes
        for i in range(pixels[0]):
            particle[first_key] = first_midpoint[i]
            for j in range(pixels[1]):
                particle[second_key] = second_midpoint[j]

                # print( particle['x'],  particle['y'],  particle['z'])
                # Get material
                particle["cell_ID"] = -1
                particle["material_ID"] = -1
                if geometry.locate_particle(particle_container, mcdc):
                    data[i, j] = colors[particle["material_ID"]]
                else:
                    data[i, j] = WHITE

        data = np.transpose(data, (1, 0, 2))
        plt.imshow(data, origin="lower", extent=first + second)
        plt.xlabel(first_key + " [cm]")
        plt.ylabel(second_key + " [cm]")
        plt.title(reference_key + " = %.2f cm" % reference + ", time = %.2f s" % t)
        if legend:
            plt.legend(handles=legend_hands, bbox_to_anchor=(1.04, 1), loc="upper left")
        if save_as is not None:
            plt.savefig(save_as + "_%.2f.png" % t)
            plt.clf()
            if movie:
                image = imageio.v2.imread(save_as + "_%.2f.png" % t)
                frames.append(image)
        else:
            plt.show()

    if movie:
        imageio.mimsave(save_as + ".gif", frames, fps=5)
