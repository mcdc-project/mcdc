import numpy as np

from numpy import ascontiguousarray as cga
from numba import njit, objmode

import mcdc.adapt as adapt
import mcdc.iqmc.iqmc_kernel as iqmc_kernel
import mcdc.kernel as kernel
import mcdc.src.geometry as geometry
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import (
    print_error,
    print_iqmc_eigenvalue_exit_code,
    print_iqmc_eigenvalue_progress,
    print_msg,
    print_progress,
    print_progress_iqmc,
)
from mcdc.type_ import iqmc_score_list

# =========================================================================
# Validate inputs
# =========================================================================


def iqmc_validate_inputs(input_deck):
    iqmc = input_deck.technique["iqmc"]
    eigenmode = input_deck.setting["mode_eigenvalue"]

    # Batched mode has only been built for eigenvalue problems (so far)
    if iqmc["mode"] == "batched" and not eigenmode:
        print_error(
            "Invalid run mode. iQMC batched mode has not been built for fixed source problems."
        )

    # Check fixed source solver
    if iqmc["fixed_source_solver"] not in ["source iteration", "gmres"]:
        print_error(
            f"Invalid fixed source solver, '{iqmc['fixed_source_solver']}'. Available iteration solvers inlcude ['source iteration', 'gmres']"
        )

    # Check sample method
    if iqmc["sample_method"] not in ["random", "halton"]:
        print_error(
            f"Unsupported sample method, '{iqmc['sample_method']}'. Available sample methods: ['halton', 'random']."
        )

    # Check run mode
    if iqmc["mode"] not in ["fixed", "batched"]:
        with objmode():
            print_error(
                f"Unsupported run mode, '{iqmc['mode']}'. Available iQMC modes are ['fixed', 'batched']"
            )

    # Check scores
    for score in list(iqmc["score_list"].keys()):
        if score not in iqmc_score_list:
            print_error(
                f"Unsupported score, '{score}'. Available iQMC scores are {iqmc_score_list}"
            )

    # Check N_inactive & N_active batches for batched mode
    if eigenmode and iqmc["mode"] == "batched":
        if (
            input_deck.setting["N_inactive"] == 0
            and input_deck.setting["N_active"] == 0
        ):
            print_error(
                "Specify N_inactive and N_active batches for iQMC batched mode."
            )


# =============================================================================
# iQMC Simulation
# =============================================================================


@njit
def iqmc_simulation(mcdc_arr):
    # Ensure `mcdc` exists for the lifetime of the program
    # by intentionally leaking their memory
    adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Preprocessing
    iqmc = mcdc["technique"]["iqmc"]
    iqmc_kernel.iqmc_preprocess(mcdc)
    iqmc_kernel.samples_init(mcdc)

    if iqmc["mode"] == "batched":
        iqmc["iterations_max"] = (
            mcdc["setting"]["N_active"] + mcdc["setting"]["N_inactive"] - 1
        )

    # Iterative Solve
    if mcdc["setting"]["mode_eigenvalue"]:
        power_iteration(mcdc)
    else:
        if iqmc["fixed_source_solver"] == "source iteration":
            source_iteration(mcdc)
        if iqmc["fixed_source_solver"] == "gmres":
            gmres(mcdc)

    # Post processing
    iqmc_kernel.iqmc_tally_closeout(mcdc)


# =============================================================================
# Iterative Solvers
# =============================================================================


@njit
def source_iteration(mcdc):
    simulation_end = False
    iqmc = mcdc["technique"]["iqmc"]
    total_source_old = iqmc["total_source"].copy()

    while not simulation_end:
        iqmc_sweep(mcdc)
        iqmc["iteration_count"] += 1
        # calculate norm of sources
        iqmc["residual"] = iqmc_kernel.iqmc_res(iqmc["total_source"], total_source_old)
        # iQMC convergence criteria
        if (iqmc["iteration_count"] == iqmc["iterations_max"]) or (
            iqmc["residual"] <= iqmc["tol"]
        ):
            simulation_end = True

        # Print progress
        if not mcdc["setting"]["mode_eigenvalue"]:
            with objmode():
                print_progress_iqmc(mcdc)

        # set  source_old = current source
        total_source_old = iqmc["total_source"].copy()


@njit
def power_iteration(mcdc):
    simulation_end = False
    iqmc = mcdc["technique"]["iqmc"]
    tol = iqmc["tol"]
    maxit = iqmc["iterations_max"]
    score_bin = iqmc["score"]
    k_old = mcdc["k_eff"]
    fission_source_old = score_bin["fission-source"]["bin"].copy()

    while not simulation_end:
        # Scramble samples if in batched mode
        if iqmc["mode"] == "batched":
            iqmc_kernel.scramble_samples(mcdc)
        # Run sweep
        iqmc_sweep(mcdc)
        # Reset counter for inner iteration
        iqmc["iteration_count"] += 1
        # Update k_eff
        mcdc["k_eff"] *= score_bin["fission-source"]["bin"][0] / fission_source_old[0]
        # Calculate diff in keff
        iqmc["residual"] = abs(mcdc["k_eff"] - k_old) / k_old
        k_old = mcdc["k_eff"]
        # Store outter iteration values
        score_bin["effective-fission-outter"] = score_bin["effective-fission"][
            "bin"
        ].copy()
        fission_source_old = score_bin["fission-source"]["bin"].copy()

        # Batch mode
        if iqmc["mode"] == "batched":
            mcdc["idx_cycle"] += 1
            iqmc_kernel.iqmc_eigenvalue_tally_closeout_history(mcdc)
            if mcdc["cycle_active"]:
                # Only accumulate statistics
                iqmc_kernel.iqmc_tally_closeout_history(mcdc)
            # Entering active cycle ?
            if mcdc["idx_cycle"] >= mcdc["setting"]["N_inactive"]:
                mcdc["cycle_active"] = True

        # Print progress
        with objmode():
            if iqmc["mode"] == "fixed":
                print_iqmc_eigenvalue_progress(mcdc)
            else:
                print_iqmc_eigenvalue_progress(mcdc)

        # iQMC convergence criteria
        if (iqmc["iteration_count"] == maxit) or (iqmc["residual"] <= tol):
            simulation_end = True
            if iqmc["mode"] == "fixed":
                with objmode():
                    print_iqmc_eigenvalue_exit_code(mcdc)


@njit
def gmres(mcdc):
    """
    GMRES solver.
    ----------
    Linear Krylov solver. Solves problem of the form Ax = b.
    This function is almost entirely linear algebra operations and does not
    directly use any functions in mcdc/kernel.py or mcdc/loop.py

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
        Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
        http://www-users.cs.umn.edu/~saad/books.html
    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html

    code adapted from: https://github.com/pygbe/pygbe/blob/master/pygbe/gmres.py

    """
    iqmc = mcdc["technique"]["iqmc"]
    max_iter = iqmc["iterations_max"]
    R = iqmc["krylov_restart"]
    tol = iqmc["tol"]

    fixed_source = iqmc["fixed_source"]
    single_vector = iqmc["fixed_source"].size
    b = np.zeros_like(iqmc["total_source"])
    b[:single_vector] = np.reshape(fixed_source, fixed_source.size)
    X = iqmc["total_source"].copy()
    # initial residual
    r = b - AxV(X, b, mcdc)
    normr = np.linalg.norm(r)

    # Defining dimension
    dimen = X.size
    # Set number of outer and inner iterations
    if R > dimen:
        # set number of outter iterations to max allowable (A.shape[0])
        R = dimen
    max_inner = R
    xtype = np.float64

    # max_outer should be max_iter/max_inner but this might not be an integer
    # so we get the ceil of the division.
    # In the inner loop there is a if statement to break in case max_iter is
    # reached.
    max_outer = int(np.ceil(max_iter / max_inner))

    # Check initial guess ( scaling by b, if b != 0, must account for
    # case when norm(b) is very small)
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol * normb:
        return X, 0

    iteration = 0

    # GMRES starts here
    for outer in range(max_outer):
        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        Q = []
        H = np.zeros((max_inner + 1, max_inner + 1), dtype=xtype)
        V = np.zeros((max_inner + 1, dimen), dtype=xtype)

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []
        V[0, :] = (1.0 / normr) * r
        vs.append(V[0, :])

        # Saving initial residual to be used to calculate the rel_resid
        if iteration == 0:
            res_0 = normb

        # RHS vector in the Krylov space
        g = np.zeros((dimen,), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            # New search direction
            v = V[inner + 1, :]
            v[:] = AxV(vs[-1], b, mcdc)
            vs.append(v)

            # Modified Gram Schmidt
            for k in range(inner + 1):
                vk = vs[k]
                alpha = np.dot(vk, v)
                H[inner, k] = alpha
                v[:] = vk * (-alpha) + v[:]

            normv = np.linalg.norm(v)
            H[inner, inner + 1] = normv

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
                v[:] = (1.0 / H[inner, inner + 1]) * v

            # Apply for Givens rotations to H
            if inner > 0:
                for j in range(inner):
                    Qloc = Q[j]
                    H[inner, :][j : j + 2] = np.dot(Qloc, H[inner, :][j : j + 2])

            # Calculate and apply next complex-valued Givens rotations

            # If max_inner = dimen, we don't need to calculate, this
            # is unnecessary for the last inner iteration when inner = dimen -1
            if inner != dimen - 1:
                if H[inner, inner + 1] != 0:
                    # Caclulate matrix rotations
                    c, s, _ = kernel.lartg(H[inner, inner], H[inner, inner + 1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]], dtype=xtype)
                    Q.append(Qblock)

                    # Apply Givens Rotations to RHS for the linear system in
                    # the krylov space.
                    g[inner : inner + 2] = np.dot(Qblock, g[inner : inner + 2])

                    # Apply Givens rotations to H
                    H[inner, inner] = np.dot(Qblock[0, :], H[inner, inner : inner + 2])
                    H[inner, inner + 1] = 0.0

            iteration += 1

            if inner < max_inner - 1:
                normr = abs(g[inner + 1])
                rel_resid = normr / res_0
                iqmc["residual"] = rel_resid

            iqmc["iteration_count"] += 1
            if not mcdc["setting"]["mode_eigenvalue"]:
                with objmode():
                    print_progress_iqmc(mcdc)

            if rel_resid < tol:
                break
            if iqmc["iteration_count"] >= max_iter:
                break

        # end inner loop, back to outer loop
        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y = np.linalg.solve(H[0 : inner + 1, 0 : inner + 1].T, g[0 : inner + 1])
        update = np.ravel(np.dot(cga(V[: inner + 1, :].T), y.reshape(-1, 1)))
        X = X + update
        aux = AxV(X, b, mcdc)
        r = b - aux
        normr = np.linalg.norm(r)
        rel_resid = normr / res_0
        iqmc["residual"] = rel_resid
        if rel_resid < tol:
            break
        if iqmc["iteration_count"] >= max_iter:
            return


# =============================================================================
# Lower Level loops
# =============================================================================


@njit
def iqmc_loop_particle(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]
    while P["alive"]:
        iqmc_step_particle(P_arr, prog)


@njit
def iqmc_step_particle(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]

    # Determine and move to event
    iqmc_kernel.iqmc_move_to_event(P_arr, mcdc)
    event = P["event"]

    # The & operator here is a bitwise and.
    # It is used to determine if an event type is part of the particle event.

    # Surface crossing
    if event & EVENT_SURFACE_CROSSING:
        iqmc_kernel.iqmc_surface_crossing(P_arr, prog)
        if event & EVENT_DOMAIN_CROSSING:
            if not (
                mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_REFLECTIVE
                or mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_VACUUM
            ):
                kernel.domain_crossing(P_arr, mcdc)

    # Lattice or mesh crossing (skipped if surface crossing)
    elif event & EVENT_LATTICE_CROSSING or event & EVENT_IQMC_MESH:
        if event & EVENT_DOMAIN_CROSSING:
            kernel.domain_crossing(P_arr, mcdc)

    # Apply weight roulette
    if P["alive"] and mcdc["technique"]["weight_roulette"]:
        # check if weight has fallen below threshold
        if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
            kernel.weight_roulette(P_arr, mcdc)


@njit
def iqmc_loop_source(mcdc):
    work_size = mcdc["bank_source"]["size"][0]
    N_prog = 0
    # loop over particles
    for idx_work in range(work_size):
        P_arr = mcdc["bank_source"]["particles"][idx_work : (idx_work + 1)]
        P = P_arr[0]
        mcdc["bank_source"]["size"] -= 1
        kernel.add_particle(P_arr, mcdc["bank_active"])

        # Loop until active bank is exhausted
        while mcdc["bank_active"]["size"] > 0:
            P_arr = adapt.local_array(1, type_.particle)
            P = P_arr[0]
            # Get particle from active bank
            kernel.get_particle(P_arr, mcdc["bank_active"], mcdc)
            # Particle loop
            iqmc_loop_particle(P_arr, mcdc)

        # Progress printout
        percent = (idx_work + 1.0) / work_size
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


@njit
def iqmc_sweep(mcdc):
    iqmc = mcdc["technique"]["iqmc"]
    # tally sweep count
    iqmc["sweep_count"] += 1
    # reset particle bank size
    kernel.set_bank_size(mcdc["bank_source"], 0)
    # initialize particles with LDS
    iqmc_kernel.iqmc_prepare_particles(mcdc)
    # reset tallies for next loop
    iqmc_kernel.iqmc_reset_tallies(iqmc)
    # sweep particles
    iqmc_loop_source(mcdc)
    # sum resultant flux on all processors
    iqmc_kernel.iqmc_reduce_tallies(iqmc)
    # update source = scattering + fission/keff + fixed
    iqmc_kernel.iqmc_update_source(mcdc)
    # combine source tallies into one vector
    iqmc_kernel.iqmc_consolidate_sources(mcdc)


# =============================================================================
# GMRES Linear operator
# =============================================================================


@njit
def AxV(V, b, mcdc):
    """
    Linear operator to be used with GMRES.
    Calculate action of A on input vector V, where A is a transport sweep
    and V is the total source (constant and tilted).
    """
    iqmc = mcdc["technique"]["iqmc"]
    iqmc["total_source"] = V.copy()
    # distribute segments of V to appropriate sources
    iqmc_kernel.iqmc_distribute_sources(mcdc)
    iqmc_sweep(mcdc)
    v_out = iqmc["total_source"].copy()
    axv = V - (v_out - b)

    return axv
