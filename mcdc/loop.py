from mpi4py import MPI
from numba import njit, objmode

import shutil

import mcdc.config as config
import mcdc.adapt as adapt
import mcdc.src.geometry as geometry
import mcdc.kernel as kernel
import mcdc.print_ as print_module
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import (
    print_header_batch,
    print_iqmc_eigenvalue_progress,
    print_iqmc_eigenvalue_exit_code,
    print_msg,
    print_progress,
    print_progress_eigenvalue,
    print_progress_iqmc,
)

caching = config.caching


# =============================================================================
# Functions for GPU Interop
# =============================================================================

# The symbols declared below will be overwritten to reference external code that
# manages GPU execution (if GPU execution is supported and selected)
alloc_state, free_state = [None] * 2

src_alloc_program, src_free_program = [None] * 2
src_load_constant, src_load_constant, src_store_constant, src_store_data = [None] * 4
src_init_program, src_exec_program, src_complete, src_clear_flags = [None] * 4

pre_alloc_program, pre_free_program = [None] * 2
pre_load_constant, pre_load_data, pre_store_constant, pre_store_data = [None] * 4
pre_init_program, pre_exec_program, pre_complete, pre_clear_flags = [None] * 4


# If GPU execution is supported and selected, the functions shown below will
# be redefined to overwrite the above symbols and perform initialization/
# finalization of GPU state
@njit
def setup_gpu(mcdc):
    pass


@njit
def teardown_gpu(mcdc):
    pass


# =========================================================================
# Fixed-source loop
# =========================================================================


@njit
def loop_fixed_source(data_tally, mcdc_arr):

    # Ensure `mcdc` exist for the lifetime of the program
    # by intentionally leaking their memory
    adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Loop over batches
    for idx_batch in range(mcdc["setting"]["N_batch"]):
        if not mcdc["technique"]["domain_decomposition"]:
            kernel.distribute_work(N=mcdc["setting"]["N_particle"], mcdc=mcdc)
        else:
            kernel.distribute_work_dd(N=mcdc["setting"]["N_particle"], mcdc=mcdc)
        mcdc["idx_batch"] = idx_batch
        seed_batch = kernel.split_seed(idx_batch, mcdc["setting"]["rng_seed"])

        # Print multi-batch header
        if mcdc["setting"]["N_batch"] > 1:
            with objmode():
                print_header_batch(mcdc)
            if mcdc["technique"]["uq"]:
                seed_uq = kernel.split_seed(seed_batch, SEED_SPLIT_UQ)
                kernel.uq_reset(mcdc, seed_uq)

        # Loop over time censuses
        for idx_census in range(mcdc["setting"]["N_census"]):
            mcdc["idx_census"] = idx_census
            seed_census = kernel.split_seed(seed_batch, SEED_SPLIT_CENSUS)

            # Set census-based tally time grids
            if mcdc["setting"]["census_based_tally"]:
                N_bin = mcdc["setting"]["census_tally_frequency"]
                if idx_census == 0:
                    t_start = 0.0
                else:
                    t_start = mcdc["setting"]["census_time"][idx_census - 1]
                t_end = mcdc["setting"]["census_time"][idx_census]
                dt = (t_end - t_start) / N_bin
                for tally in mcdc["mesh_tallies"]:
                    tally["filter"]["t"][0] = t_start
                    for i in range(N_bin):
                        tally["filter"]["t"][i + 1] = tally["filter"]["t"][i] + dt

            # Check and accordingly promote future particles to censused particle
            if kernel.get_bank_size(mcdc["bank_future"]) > 0:
                kernel.check_future_bank(mcdc)
            if (
                idx_census > 0
                and kernel.get_bank_size(mcdc["bank_source"]) == 0
                and kernel.get_bank_size(mcdc["bank_census"]) == 0
                and kernel.get_bank_size(mcdc["bank_future"]) == 0
            ):
                # No more particle to work on
                break
            # Loop over source particles
            seed_source = kernel.split_seed(seed_census, SEED_SPLIT_SOURCE)
            loop_source(seed_source, data_tally, mcdc)

            # Loop over source precursors
            if kernel.get_bank_size(mcdc["bank_precursor"]) > 0:
                seed_source_precursor = kernel.split_seed(
                    seed_census, SEED_SPLIT_SOURCE_PRECURSOR
                )
                loop_source_precursor(seed_source_precursor, data_tally, mcdc)

            # Manage particle banks: population control and work rebalance
            seed_bank = kernel.split_seed(seed_census, SEED_SPLIT_BANK)
            kernel.manage_particle_banks(seed_bank, mcdc)

            # Time census-based tally closeout
            if mcdc["setting"]["census_based_tally"]:
                kernel.tally_reduce(data_tally, mcdc)
                if mcdc["mpi_master"]:
                    kernel.census_based_tally_output(data_tally, mcdc)
                    if (
                        mcdc["technique"]["weight_window"]
                        and idx_census < mcdc["setting"]["N_census"] - 2
                    ):
                        kernel.update_weight_windows(data_tally, mcdc)
                # TODO: UQ tally

        # Multi-batch closeout
        if mcdc["setting"]["N_batch"] > 1:
            # Reset banks
            kernel.set_bank_size(mcdc["bank_active"], 0)
            kernel.set_bank_size(mcdc["bank_census"], 0)
            kernel.set_bank_size(mcdc["bank_source"], 0)
            kernel.set_bank_size(mcdc["bank_future"], 0)

            # DD closeout
            if mcdc["technique"]["domain_decomposition"]:
                mcdc["dd_N_local_source"] = 0
                mcdc["domain_decomp"]["work_done"] = False

            if not mcdc["setting"]["census_based_tally"]:
                # Tally history closeout
                kernel.tally_reduce(data_tally, mcdc)
                kernel.tally_accumulate(data_tally, mcdc)

                # Uq closeout
                if mcdc["technique"]["uq"]:
                    kernel.uq_tally_closeout_batch(data_tally, mcdc)

    # Tally closeout
    if not mcdc["setting"]["census_based_tally"]:
        if mcdc["technique"]["uq"]:
            kernel.uq_tally_closeout(data_tally, mcdc)
        kernel.tally_closeout(data_tally, mcdc)


# =========================================================================
# Eigenvalue loop
# =========================================================================


@njit
def loop_eigenvalue(data_tally, mcdc_arr):
    # Ensure `mcdc` exist for the lifetime of the program
    # by intentionally leaking their memory
    adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Loop over power iteration cycles
    for idx_cycle in range(mcdc["setting"]["N_cycle"]):
        seed_cycle = kernel.split_seed(idx_cycle, mcdc["setting"]["rng_seed"])

        # Loop over source particles
        seed_source = kernel.split_seed(seed_cycle, SEED_SPLIT_SOURCE)
        loop_source(seed_source, data_tally, mcdc)

        # Tally "history" closeout
        kernel.eigenvalue_tally_closeout_history(mcdc)
        if mcdc["cycle_active"]:
            kernel.tally_reduce(data_tally, mcdc)
            kernel.tally_accumulate(data_tally, mcdc)

        # DD closeout
        if mcdc["technique"]["domain_decomposition"]:
            mcdc["dd_N_local_source"] = 0
            mcdc["domain_decomp"]["work_done"] = False

        # Print progress
        with objmode():
            print_progress_eigenvalue(mcdc)

        # Manage particle banks
        seed_bank = kernel.split_seed(seed_cycle, SEED_SPLIT_BANK)
        kernel.manage_particle_banks(seed_bank, mcdc)

        # Entering active cycle?
        mcdc["idx_cycle"] += 1
        if mcdc["idx_cycle"] >= mcdc["setting"]["N_inactive"]:
            mcdc["cycle_active"] = True

    # Tally closeout
    kernel.tally_closeout(data_tally, mcdc)
    kernel.eigenvalue_tally_closeout(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def generate_source_particle(work_start, idx_work, seed, prog):
    mcdc = adapt.mcdc_global(prog)

    seed_work = kernel.split_seed(work_start + idx_work, seed)

    # =====================================================================
    # Get a source particle and put into active bank
    # =====================================================================

    P_arr = adapt.local_array(1, type_.particle_record)
    P = P_arr[0]

    # Get from fixed-source?
    if kernel.get_bank_size(mcdc["bank_source"]) == 0:
        # Sample source
        kernel.source_particle(P_arr, seed_work, mcdc)

    # Get from source bank
    else:
        P_arr = mcdc["bank_source"]["particles"][idx_work : (idx_work + 1)]
        P = P_arr[0]

    # Skip if beyond time boundary
    if P["t"] > mcdc["setting"]["time_boundary"]:
        return

    # If domain is decomposed, check if particle is in the domain
    if mcdc["technique"]["domain_decomposition"]:
        if not kernel.particle_in_domain(P_arr, mcdc):
            return

        # Also check if it belongs to the current rank
        mcdc["dd_N_local_source"] += 1
        if mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]] > 1:
            if (
                mcdc["dd_N_local_source"]
                % mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]]
                != mcdc["dd_local_rank"]
            ):
                return

    # Check if it is beyond current or next census times
    hit_census = False
    hit_next_census = False
    idx_census = mcdc["idx_census"]
    if idx_census < mcdc["setting"]["N_census"] - 1:
        if P["t"] > mcdc["setting"]["census_time"][idx_census + 1]:
            hit_census = True
            hit_next_census = True
        elif P["t"] > mcdc["setting"]["census_time"][idx_census]:
            hit_census = True

    # Put into the right bank
    if not hit_census:
        adapt.add_active(P_arr, prog)
    elif not hit_next_census:
        # Particle will participate after the current census
        adapt.add_census(P_arr, prog)
    else:
        # Particle will participate in the future
        adapt.add_future(P_arr, prog)

    """
        if mcdc["technique"]["domain_decomposition"]:
            if mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]] > 0:
                P["w"] /= mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]]
            if kernel.particle_in_domain(P_arr, mcdc):
                adapt.add_census(P_arr, prog)
        else:
            adapt.add_census(P_arr, prog)
    else:
        P_new_arr = adapt.local_array(1, type_.particle)
        P_new = P_new_arr[0]
        # Add the source particle into the active bank
        if mcdc["technique"]["domain_decomposition"]:
            if mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]] > 0:
                P["w"] /= mcdc["technique"]["dd_work_ratio"][mcdc["dd_idx"]]
            if kernel.particle_in_domain(P_arr, mcdc):
                kernel.recordlike_to_particle(P_new_arr, P_arr)
                adapt.add_active(P_new_arr, prog)
        else:
            kernel.recordlike_to_particle(P_new_arr, P_arr)
            adapt.add_active(P_new_arr, prog)
    """


@njit
def prep_particle(P_arr, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Apply weight window
    if mcdc["technique"]["weight_window"]:
        kernel.weight_window(P_arr, prog)


@njit
def exhaust_active_bank(data_tally, prog):
    mcdc = adapt.mcdc_global(prog)
    P_arr = adapt.local_array(1, type_.particle)
    P = P_arr[0]

    # Loop until active bank is exhausted
    while kernel.get_bank_size(mcdc["bank_active"]) > 0:
        # Get particle from active bank
        kernel.get_particle(P_arr, mcdc["bank_active"], mcdc)

        prep_particle(P_arr, prog)

        # Particle loop
        loop_particle(P_arr, data_tally, mcdc)


@njit
def source_closeout(prog, idx_work, N_prog, data_tally):
    mcdc = adapt.mcdc_global(prog)

    # Tally history closeout for one-batch fixed-source simulation
    if not mcdc["setting"]["mode_eigenvalue"] and mcdc["setting"]["N_batch"] == 1:
        if not mcdc["setting"]["census_based_tally"]:
            kernel.tally_accumulate(data_tally, mcdc)

    # Tally history closeout for multi-batch uq simulation
    if mcdc["technique"]["uq"]:
        kernel.uq_tally_closeout_history(data_tally, mcdc)

    # Progress printout
    percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
    if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, mcdc)


@njit
def source_dd_resolution(data, prog):
    mcdc = adapt.mcdc_global(prog)

    kernel.dd_particle_send(mcdc)
    terminated = False
    max_work = 1
    kernel.dd_recv(mcdc)
    if mcdc["domain_decomp"]["work_done"]:
        terminated = True

    P_arr = adapt.local_array(1, type_.particle)
    P = P_arr[0]

    while not terminated:
        if kernel.get_bank_size(mcdc["bank_active"]) > 0:
            # Loop until active bank is exhausted
            while kernel.get_bank_size(mcdc["bank_active"]) > 0:

                kernel.get_particle(P_arr, mcdc["bank_active"], mcdc)
                if not kernel.particle_in_domain(P_arr, mcdc) and P["alive"] == True:
                    print(f"recieved particle not in domain")

                # Apply weight window
                if mcdc["technique"]["weight_window"]:
                    kernel.weight_window(P_arr, mcdc)

                # Particle loop
                loop_particle(P_arr, data, mcdc)

                # Tally history closeout for one-batch fixed-source simulation
                if (
                    not mcdc["setting"]["mode_eigenvalue"]
                    and mcdc["setting"]["N_batch"] == 1
                ):
                    kernel.tally_accumulate(data, mcdc)

        # Send all domain particle banks
        kernel.dd_particle_send(mcdc)

        kernel.dd_recv(mcdc)

        # Progress printout
        """
        percent = 1 - work_remaining / max_work
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)
        """
        if kernel.dd_check_halt(mcdc):
            kernel.dd_check_out(mcdc)
            terminated = True


@njit
def loop_source(seed, data_tally, mcdc):
    # Progress bar indicator
    N_prog = 0

    if mcdc["technique"]["domain_decomposition"]:
        kernel.dd_check_in(mcdc)

    # Loop over particle sources
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]
    work_end = work_start + work_size

    for idx_work in range(work_size):
        generate_source_particle(work_start, idx_work, seed, mcdc)

        # Run the source particle and its secondaries
        exhaust_active_bank(data_tally, mcdc)

        source_closeout(mcdc, idx_work, N_prog, data_tally)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data_tally, mcdc)


def gpu_sources_spec():
    def make_work(prog: nb.uintp) -> nb.boolean:
        mcdc = adapt.mcdc_global(prog)

        idx_work = adapt.global_add(mcdc["mpi_work_iter"], 0, 1)

        if idx_work >= mcdc["mpi_work_size"]:
            return False

        generate_source_particle(
            mcdc["mpi_work_start"], nb.uint64(idx_work), mcdc["source_seed"], prog
        )
        return True

    def initialize(prog: nb.uintp):
        pass

    def finalize(prog: nb.uintp):
        pass

    base_fns = (initialize, finalize, make_work)

    def step(prog: nb.uintp, P_input: adapt.particle_gpu):
        mcdc = adapt.mcdc_global(prog)
        data = adapt.mcdc_data(prog)
        P_arr = adapt.local_array(1, type_.particle)
        P_arr[0] = P_input
        P = P_arr[0]
        if P["fresh"]:
            prep_particle(P_arr, prog)
        P["fresh"] = False
        step_particle(P_arr, data, prog)
        if P["alive"]:
            adapt.step_async(prog, P)

    async_fns = [step]
    return adapt.harm.RuntimeSpec("mcdc_source", adapt.state_spec, base_fns, async_fns)


BLOCK_COUNT = config.args.gpu_block_count

ASYNC_EXECUTION = config.args.gpu_strat == "async"


@njit(cache=caching)
def gpu_loop_source(seed, data, mcdc):

    # Progress bar indicator
    N_prog = 0

    if mcdc["technique"]["domain_decomposition"]:
        kernel.dd_check_in(mcdc)

    # =====================================================================
    # GPU Interop
    # =====================================================================

    # For async execution
    iter_count = 655360000
    # For event-based execution
    batch_size = 1

    full_work_size = mcdc["mpi_work_size"]
    if ASYNC_EXECUTION:
        phase_size = 1000000000
    else:
        phase_size = 1000000
    phase_count = (full_work_size + phase_size - 1) // phase_size

    for phase in range(phase_count):

        mcdc["mpi_work_iter"][0] = phase_size * phase
        mcdc["mpi_work_size"] = min(phase_size * (phase + 1), full_work_size)
        mcdc["source_seed"] = seed

        # Store the global state to the GPU
        src_store_constant(mcdc["gpu_state_pointer"], mcdc)
        src_store_data(mcdc["gpu_state_pointer"], data)

        # Execute the program, and continue to do so until it is done
        if ASYNC_EXECUTION:
            src_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, iter_count)
            while not src_complete(mcdc["source_program_pointer"]):
                kernel.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["source_program_pointer"], BLOCK_COUNT, iter_count
                )
        else:
            src_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, batch_size)
            while not src_complete(mcdc["source_program_pointer"]):
                kernel.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["source_program_pointer"], BLOCK_COUNT, batch_size
                )

        # Recover the original program state
        src_load_constant(mcdc, mcdc["gpu_state_pointer"])
        src_load_data(data, mcdc["gpu_state_pointer"])
        src_clear_flags(mcdc["source_program_pointer"])

    mcdc["mpi_work_size"] = full_work_size

    kernel.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


# =========================================================================
# Particle loop
# =========================================================================


@njit
def loop_particle(P_arr, data_tally, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)
    while P["alive"]:
        step_particle(P_arr, data_tally, prog)


@njit
def step_particle(P_arr, data_tally, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Determine and move to event
    kernel.move_to_event(P_arr, data_tally, mcdc)

    # Execute events
    if P["event"] == EVENT_LOST:
        P["alive"] = False
        return

    # Collision
    if P["event"] & EVENT_COLLISION:
        # Generate IC?
        if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
            kernel.bank_IC(P_arr, prog)

        # Branchless collision?
        if mcdc["technique"]["branchless_collision"]:
            kernel.branchless_collision(P_arr, prog)

        # Analog collision
        else:
            # Get collision type
            kernel.collision(P_arr, mcdc)

            # Perform collision
            if P["event"] & EVENT_CAPTURE:
                P["alive"] = False

            elif P["event"] & EVENT_SCATTERING:
                kernel.scattering(P_arr, prog)

            elif P["event"] & EVENT_FISSION:
                kernel.fission(P_arr, prog)

    if P["event"] & EVENT_PHANTOM_COLLISION:
        # delta tracking
        P["alive"] = True

    # Surface and domain crossing
    if P["event"] & EVENT_SURFACE_CROSSING:
        kernel.surface_crossing(P_arr, data_tally, prog)
        if P["event"] & EVENT_DOMAIN_CROSSING:
            if mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_NONE:
                kernel.domain_crossing(P_arr, prog)

    elif P["event"] & EVENT_DOMAIN_CROSSING:
        kernel.domain_crossing(P_arr, prog)

    # Census time crossing
    if P["event"] & EVENT_TIME_CENSUS:
        adapt.add_census(P_arr, prog)
        P["alive"] = False

    # Time boundary crossing
    if P["event"] & EVENT_TIME_BOUNDARY:
        P["alive"] = False

    # Apply weight window
    if P["alive"] and mcdc["technique"]["weight_window"]:
        kernel.weight_window(P_arr, prog)

    # Apply weight roulette
    if P["alive"] and mcdc["technique"]["weight_roulette"]:
        # check if weight has fallen below threshold
        if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
            kernel.weight_roulette(P_arr, mcdc)


# =============================================================================
# Precursor source loop
# =============================================================================


@njit
def generate_precursor_particle(DNP_arr, particle_idx, seed, prog):
    mcdc = adapt.mcdc_global(prog)
    DNP = DNP_arr[0]

    # Set groups
    j = DNP["g"]
    g = DNP["n_g"]

    # Create new particle
    P_new_arr = adapt.local_array(1, type_.particle)
    P_new = P_new_arr[0]
    part_seed = kernel.split_seed(particle_idx, seed)
    P_new["rng_seed"] = part_seed
    P_new["alive"] = True
    P_new["w"] = 1.0

    # Set position
    P_new["x"] = DNP["x"]
    P_new["y"] = DNP["y"]
    P_new["z"] = DNP["z"]
    P_new["t"] = 0.0

    # Sample direction
    P_new["ux"], P_new["uy"], P_new["uz"] = kernel.sample_isotropic_direction(P_new_arr)

    # Get cell and material
    P_new["cell_ID"] = -1
    P_new["material_ID"] = -1
    _ = geometry.locate_particle(P_new_arr, mcdc)

    # Skip if particle is lost
    if P_new["material_ID"] == -1:
        return

    material_ID = P_new["material_ID"]
    material = mcdc["materials"][material_ID]
    G = material["G"]

    # Sample nuclide and get spectrum and decay constant
    N_nuclide = material["N_nuclide"]
    if N_nuclide == 1:
        nuclide = mcdc["nuclides"][material["nuclide_IDs"][0]]
        spectrum = nuclide["chi_d"][j]
        decay = nuclide["decay"][j]
    else:
        SigmaF = material["fission"][g]  # MG only
        nu_d = material["nu_d"][g]
        xi = kernel.rng(P_new_arr) * nu_d[j] * SigmaF
        tot = 0.0
        for i in range(N_nuclide):
            nuclide = mcdc["nuclides"][material["nuclide_IDs"][i]]
            density = material["nuclide_densities"][i]
            tot += density * nuclide["nu_d"][g, j] * nuclide["fission"][g]
            if xi < tot:
                # Nuclide determined, now get the constant and spectruum
                spectrum = nuclide["chi_d"][j]
                decay = nuclide["decay"][j]
                break

    # Sample emission time
    P_new["t"] = -math.log(kernel.rng(P_new_arr)) / decay
    idx_census = mcdc["idx_census"]
    if idx_census > 0:
        P_new["t"] += mcdc["setting"]["census_time"][idx_census - 1]

    # Accept if it is inside current census index
    if P_new["t"] < mcdc["setting"]["census_time"][idx_census]:
        # Skip if it's beyond time boundary
        if P_new["t"] > mcdc["setting"]["time_boundary"]:
            return

        # Sample energy
        xi = kernel.rng(P_new_arr)
        tot = 0.0
        for g_out in range(G):
            tot += spectrum[g_out]
            if tot > xi:
                break
        P_new["g"] = g_out

        # Push to active bank
        adapt.add_active(P_new_arr, prog)


@njit
def source_precursor_closeout(prog, idx_work, N_prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Tally history closeout for fixed-source simulation
    if not mcdc["setting"]["mode_eigenvalue"]:
        kernel.tally_accumulate(data, mcdc)

    # Progress printout
    percent = (idx_work + 1.0) / mcdc["mpi_work_size_precursor"]
    if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, mcdc)


@njit
def loop_source_precursor(seed, data, mcdc):
    # Progress bar indicator
    N_prog = 0

    # TODO: Domain decomposition (see loop_source)

    # Loop over precursor sources
    work_start = mcdc["mpi_work_start_precursor"]
    work_size = mcdc["mpi_work_size_precursor"]
    work_end = work_start + work_size

    for idx_work in range(work_size):
        # Get precursor
        DNP_arr = mcdc["bank_precursor"]["precursors"][idx_work : (idx_work + 1)]
        DNP = DNP_arr[0]
        seed_precursor = kernel.split_seed(work_start + idx_work, seed)
        # Note the seed is only used once

        # Determine number of particles to be generated
        nu = DNP["w"]
        xi = kernel.rng_from_seed(seed_precursor)
        N = int(math.floor(nu + xi))

        # Loop over source particles from the source precursor
        for particle_idx in range(N):
            generate_precursor_particle(DNP_arr, particle_idx, seed_precursor, mcdc)
            exhaust_active_bank(data, mcdc)

        source_precursor_closeout(mcdc, idx_work, N_prog, data)


def gpu_precursor_spec():
    def make_work(prog: nb.uintp) -> nb.boolean:
        mcdc = adapt.mcdc_global(prog)

        idx_work = adapt.global_add(mcdc["mpi_work_iter"], 0, 1)

        if idx_work >= mcdc["mpi_work_size_precursor"]:
            return False

        seed = mcdc["source_seed"]

        # Get precursor
        DNP_arr = mcdc["bank_precursor"]["precursors"][idx_work : (idx_work + 1)]
        DNP = DNP_arr[0]

        # Determine number of particles to be generated
        w = DNP["w"]
        N = math.floor(w)
        # "Roulette" the last particle
        seed_work = kernel.split_seed(idx_work, seed)
        if kernel.rng_from_seed(seed_work) < w - N:
            N += 1
        DNP["w"] = N

        # =====================================================================
        # Loop over source particles from the source precursor
        # =====================================================================

        for particle_idx in range(N):
            generate_precursor_particle(DNP_arr, particle_idx, seed_work, prog)

        return True

    def initialize(prog: nb.uintp):
        pass

    def finalize(prog: nb.uintp):
        pass

    base_fns = (initialize, finalize, make_work)

    def step(prog: nb.uintp, P_input: adapt.particle_gpu):
        mcdc = adapt.mcdc_global(prog)
        data = adapt.mcdc_data(prog)
        P_arr = adapt.local_array(1, type_.particle)
        P_arr[0] = P_input
        P = P_arr[0]
        if P["fresh"]:
            prep_particle(P_arr, prog)
        P["fresh"] = False
        step_particle(P_arr, data, prog)
        if P["alive"]:
            adapt.step_async(prog, P)

    async_fns = [step]
    return adapt.harm.RuntimeSpec(
        "mcdc_precursor", adapt.state_spec, base_fns, async_fns
    )


@njit(cache=caching)
def gpu_loop_source_precursor(seed, data, mcdc):

    # Progress bar indicator
    N_prog = 0

    # =========================================================================
    # Sync. RNG skip ahead for reproducibility
    # =========================================================================

    # Exscan upper estimate of number of particles generated locally
    idx_start, N_local, N_global = kernel.bank_scanning_DNP(
        mcdc["bank_precursor"], mcdc
    )

    # =====================================================================
    # GPU Interop
    # =====================================================================

    # Number of blocks to launch and number of iterations to run

    # For async execution
    iter_count = 655360000
    # For event-based execution
    batch_size = 1

    mcdc["mpi_work_iter"][0] = 0
    mcdc["source_seed"] = seed

    # Store the global state to the GPU
    pre_store_constant(mcdc["gpu_state_pointer"], mcdc)
    pre_store_data(mcdc["gpu_state_pointer"], data)

    # Execute the program, and continue to do so until it is done

    # Execute the program, and continue to do so until it is done
    if ASYNC_EXECUTION:
        pre_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, iter_count)
        while not pre_complete(mcdc["source_program_pointer"]):
            kernel.dd_particle_send(mcdc)
            pre_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, iter_count)
    else:
        pre_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, batch_size)
        while not pre_complete(mcdc["source_program_pointer"]):
            kernel.dd_particle_send(mcdc)
            pre_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, batch_size)
    # Recover the original program state
    pre_load_constant(mcdc, mcdc["gpu_state_pointer"])
    pre_load_data(data, mcdc["gpu_state_pointer"])
    pre_clear_flags(mcdc["source_program_pointer"])

    kernel.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (moved out of loop)
    # =====================================================================

    source_precursor_closeout(mcdc, 1, 1, data)


def build_gpu_progs(input_deck, args):

    STRAT = args.gpu_strat

    src_spec = gpu_sources_spec()
    pre_spec = gpu_precursor_spec()

    adapt.harm.RuntimeSpec.bind_specs()

    rank = MPI.COMM_WORLD.Get_rank()
    device_id = rank % args.gpu_share_stride

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()

    adapt.harm.RuntimeSpec.load_specs()

    if STRAT == "async":
        args.gpu_arena_size = args.gpu_arena_size // 32
        src_fns = src_spec.async_functions()
        pre_fns = pre_spec.async_functions()
    else:
        src_fns = src_spec.event_functions()
        pre_fns = pre_spec.event_functions()

    ARENA_SIZE = args.gpu_arena_size
    BLOCK_COUNT = args.gpu_block_count

    global alloc_state, free_state
    alloc_state = src_fns["alloc_state"]
    free_state = src_fns["free_state"]

    global src_alloc_program, src_free_program
    global src_load_constant, src_store_constant, src_load_data, src_store_data
    global src_init_program, src_exec_program, src_complete, src_clear_flags
    src_alloc_program = src_fns["alloc_program"]
    src_free_program = src_fns["free_program"]
    src_load_constant = src_fns["load_state_device_global"]
    src_store_constant = src_fns["store_state_device_global"]
    src_load_data = src_fns["load_state_device_data"]
    src_store_data = src_fns["store_state_device_data"]
    src_init_program = src_fns["init_program"]
    src_exec_program = src_fns["exec_program"]
    src_complete = src_fns["complete"]
    src_clear_flags = src_fns["clear_flags"]
    src_set_device = src_fns["set_device"]

    global pre_alloc_program, pre_free_program
    global pre_load_constant, pre_store_constant, pre_load_data, pre_store_data
    global pre_init_program, pre_exec_program, pre_complete, pre_clear_flags
    pre_alloc_state = pre_fns["alloc_state"]
    pre_free_state = pre_fns["free_state"]
    pre_alloc_program = pre_fns["alloc_program"]
    pre_free_program = pre_fns["free_program"]
    pre_load_constant = pre_fns["load_state_device_global"]
    pre_store_constant = pre_fns["store_state_device_global"]
    pre_load_data = pre_fns["load_state_device_data"]
    pre_store_data = pre_fns["store_state_device_data"]
    pre_init_program = pre_fns["init_program"]
    pre_exec_program = pre_fns["exec_program"]
    pre_complete = pre_fns["complete"]
    pre_clear_flags = pre_fns["clear_flags"]

    @njit
    def real_setup_gpu(mcdc):
        src_set_device(device_id)
        arena_size = ARENA_SIZE
        mcdc["gpu_state_pointer"] = adapt.cast_voidptr_to_uintp(alloc_state())
        mcdc["source_program_pointer"] = adapt.cast_voidptr_to_uintp(
            src_alloc_program(mcdc["gpu_state_pointer"], ARENA_SIZE)
        )
        src_init_program(mcdc["source_program_pointer"], BLOCK_COUNT)
        mcdc["precursor_program_pointer"] = adapt.cast_voidptr_to_uintp(
            pre_alloc_program(mcdc["gpu_state_pointer"], ARENA_SIZE)
        )
        pre_init_program(mcdc["precursor_program_pointer"], BLOCK_COUNT)
        return

    @njit
    def real_teardown_gpu(mcdc):
        src_free_program(adapt.cast_uintp_to_voidptr(mcdc["source_program_pointer"]))
        pre_free_program(adapt.cast_uintp_to_voidptr(mcdc["precursor_program_pointer"]))
        free_state(adapt.cast_uintp_to_voidptr(mcdc["gpu_state_pointer"]))

    global setup_gpu, teardown_gpu
    setup_gpu = real_setup_gpu
    teardown_gpu = real_teardown_gpu

    global loop_source, loop_source_precursor
    loop_source = gpu_loop_source
    loop_source_precursor = gpu_loop_source_precursor
