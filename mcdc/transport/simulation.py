import numpy as np

from mpi4py import MPI
from numba import njit, objmode, uint64

####

import mcdc.code_factory.adapt as adapt
import mcdc.config as config
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.output as output_module
import mcdc.transport.geometry as geometry
import mcdc.transport.mpi as mpi
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.physics as physics
import mcdc.transport.rng as rng
import mcdc.transport.tally as tally_module
import mcdc.transport.technique as technique
import mcdc.transport.sensitivity as sensitivity

from mcdc.constant import *
from mcdc.print_ import (
    print_header_batch,
    print_progress,
    print_progress_eigenvalue,
)
from mcdc.transport.source import source_particle

caching = config.caching


# =============================================================================
# Functions for GPU Interop
# =============================================================================

# The symbols declared below will be overwritten to reference external code that
# manages GPU execution (if GPU execution is supported and selected)
alloc_state, free_state = [None] * 2

src_alloc_program, src_free_program = [None] * 2
(
    src_load_global,
    src_load_constant,
    src_store_global,
    src_store_data,
    src_store_pointer_data,
) = [None] * 5
src_init_program, src_exec_program, src_complete, src_clear_flags = [None] * 4

pre_alloc_program, pre_free_program = [None] * 2
pre_load_global, pre_load_data, pre_store_global, pre_store_data = [None] * 4
pre_init_program, pre_exec_program, pre_complete, pre_clear_flags = [None] * 4


# If GPU execution is supported and selected, the functions shown below will
# be redefined to overwrite the above symbols and perform initialization/
# finalization of GPU state
@njit
def setup_gpu(mcdc, data_tally):
    pass


@njit
def teardown_gpu(mcdc):
    pass


# ======================================================================================
# Fixed-source simulation
# ======================================================================================


def fixed_source_simulation(mcdc_arr, data):
    # Ensure `mcdc` exist for the lifetime of the program by intentionally leaking their memory
    # adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Get some settings
    settings = mcdc["settings"]
    N_batch = settings["N_batch"]
    N_particle = settings["N_particle"]
    N_census = settings["N_census"]
    use_census_based_tally = settings["use_census_based_tally"]

    # Loop over batches
    for idx_batch in range(N_batch):
        mcdc["idx_batch"] = idx_batch
        seed_batch = rng.split_seed(uint64(idx_batch), settings["rng_seed"])

        # Distribute work
        mpi.distribute_work(N_particle, mcdc)

        # Print multi-batch header
        if N_batch > 1:
            with objmode():
                print_header_batch(idx_batch, N_batch)

        # Loop over time censuses
        for idx_census in range(N_census):
            mcdc["idx_census"] = idx_census
            seed_census = rng.split_seed(uint64(seed_batch), rng.SEED_SPLIT_CENSUS)

            # Reset tally time filters if census-based tally is used
            if use_census_based_tally:
                tally_module.filter.set_census_based_time_grid(mcdc, data)

            # Check and accordingly promote future particles to censused particles
            if particle_bank_module.get_bank_size(mcdc["bank_future"]) > 0:
                particle_bank_module.check_future_bank(mcdc, data)

            # Loop over source particles
            seed_source = rng.split_seed(uint64(seed_census), rng.SEED_SPLIT_SOURCE)
            loop_source(uint64(seed_source), mcdc, data)

            # Manage particle banks: population control and work rebalance
            particle_bank_module.manage_particle_banks(mcdc)

            # Time census-based tally closeout
            if use_census_based_tally:
                tally_module.closeout.reduce(mcdc, data)
                tally_module.closeout.accumulate(mcdc, data)
                if mcdc["mpi_master"]:
                    with objmode():
                        output_module.generate_census_based_tally(mcdc, data)
                tally_module.closeout.reset_sum_bins(mcdc, data)

            # Terminate census loop if all banks are empty
            if (
                idx_census > 0
                and particle_bank_module.total_size(mcdc["bank_source"]) == 0
                and particle_bank_module.total_size(mcdc["bank_census"]) == 0
                and particle_bank_module.total_size(mcdc["bank_future"]) == 0
            ):
                break

        # Multi-batch closeout
        if N_batch > 1:
            # Reset banks
            particle_bank_module.set_bank_size(mcdc["bank_active"], 0)
            particle_bank_module.set_bank_size(mcdc["bank_census"], 0)
            particle_bank_module.set_bank_size(mcdc["bank_source"], 0)
            particle_bank_module.set_bank_size(mcdc["bank_future"], 0)

            if not use_census_based_tally:
                # Tally history closeout
                tally_module.closeout.reduce(mcdc, data)
                tally_module.closeout.accumulate(mcdc, data)

    # Tally closeout
    if not use_census_based_tally:
        tally_module.closeout.finalize(mcdc, data)


# =========================================================================
# Eigenvalue simulation
# =========================================================================


def eigenvalue_simulation(mcdc_arr, data):
    # Ensure `mcdc` exist for the lifetime of the program
    # by intentionally leaking their memory
    # adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Get some settings
    settings = mcdc["settings"]
    N_inactive = settings["N_inactive"]
    N_cycle = settings["N_cycle"]
    N_particle = settings["N_particle"]

    # Distribute work
    mpi.distribute_work(N_particle, mcdc)

    # Loop over power iteration cycles
    for idx_cycle in range(N_cycle):
        mcdc["idx_cycle"] = idx_cycle
        seed_cycle = rng.split_seed(uint64(idx_cycle), settings["rng_seed"])

        # Loop over source particles
        loop_source(uint64(seed_cycle), mcdc, data)

        # Tally "history" closeout
        tally_module.closeout.eigenvalue_cycle(mcdc, data)
        if mcdc["cycle_active"]:
            tally_module.closeout.reduce(mcdc, data)
            tally_module.closeout.accumulate(mcdc, data)

        # Manage particle banks: population control and work rebalance
        particle_bank_module.manage_particle_banks(mcdc)

        # Print progress
        with objmode():
            print_progress_eigenvalue(mcdc, data)

        # Entering active cycle?
        mcdc["idx_cycle"] += 1
        if mcdc["idx_cycle"] >= N_inactive:
            mcdc["cycle_active"] = True

    # Tally closeout
    tally_module.closeout.finalize(mcdc, data)
    tally_module.closeout.eigenvalue_simulation(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def loop_source(seed, mcdc, data):
    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]

    for idx_work in range(work_size):
        mcdc["idx_work"] = work_start + idx_work
        generate_source_particle(work_start, idx_work, seed, mcdc, data)

        # Run the source particle and its secondaries
        exhaust_active_bank(mcdc, data)

        source_closeout(mcdc, idx_work, N_prog, data)


@njit
def generate_source_particle(work_start, idx_work, seed, prog, data):
    """Get a source particle and put into one of the banks"""
    mcdc = adapt.mcdc_global(prog)
    settings = mcdc["settings"]

    particle_container = np.zeros(1, type_.particle_data)
    particle = particle_container[0]

    # Get from fixed-source?
    if particle_bank_module.get_bank_size(mcdc["bank_source"]) == 0:
        # Sample source
        seed_work = rng.split_seed(work_start + idx_work, seed)
        source_particle(particle_container, seed_work, mcdc, data)

    # Get from source bank
    else:
        particle_container = mcdc["bank_source"]["particles"][idx_work : (idx_work + 1)]
        particle = particle_container[0]

    # Skip if beyond time boundary
    if particle["t"] > settings["time_boundary"]:
        return

    # Check if it is beyond current or next census times
    hit_census = False
    hit_next_census = False
    idx_census = mcdc["idx_census"]

    if idx_census < settings["N_census"] - 1:
        if particle["t"] > mcdc_get.settings.census_time(
            idx_census + 1, settings, data
        ):
            hit_census = True
            hit_next_census = True
        elif particle["t"] > mcdc_get.settings.census_time(idx_census, settings, data):
            hit_census = True

    # Put into the right bank
    if not hit_census:
        particle_bank_module.add_active(particle_container, prog)
    elif not hit_next_census:
        # Particle will participate after the current census
        particle_bank_module.add_census(particle_container, prog)
    else:
        # Particle will participate in the future
        particle_bank_module.add_future(particle_container, prog)


@njit
def exhaust_active_bank(prog, data):
    mcdc = adapt.mcdc_global(prog)
    particle_container = np.zeros(1, type_.particle)
    particle = particle_container[0]

    # Loop until active bank is exhausted
    while particle_bank_module.get_bank_size(mcdc["bank_active"]) > 0:
        # Get particle from active bank
        particle_bank_module.get_particle(particle_container, mcdc["bank_active"], mcdc)

        prep_particle(particle_container, prog)

        # Particle loop
        loop_particle(particle_container, mcdc, data)


@njit
def prep_particle(particle_container, prog):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)


@njit
def source_closeout(prog, idx_work, N_prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Tally history closeout for one-batch fixed-source simulation
    if not mcdc["settings"]["eigenvalue_mode"] and mcdc["settings"]["N_batch"] == 1:
        if not mcdc["settings"]["use_census_based_tally"]:
            tally_module.closeout.accumulate(mcdc, data)

    # Progress printout
    percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
    if mcdc["settings"]["use_progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, mcdc)


@njit
def loop_source(seed, mcdc, data):
    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]

    for idx_work in range(work_size):
        mcdc["idx_work"] = work_start + idx_work
        generate_source_particle(work_start, idx_work, seed, mcdc, data)

        # Run the source particle and its secondaries
        exhaust_active_bank(mcdc, data)

        source_closeout(mcdc, idx_work, N_prog, data)


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

    shape = eval(f"{adapt.tally_shape_literal}")

    # Just do exec/eval
    def step(prog: nb.uintp, P_input: adapt.particle_gpu):
        mcdc = adapt.mcdc_global(prog)
        data_ptr = adapt.mcdc_data(prog)
        data = adapt.harm.array_from_ptr(data_ptr, shape, nb.float64)
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
    batch_size = 64

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
        if config.gpu_state_storage == "separate":
            adapt.harm.memcpy_host_to_device(mcdc["gpu_meta"]["state_pointer"], mcdc)
            adapt.harm.memcpy_host_to_device(mcdc["gpu_meta"]["state_pointer"], data)

        # Execute the program, and continue to do so until it is done
        if ASYNC_EXECUTION:
            src_exec_program(
                mcdc["gpu_meta"]["source_program_pointer"], BLOCK_COUNT, iter_count
            )
            while not src_complete(mcdc["gpu_meta"]["source_program_pointer"]):
                kernel.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["gpu_meta"]["source_program_pointer"], BLOCK_COUNT, iter_count
                )
        else:
            src_exec_program(
                mcdc["gpu_meta"]["source_program_pointer"], BLOCK_COUNT, batch_size
            )
            while not src_complete(mcdc["gpu_meta"]["source_program_pointer"]):
                kernel.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["gpu_meta"]["source_program_pointer"], BLOCK_COUNT, batch_size
                )
        src_clear_flags(mcdc["gpu_meta"]["source_program_pointer"])
        # Recover the original program state

        if config.gpu_state_storage == "separate":
            adapt.harm.memcpy_device_to_host(mcdc, mcdc["gpu_meta"]["state_pointer"])
            adapt.harm.memcpy_device_to_host(data, mcdc["gpu_meta"]["state_pointer"])

        src_clear_flags(mcdc["gpu_meta"]["source_program_pointer"])

    mcdc["mpi_work_size"] = full_work_size

    kernel.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


# ======================================================================================
# Particle loop
# ======================================================================================


@njit
def loop_particle(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)

    while particle["alive"]:
        step_particle(particle_container, prog, data)


@njit
def step_particle(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)

    # Determine and move to event
    move_to_event(particle_container, mcdc, data)

    # Execute events
    if particle["event"] == EVENT_LOST:
        return

    # Collision
    if particle["event"] & EVENT_COLLISION:
        physics.collision(particle_container, prog, data)

    # Surface and domain crossing
    if particle["event"] & EVENT_SURFACE_CROSSING:
        geometry.surface_crossing(particle_container, prog, data)

    # Census time crossing
    if particle["event"] & EVENT_TIME_CENSUS:
        particle_bank_module.add_census(particle_container, prog)
        particle["alive"] = False

    # Time boundary crossing
    if particle["event"] & EVENT_TIME_BOUNDARY:
        particle["alive"] = False

    # Weight roulette
    if particle["alive"]:
        technique.weight_roulette(particle_container, prog)


@njit
def move_to_event(particle_container, mcdc, data):
    settings = mcdc["settings"]

    # ==================================================================================
    # Preparation (as needed)
    # ==================================================================================
    particle = particle_container[0]

    # Multigroup preparation
    #   In MG mode, particle speed is material-dependent.
    if settings["multigroup_mode"]:
        # If material is not identified yet, locate the particle
        if particle["material_ID"] == -1:
            if not geometry.locate_particle(particle_container, mcdc, data):
                # Particle is lost
                particle["event"] = EVENT_LOST
                return

    # ==================================================================================
    # Geometry inspection
    # ==================================================================================
    #   - Set particle top cell and material IDs (if not lost)
    #   - Set surface ID (if surface hit)
    #   - Set particle boundary event (surface or lattice crossing, or lost)
    #   - Return distance to boundary (surface or lattice)

    d_boundary = geometry.inspect_geometry(particle_container, mcdc, data)

    # Particle is lost?
    if particle["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Get distances to other events
    # ==================================================================================

    # Distance to domain
    speed = physics.particle_speed(particle_container, mcdc, data)

    # Distance to time boundary
    d_time_boundary = speed * (settings["time_boundary"] - particle["t"])

    # Distance to census time
    idx = mcdc["idx_census"]
    d_time_census = speed * (
        mcdc_get.settings.census_time(idx, settings, data) - particle["t"]
    )

    # Distance to next collision
    d_collision = physics.collision_distance(particle_container, mcdc, data)

    # ==================================================================================
    # Determine event(s)
    # ==================================================================================
    # TODO: Make a function to better maintain the repeating operation

    distance = d_boundary

    # Check distance to collision
    if d_collision < distance - COINCIDENCE_TOLERANCE:
        distance = d_collision
        particle["event"] = EVENT_COLLISION
        particle["surface_ID"] = -1
    elif geometry.check_coincidence(d_collision, distance):
        particle["event"] += EVENT_COLLISION

    # Check distance to time census
    if d_time_census < distance - COINCIDENCE_TOLERANCE:
        distance = d_time_census
        particle["event"] = EVENT_TIME_CENSUS
        particle["surface_ID"] = -1
    elif geometry.check_coincidence(d_time_census, distance):
        particle["event"] += EVENT_TIME_CENSUS

    # Check distance to time boundary (exclusive event)
    if d_time_boundary < distance + COINCIDENCE_TOLERANCE:
        distance = d_time_boundary
        particle["event"] = EVENT_TIME_BOUNDARY
        particle["surface_ID"] = -1

    # ==================================================================================
    # Move particle
    # ==================================================================================

    # Score tracklength tallies
    if mcdc["cycle_active"]:
        # Cell tallies
        cell = mcdc["cells"][particle["cell_ID"]]
        for i in range(cell["N_tally"]):
            tally_ID = int(mcdc_get.cell.tally_IDs(i, cell, data))
            tally = mcdc["cell_tallies"][tally_ID]
            tally_module.score.tracklength_tally(
                particle_container, distance, tally, mcdc, data
            )

        # Global tallies
        for i in range(mcdc["N_global_tally"]):
            tally = mcdc["global_tallies"][i]
            tally_module.score.tracklength_tally(
                particle_container, distance, tally, mcdc, data
            )

        # Mesh tallies
        for i in range(mcdc["N_mesh_tally"]):
            tally = mcdc["mesh_tallies"][i]
            tally_module.score.mesh_tally(
                particle_container, distance, tally, mcdc, data
            )

    if settings["eigenvalue_mode"]:
        tally_module.score.eigenvalue_tally(particle_container, distance, mcdc, data)

    if settings["sensitivity_mode"]:
        # Sensitivity response accumulator (track-length)
        sensitivity.score_resp_cum_tracklength(particle_container, distance, mcdc, data)

    # Move particle
    particle_module.move(particle_container, distance, mcdc, data)


# ======================================================================================
# Unsorted [TODO]
# ======================================================================================


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
        particle_container = np.zeros(1, type_.particle)
        particle_container[0] = P_input
        particle = particle_container[0]
        if particle["fresh"]:
            prep_particle(particle_container, prog)
        particle["fresh"] = False
        step_particle(particle_container, data, prog)
        if particle["alive"]:
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
        particle_bank_module.dd_check_in(mcdc)

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
                particle_bank_module.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["source_program_pointer"], BLOCK_COUNT, iter_count
                )
        else:
            src_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, batch_size)
            while not src_complete(mcdc["source_program_pointer"]):
                particle_bank_module.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["source_program_pointer"], BLOCK_COUNT, batch_size
                )

        # Recover the original program state
        src_load_constant(mcdc, mcdc["gpu_state_pointer"])
        src_load_data(data, mcdc["gpu_state_pointer"])
        src_clear_flags(mcdc["source_program_pointer"])

    mcdc["mpi_work_size"] = full_work_size

    particle_bank_module.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


def build_gpu_progs(input_deck, args):

    STRAT = args.gpu_strat

    src_spec = gpu_sources_spec()

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
    global src_load_global, src_store_global, src_load_data, src_store_data, src_store_pointer_data
    global src_init_program, src_exec_program, src_complete, src_clear_flags
    src_alloc_program = src_fns["alloc_program"]
    src_free_program = src_fns["free_program"]
    src_load_global = src_fns["load_state_device_global"]
    src_store_global = src_fns["store_state_device_global"]
    src_store_pointer_global = src_fns["store_pointer_state_device_global"]
    src_load_data = src_fns["load_state_device_data"]
    src_store_data = src_fns["store_state_device_data"]
    src_store_pointer_data = src_fns["store_pointer_state_device_data"]
    src_init_program = src_fns["init_program"]
    src_exec_program = src_fns["exec_program"]
    src_complete = src_fns["complete"]
    src_clear_flags = src_fns["clear_flags"]
    src_set_device = src_fns["set_device"]

    global pre_alloc_program, pre_free_program
    global pre_load_global, pre_store_global, pre_load_data, pre_store_data
    global pre_init_program, pre_exec_program, pre_complete, pre_clear_flags
    pre_alloc_state = pre_fns["alloc_state"]
    pre_free_state = pre_fns["free_state"]
    pre_alloc_program = pre_fns["alloc_program"]
    pre_free_program = pre_fns["free_program"]
    pre_load_global = pre_fns["load_state_device_global"]
    pre_store_global = pre_fns["store_state_device_global"]
    pre_load_data = pre_fns["load_state_device_data"]
    pre_store_data = pre_fns["store_state_device_data"]
    pre_init_program = pre_fns["init_program"]
    pre_exec_program = pre_fns["exec_program"]
    pre_complete = pre_fns["complete"]
    pre_clear_flags = pre_fns["clear_flags"]

    @njit
    def real_setup_gpu(mcdc_array, data_tally):
        mcdc = mcdc_array[0]
        src_set_device(device_id)
        arena_size = ARENA_SIZE
        mcdc["gpu_meta"]["state_pointer"] = adapt.cast_voidptr_to_uintp(alloc_state())
        # src_store_global(mcdc["gpu_meta"]["state_pointer"], mcdc_array[0])
        if config.gpu_state_storage == "separate":
            src_store_pointer_global(
                mcdc["gpu_meta"]["state_pointer"], mcdc["gpu_meta"]["global_pointer"]
            )
            src_store_pointer_data(
                mcdc["gpu_meta"]["state_pointer"], mcdc["gpu_meta"]["tally_pointer"]
            )
        else:
            src_store_pointer_global(mcdc["gpu_meta"]["state_pointer"], mcdc_array)
            src_store_pointer_data(mcdc["gpu_meta"]["state_pointer"], data_tally)

        mcdc["gpu_meta"]["source_program_pointer"] = adapt.cast_voidptr_to_uintp(
            src_alloc_program(mcdc["gpu_meta"]["state_pointer"], ARENA_SIZE)
        )
        src_init_program(mcdc["gpu_meta"]["source_program_pointer"], BLOCK_COUNT)
        return

    @njit
    def real_teardown_gpu(mcdc):
        src_free_program(
            adapt.cast_uintp_to_voidptr(mcdc["gpu_meta"]["source_program_pointer"])
        )
        free_state(adapt.cast_uintp_to_voidptr(mcdc["gpu_meta"]["state_pointer"]))

    global setup_gpu, teardown_gpu
    setup_gpu = real_setup_gpu
    teardown_gpu = real_teardown_gpu

    global loop_source
    loop_source = gpu_loop_source


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
