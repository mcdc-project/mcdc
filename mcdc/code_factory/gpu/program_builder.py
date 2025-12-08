import numba as nb
import numpy as np

from mpi4py import MPI

####

import mcdc.config as config


def build_gpu_program(data):
    import harmonize
    import mcdc.numba_types as type_

    # Compilation check
    if MPI.COMM_WORLD.Get_rank() != 0:
        harmonize.config.should_compile(harmonize.config.ShouldCompile.NEVER)
    elif config.caching == False:
        harmonize.config.should_compile(harmonize.config.ShouldCompile.ALWAYS)

    # ==================================================================================
    # Forward declaration
    # ==================================================================================

    # ROCm and CUDA paths
    if config.args.gpu_cuda_path != None:
        harmonize.config.set_cuda_path(config.args.gpu_cuda_path)
    if config.args.gpu_rocm_path != None:
        harmonize.config.set_rocm_path(config.args.gpu_rocm_path)

    # Main types: none, simulation structure, and simulation data
    none_type = nb.from_dtype(np.dtype([]))
    simulation_type = nb.types.Array(nb.from_dtype(type_.simulation), (1,), "C")
    data_type = nb.types.Array(nb.float64, 1, "C")

    # Set access functions
    state_spec = (
        {
            "global": simulation_type,
            "data": data_type,
        },
        none_type,
        none_type,
    )
    access_fns = harmonize.RuntimeSpec.access_fns(state_spec)
    simulation_gpu = access_fns["device"]["global"]["indirect"]
    data_gpu = access_fns["device"]["data"]["direct"]
    group_gpu = access_fns["group"]
    thread_gpu = access_fns["thread"]
    particle_gpu = nb.from_dtype(type_.particle)
    particle_record_gpu = nb.from_dtype(type_.particle_data)

    # Functions, and their signatures
    def step(prog: nb.uintp, P: particle_gpu):
        pass

    def find_cell(prog: nb.uintp, P: particle_gpu):
        pass

    # Asynchronous versions
    step_async, find_cell_async = harmonize.RuntimeSpec.async_dispatch(step, find_cell)

    # Program interfaces
    interface = harmonize.RuntimeSpec.program_interface()
    halt_early = interface["halt_early"]

    # ==================================================================================
    # Build the program
    # ==================================================================================

    src_spec = gpu_sources_spec()

    harmonize.RuntimeSpec.bind_specs()

    rank = MPI.COMM_WORLD.Get_rank()
    device_id = rank % config.args.gpu_share_stride

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()

    harmonize.RuntimeSpec.load_specs()

    strategy = config.args.gpu_strat
    if strategy == "async":
        config.args.gpu_arena_size = config.args.gpu_arena_size // 32
        src_fns = src_spec.async_functions()
    else:
        src_fns = src_spec.event_functions()

    arena_size = config.args.gpu_arena_size
    block_count = config.args.gpu_block_count

    alloc_state = src_fns["alloc_state"]
    free_state = src_fns["free_state"]

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

    global loop_source
    loop_source = gpu_loop_source

    # Overwrite function
    for impl in target_rosters["cpu"].values():
        overwrite_func(impl, impl)

    # ==================================================================================
    # Setup
    # ==================================================================================

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


# ======================================================================================
# Teardown
# ======================================================================================


def teardown_gpu(mcdc):
    src_free_program(
        adapt.cast_uintp_to_voidptr(mcdc["gpu_meta"]["source_program_pointer"])
    )
    free_state(adapt.cast_uintp_to_voidptr(mcdc["gpu_meta"]["state_pointer"]))


# ======================================================================================
# Source loop
# ======================================================================================


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
            harmonize.memcpy_host_to_device(mcdc["gpu_meta"]["state_pointer"], mcdc)
            harmonize.memcpy_host_to_device(mcdc["gpu_meta"]["state_pointer"], data)

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
            harmonize.memcpy_device_to_host(mcdc, mcdc["gpu_meta"]["state_pointer"])
            harmonize.memcpy_device_to_host(data, mcdc["gpu_meta"]["state_pointer"])

        src_clear_flags(mcdc["gpu_meta"]["source_program_pointer"])

    mcdc["mpi_work_size"] = full_work_size

    kernel.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)
