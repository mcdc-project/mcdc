import mcdc.config as config

caching = config.caching


@njit(cache=caching)
def source_loop(seed, data, mcdc):
    # Progress bar indicator
    N_prog = 0

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

    source_closeout(simulation, 1, 1, data)
