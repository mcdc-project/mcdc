import harmonize

from numba import njit

###

import mcdc.code_factory.gpu.program_builder as gpu_module
import mcdc.config as config
import mcdc.transport.particle_bank as particle_bank_module

from mcdc.constant import GPU_STORAGE_SEPARATE, GPU_STRATEGY_ASYNC
from mcdc.transport.simulation import source_closeout

caching = config.caching


@njit(cache=caching)
def source_loop(seed, simulation, data):
    # For async execution
    iter_count = 655360000
    # For event-based execution
    batch_size = 64

    settings = simulation["settings"]

    full_work_size = simulation["mpi_work_size"]

    if settings["gpu_strategy"] == GPU_STRATEGY_ASYNC:
        phase_size = 1000000000
    else:
        phase_size = 1000000
    phase_count = (full_work_size + phase_size - 1) // phase_size

    for phase in range(phase_count):

        simulation["mpi_work_iter"][0] = phase_size * phase
        simulation["mpi_work_size"] = min(phase_size * (phase + 1), full_work_size)
        simulation["source_seed"] = seed

        # Store the global state to the GPU
        if settings["gpu_storage"] == GPU_STORAGE_SEPARATE:
            harmonize.memcpy_host_to_device(
                simulation["gpu_meta"]["state_pointer"], simulation
            )
            harmonize.memcpy_host_to_device(
                simulation["gpu_meta"]["state_pointer"], data
            )

        # Execute the program, and continue to do so until it is done
        block_count = gpu_module.BLOCK_COUNT

        if settings["gpu_strategy"] == GPU_STRATEGY_ASYNC:
            gpu_module.exec_program(
                simulation["gpu_meta"]["program_pointer"], block_count, iter_count
            )
            while not gpu_module.complete(simulation["gpu_meta"]["program_pointer"]):
                gpu_module.exec_program(
                    simulation["gpu_meta"]["program_pointer"], block_count, iter_count
                )
        else:
            gpu_module.exec_program(
                simulation["gpu_meta"]["program_pointer"], block_count, batch_size
            )
            while not gpu_module.complete(simulation["gpu_meta"]["program_pointer"]):
                gpu_module.exec_program(
                    simulation["gpu_meta"]["program_pointer"], block_count, batch_size
                )
        gpu_module.clear_flags(simulation["gpu_meta"]["program_pointer"])

        # Recover the original program state
        if config.gpu_state_storage == "separate":
            harmonize.memcpy_device_to_host(
                simulation, simulation["gpu_meta"]["state_pointer"]
            )
            harmonize.memcpy_device_to_host(
                data, simulation["gpu_meta"]["state_pointer"]
            )

        gpu_module.clear_flags(simulation["gpu_meta"]["program_pointer"])

    simulation["mpi_work_size"] = full_work_size

    particle_bank_module.set_bank_size(simulation["bank_active"], 0)

    source_closeout(simulation, 1, 1, data)
