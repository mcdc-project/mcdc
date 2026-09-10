import harmonize

from numba import njit

###

import mcdc.code_factory.gpu.interface as gpu_interface
import mcdc.config as config
import mcdc.transport.particle_bank as particle_bank_module

from mcdc.constant import GPU_STORAGE_SEPARATE, GPU_STRATEGY_ASYNC
from mcdc.transport.simulation import source_closeout

caching = config.caching


@njit(cache=False)
def source_loop(seed, simulation, data):
    # For async execution
    iter_count = 655360000
    # For event-based execution
    batch_size = 64

    settings = simulation["settings"]

    full_work_size = simulation["mpi_work_size"]

    if full_work_size == 0:
        return

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
            gpu_interface.store_state_device_simulation(
                simulation["gpu_meta"]["state_pointer"], simulation
            )
            gpu_interface.store_state_device_data(
                simulation["gpu_meta"]["state_pointer"], data
            )

        # Execute the program, and continue to do so until it is done
        block_count = gpu_interface.BLOCK_COUNT

        if settings["gpu_strategy"] == GPU_STRATEGY_ASYNC:
            gpu_interface.exec_program(
                simulation["gpu_meta"]["program_pointer"], block_count, iter_count
            )
            while not gpu_interface.complete(simulation["gpu_meta"]["program_pointer"]):
                gpu_interface.exec_program(
                    simulation["gpu_meta"]["program_pointer"], block_count, iter_count
                )
        else:
            gpu_interface.exec_program(
                simulation["gpu_meta"]["program_pointer"], block_count, batch_size
            )
            while not gpu_interface.complete(simulation["gpu_meta"]["program_pointer"]):
                gpu_interface.exec_program(
                    simulation["gpu_meta"]["program_pointer"], block_count, batch_size
                )
        gpu_interface.clear_flags(simulation["gpu_meta"]["program_pointer"])

        # Recover the original program state
        if settings["gpu_storage"] == GPU_STORAGE_SEPARATE:
            gpu_interface.load_state_device_simulation(
                simulation, simulation["gpu_meta"]["state_pointer"]
            )
            gpu_interface.load_state_device_data(
                data, simulation["gpu_meta"]["state_pointer"]
            )

    simulation["mpi_work_size"] = full_work_size

    particle_bank_module.set_bank_size(simulation["bank_active"], 0)

    source_closeout(simulation, 1, 1, data)
