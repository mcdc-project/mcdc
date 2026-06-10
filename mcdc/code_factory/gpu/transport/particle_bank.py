from numba import njit

###

import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.util as util
import mcdc.code_factory.gpu.program_builder as gpu_program

from mcdc.constant import GPU_ASYNC_SIMPLE

# =============================================================================
# Bank and pop particle
# =============================================================================


@njit
def bank_active_particle(particle_container, program):
    simulation = util.access_simulation(program)

    active_particle_container = util.local_array(1, type_.particle)
    particle_module.copy(active_particle_container, particle_container)
    if simulation["settings"]["gpu_async_type"] == GPU_ASYNC_SIMPLE:
        gpu_program.step_async(program, active_particle_container[0])
    """
    else:
        gpu_program.find_cell_async(program, active_particle_container[0])
    """


@njit
def report_full_bank(bank):
    pass


@njit
def report_empty_bank(bank):
    pass
