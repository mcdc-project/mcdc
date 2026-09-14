from numba import njit

###

import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank
import mcdc.transport.util as util
import mcdc.code_factory.gpu.program.builder as gpu_program

from mcdc.constant import GPU_ASYNC_SIMPLE

import mcdc.code_factory.gpu.interface as interface
import mcdc.code_factory.gpu.substitution as sub

# =============================================================================
# Bank and pop particle
# =============================================================================


@sub.candidate(particle_bank.bank_active_particle)
def bank_active_particle(particle_container, program):
    simulation = util.access_simulation(program)

    active_particle_container = util.local_array(1, type_.particle)
    particle_module.copy(active_particle_container, particle_container)
    interface.step_async(program, active_particle_container[0])


@sub.candidate(particle_bank.report_full_bank)
def report_full_bank(bank):
    pass


@sub.candidate(particle_bank.report_empty_bank)
def report_empty_bank(bank):
    pass
