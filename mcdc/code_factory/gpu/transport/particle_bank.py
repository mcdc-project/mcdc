from numba import njit

# =============================================================================
# Bank and pop particle
# =============================================================================


@njit
def bank_active_particle(P_rec_arr, mcdc):
    particle_container = local_array(1, type_.particle)
    kernel.recordlike_to_particle(particle_container, P_rec_arr)
    if SIMPLE_ASYNC:
        step_async(prog, particle_container[0])
    else:
        find_cell_async(prog, particle_container[0])


@njit
def report_full_bank(bank):
    pass


@njit
def report_empty_bank(bank):
    pass
