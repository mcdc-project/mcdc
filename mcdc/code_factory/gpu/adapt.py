import harmonize

# ======================================================================================
# Particle bank
# ======================================================================================

from mcdc.transport import particle_bank


@adapt(particle_bank)
def report_full_bank(bank):
    pass


@adapt(particle_bank)
def report_empty_bank(bank):
    pass


@adapt(particle_bank)
def bank_active_particle(P_rec_arr, prog):
    particle_container = local_array(1, type_.particle)
    kernel.recordlike_to_particle(particle_container, P_rec_arr)
    if SIMPLE_ASYNC:
        step_async(prog, particle_container[0])
    else:
        find_cell_async(prog, particle_container[0])


# ======================================================================================
# Utilities
# ======================================================================================

from mcdc.transport import util


@adapt(util)
def atomic_add(ary, idx, val):
    return harmonize.array_atomic_add(ary, idx, val)


# ======================================================================================
# Geometry
# ======================================================================================

from mcdc.transport import geometry


@adapt(geometry.interface)
def reporrt_lost_particle(particle_container, mcdc):
    particle = particle_container[0]
    particle["alive"] = False
