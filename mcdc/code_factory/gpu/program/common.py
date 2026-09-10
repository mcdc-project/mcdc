import numba as nb
import harmonize

import mcdc.code_factory.gpu.interface as interface
from mcdc.transport.simulation import generate_source_particle
from mcdc.transport.util import atomic_add

data_shape = eval(f"{interface.data_shape}")


def make_work(program: nb.uintp) -> nb.boolean:
    simulation = interface.access_simulation(program)
    data_ptr = interface.access_data_ptr(program)
    data = harmonize.array_from_ptr(data_ptr, data_shape, nb.float64)

    idx_work = atomic_add(simulation["mpi_work_iter"], 0, 1)

    if idx_work >= simulation["mpi_work_size"]:
        return False

    work_start = simulation["mpi_work_start"]

    generate_source_particle(
        simulation["mpi_work_start"],
        nb.uint64(idx_work),
        simulation["source_seed"],
        program,
        data,
    )
    return True


def initialize(program: nb.uintp):
    pass


def finalize(program: nb.uintp):
    pass
