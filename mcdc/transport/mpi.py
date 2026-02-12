import math

from numba import njit


@njit
def distribute_work(N_work, mcdc):
    size = mcdc["mpi_size"]
    rank = mcdc["mpi_rank"]

    # Total number of work
    work_size_total = N_work

    # Evenly distribute work
    work_size = math.floor(N_work / size)

    # Starting index (based on even distribution)
    work_start = work_size * rank

    # Count reminder
    rem = N_work % size

    # Assign reminder and update starting index
    if rank < rem:
        work_size += 1
        work_start += rank
    else:
        work_start += rem

    # Store the workload specification
    mcdc["mpi_work_start"] = work_start
    mcdc["mpi_work_size"] = work_size
    mcdc["mpi_work_size_total"] = work_size_total
