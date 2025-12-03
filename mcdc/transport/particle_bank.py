import mcdc.mcdc_get as mcdc_get

import numpy as np

from mpi4py import MPI
from numba import (
    njit,
    objmode,
)

####

import mcdc.code_factory.adapt as adapt
import mcdc.object_.numba_types as type_
import mcdc.transport.mpi as mpi
import mcdc.transport.particle as particle_module
import mcdc.transport.technique as technique

from mcdc.constant import *
from mcdc.print_ import print_error


# =============================================================================
# Particle bank operations
# =============================================================================


@njit
def get_bank_size(bank):
    return bank["size"][0]


@njit
def set_bank_size(bank, value):
    bank["size"][0] = value


@njit
def add_bank_size(bank, value):
    return adapt.global_add(bank["size"], 0, value)


@njit
def add_particle(P_arr, bank):
    P = P_arr[0]

    idx = add_bank_size(bank, 1)

    # Check if bank is full
    if idx >= bank["particles"].shape[0]:
        full_bank_print(bank)

    # Set particle
    particle_module.copy(bank["particles"][idx : idx + 1], P_arr)


@njit
def get_particle(P_arr, bank, mcdc):
    P = P_arr[0]

    idx = add_bank_size(bank, -1) - 1

    # Check if bank is empty
    if idx < 0:
        return False
        # with objmode():
        #    print_error("Particle %s bank is empty." % bank["tag"])

    # Set attribute
    P_rec = bank["particles"][idx]
    P["x"] = P_rec["x"]
    P["y"] = P_rec["y"]
    P["z"] = P_rec["z"]
    P["t"] = P_rec["t"]
    P["ux"] = P_rec["ux"]
    P["uy"] = P_rec["uy"]
    P["uz"] = P_rec["uz"]
    P["g"] = P_rec["g"]
    P["E"] = P_rec["E"]
    P["w"] = P_rec["w"]
    P["particle_type"] = P_rec["particle_type"]
    P["rng_seed"] = P_rec["rng_seed"]

    # Set default IDs and event
    P["alive"] = True
    P["material_ID"] = -1
    P["cell_ID"] = -1
    P["surface_ID"] = -1
    P["event"] = -1
    return True


@njit
def check_future_bank(mcdc, data):
    # Get the data needed
    settings = mcdc["settings"]
    bank_future = mcdc["bank_future"]
    bank_census = mcdc["bank_census"]
    next_census_time = mcdc_get.settings.census_time(
        mcdc["idx_census"] + 1, settings, data
    )

    # Particle container
    P_arr = np.zeros(1, type_.particle_data)
    P = P_arr[0]

    # Loop over all particles in future bank
    N = get_bank_size(bank_future)
    for i in range(N):
        # Get the next future particle index
        idx = i - get_bank_size(bank_census)
        particle_module.copy(P_arr, bank_future["particles"][idx : idx + 1])

        # Promote the future particle to census bank
        if P["t"] < next_census_time:
            add_census(P_arr, mcdc)
            add_bank_size(bank_future, -1)

            # Consolidate the emptied space in the future bank
            j = get_bank_size(bank_future)
            particle_module.copy(
                bank_future["particles"][idx : idx + 1],
                bank_future["particles"][j : j + 1],
            )


@njit
def manage_particle_banks(mcdc):
    # Record time
    if mcdc["mpi_master"]:
        with objmode(time_start="float64"):
            time_start = MPI.Wtime()

    # Reset source bank
    set_bank_size(mcdc["bank_source"], 0)

    # Normalize weight
    if mcdc["settings"]["eigenvalue_mode"]:
        normalize_weight(mcdc["bank_census"], mcdc["settings"]["N_particle"])

    # Population control
    if mcdc["population_control"]["active"]:
        technique.population_control(mcdc)
    else:
        # Swap census and source bank
        source_bank = mcdc["bank_source"]
        census_bank = mcdc["bank_census"]

        size = get_bank_size(census_bank)
        if size >= source_bank["particles"].shape[0]:
            full_bank_print(source_bank)
        source_bank["particles"][:size] = census_bank["particles"][:size]
        set_bank_size(source_bank, size)
    # TODO: Population control future bank?

    # MPI rebalance
    bank_rebalance(mcdc)

    # Zero out census bank
    set_bank_size(mcdc["bank_census"], 0)

    # Accumulate time
    if mcdc["mpi_master"]:
        with objmode(time_end="float64"):
            time_end = MPI.Wtime()
        mcdc["runtime_bank_management"] += time_end - time_start


@njit
def bank_scanning(bank, mcdc):
    N_local = get_bank_size(bank)

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def bank_scanning_weight(bank, mcdc):
    # Local weight CDF
    N_local = get_bank_size(bank)
    w_cdf = np.zeros(N_local + 1)
    for i in range(N_local):
        w_cdf[i + 1] = w_cdf[i] + bank["particles"][i]["w"]
    W_local = w_cdf[-1]

    # Starting weight
    buff = np.zeros(1, dtype=np.float64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([W_local]), buff, MPI.SUM)
    w_start = buff[0]
    w_cdf += w_start

    # Global weight
    buff[0] = w_cdf[-1]
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    W_global = buff[0]

    return w_start, w_cdf, W_global


@njit
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for i in range(get_bank_size(bank)):
        bank["particles"][i]["w"] *= norm / W


@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(get_bank_size(bank)):
        W_local[0] += bank["particles"][i]["w"]

    # MPI Allreduce
    buff = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(W_local, buff, MPI.SUM)
    return buff[0]


@njit
def total_size(bank):
    # Local total weight
    local_size = np.ones(1, np.int64) * bank["size"]

    # MPI Allreduce
    buff = np.zeros(1, np.int64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(local_size, buff, MPI.SUM)
    return buff[0]


@njit
def bank_rebalance(mcdc):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(mcdc["bank_source"], mcdc)
    idx_end = idx_start + N_local

    # Abort if source bank is empty
    if N == 0:
        return

    mpi.distribute_work(N, mcdc)

    # Rebalance not needed if there is only one rank
    if mcdc["mpi_size"] <= 1:
        return

    # Some constants
    work_start = mcdc["mpi_work_start"]
    work_end = work_start + mcdc["mpi_work_size"]
    left = mcdc["mpi_rank"] - 1
    right = mcdc["mpi_rank"] + 1

    # Need more or less?
    more_left = idx_start < work_start
    less_left = idx_start > work_start
    more_right = idx_end > work_end
    less_right = idx_end < work_end

    # Offside?
    offside_left = idx_end <= work_start and work_start != work_end
    offside_right = idx_start >= work_end and work_start != work_end

    # MPI nearest-neighbor send/receive
    buff = np.zeros(
        mcdc["bank_source"]["particles"].shape[0], dtype=type_.particle_data
    )

    with objmode(size="int64"):
        # Create MPI-supported numpy object
        size = get_bank_size(mcdc["bank_source"])
        bank = np.array(mcdc["bank_source"]["particles"][:size])

        # If offside, need to receive first
        if offside_left:
            # Receive from right
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))
            less_right = False
        if offside_right:
            # Receive from left
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
            less_left = False

        # Send
        if more_left:
            n = work_start - idx_start
            request_left = MPI.COMM_WORLD.isend(bank[:n], dest=left)
            bank = bank[n:]
        if more_right:
            n = idx_end - work_end
            request_right = MPI.COMM_WORLD.isend(bank[-n:], dest=right)
            bank = bank[:-n]

        # Receive
        if less_left:
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
        if less_right:
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))

        # Wait until sent massage is received
        if more_left:
            request_left.Wait()
        if more_right:
            request_right.Wait()

        # Set output buffer
        size = bank.shape[0]
        for i in range(size):
            buff[i] = bank[i]

    # Set source bank from buffer
    set_bank_size(mcdc["bank_source"], size)
    for i in range(size):
        mcdc["bank_source"]["particles"][i] = buff[i]


# ======================================================================================
# Adaptive functions
# ======================================================================================
# TODO: Need review


@adapt.for_cpu()
def full_bank_print(bank):
    with objmode():
        print_error("Particle %s bank is full." % bank["tag"])


@adapt.for_gpu()
def full_bank_print(bank):
    pass


@adapt.for_cpu()
def add_active(P_arr, prog):
    add_particle(P_arr, prog["bank_active"])


@adapt.for_gpu()
def add_active(P_rec_arr, prog):
    P_arr = local_array(1, type_.particle)
    kernel.recordlike_to_particle(P_arr, P_rec_arr)
    if SIMPLE_ASYNC:
        step_async(prog, P_arr[0])
    else:
        find_cell_async(prog, P_arr[0])


@adapt.for_cpu()
def add_source(P_arr, prog):
    add_particle(P_arr, prog["bank_source"])


@adapt.for_gpu()
def add_source(P_arr, prog):
    mcdc = mcdc_global(prog)
    add_particle(P_arr, mcdc["bank_source"])


@adapt.for_cpu()
def add_census(P_arr, prog):
    add_particle(P_arr, prog["bank_census"])


@adapt.for_gpu()
def add_census(P_arr, prog):
    mcdc = mcdc_global(prog)
    add_particle(P_arr, mcdc["bank_census"])


@adapt.for_cpu()
def add_future(P_arr, prog):
    add_particle(P_arr, prog["bank_future"])


@adapt.for_gpu()
def add_future(P_arr, prog):
    mcdc = mcdc_global(prog)
    add_particle(P_arr, mcdc["bank_future"])
