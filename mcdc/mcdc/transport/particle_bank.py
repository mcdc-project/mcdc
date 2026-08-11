import numpy as np

from mpi4py import MPI
from numba import (
    njit,
    objmode,
)

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.mpi as mpi
import mcdc.transport.particle as particle_module
import mcdc.transport.technique as technique
import mcdc.transport.util as util

from mcdc.constant import *
from mcdc.print_ import print_error

# =============================================================================
# Bank size
# =============================================================================


@njit
def get_bank_size(bank):
    return bank["size"][0]


@njit
def set_bank_size(bank, value):
    bank["size"][0] = value


@njit
def add_bank_size(bank, value):
    util.atomic_add(bank["size"], 0, value)


# =============================================================================
# Bank and pop particle
# =============================================================================


@njit
def _bank_particle(particle_container, bank):
    # Check if bank is full
    if get_bank_size(bank) == bank["particle_data"].shape[0]:
        report_full_bank(bank)

    # Set particle data
    idx = get_bank_size(bank)
    particle_module.copy(bank["particle_data"][idx : idx + 1], particle_container)


@njit
def bank_active_particle(particle_container, program):
    simulation = util.access_simulation(program)
    bank = simulation["bank_active"]
    _bank_particle(particle_container, bank)

    # Increment bank size
    add_bank_size(bank, 1)


@njit
def bank_census_particle(particle_container, program):
    simulation = util.access_simulation(program)
    bank = simulation["bank_census"]
    _bank_particle(particle_container, bank)

    # Increment bank size
    add_bank_size(bank, 1)


@njit
def bank_future_particle(particle_container, program):
    simulation = util.access_simulation(program)
    bank = simulation["bank_future"]
    _bank_particle(particle_container, bank)

    # Increment bank size
    add_bank_size(bank, 1)


@njit
def bank_source_particle(particle_container, simulation):
    bank = simulation["bank_source"]
    _bank_particle(particle_container, bank)

    # Increment bank size
    #   Note that we don't use the atomic operation in add_bank_size function
    #   as source particle banking is not thread-parallelized
    bank["size"][0] += 1


@njit
def pop_particle(particle_container, bank):
    # Check if bank is empty
    if get_bank_size(bank) == 0:
        report_empty_bank(bank)

    # Set particle data
    idx = get_bank_size(bank) - 1
    particle_module.copy(particle_container, bank["particle_data"][idx : idx + 1])

    # Decrement bank size
    add_bank_size(bank, -1)

    # Set default IDs and event for the live particle
    particle = particle_container[0]
    particle["alive"] = True
    particle["material_ID"] = -1
    particle["cell_ID"] = -1
    particle["surface_ID"] = -1
    particle["event"] = -1


@njit
def report_full_bank(bank):
    with objmode():
        print_error("Particle %s bank is full." % bank["tag"])


@njit
def report_empty_bank(bank):
    with objmode():
        print_error("Attempting to get a particle from an empty %s bank." % bank["tag"])


# ======================================================================================
# Future bank management
# ======================================================================================


@njit
def promote_future_particles(program, data):
    simulation = util.access_simulation(program)

    # Get the banks
    future_bank = simulation["bank_future"]

    # Get the next census time
    idx = simulation["idx_census"] + 1
    next_census_time = mcdc_get.settings.census_time(idx, simulation["settings"], data)

    # Particle container
    particle_container = util.local_array(1, type_.particle_data)
    particle = particle_container[0]

    # Loop over all particles in future bank
    initial_size = get_bank_size(future_bank)
    for i in range(initial_size):
        # Get the next future particle index
        #   NOTE: future bank size decreases as particles are promoted to census bank
        idx = i - (initial_size - get_bank_size(future_bank))
        particle_module.copy(
            particle_container, future_bank["particle_data"][idx : idx + 1]
        )

        # Promote the future particle to census bank
        if particle["t"] < next_census_time:

            bank_census_particle(particle_container, program)
            add_bank_size(future_bank, -1)

            # Consolidate the emptied space in the future bank
            j = get_bank_size(future_bank)
            particle_module.copy(
                future_bank["particle_data"][idx : idx + 1],
                future_bank["particle_data"][j : j + 1],
            )


# ======================================================================================
# All-bank management
# ======================================================================================


@njit
def manage_particle_banks(simulation):
    master = simulation["mpi_master"]
    serial = simulation["mpi_size"] == 1

    # TIMER: bank management
    time_start = 0.0
    if master:
        with objmode(time_start="float64"):
            time_start = MPI.Wtime()
    time_spent = -time_start

    # Reset source bank
    set_bank_size(simulation["bank_source"], 0)

    # Normalize weight
    if simulation["settings"]["neutron_eigenvalue_mode"]:
        normalize_weight(
            simulation["bank_census"], simulation["settings"]["N_particle"]
        )

    # Population control
    if simulation["population_control"]["active"]:
        technique.population_control(simulation)
    else:
        # Swap census and source bank
        source_bank = simulation["bank_source"]
        census_bank = simulation["bank_census"]

        size = get_bank_size(census_bank)
        if size >= source_bank["particle_data"].shape[0]:
            report_full_bank(source_bank)

        # TODO: better alternative?
        source_bank["particle_data"][:size] = census_bank["particle_data"][:size]
        set_bank_size(source_bank, size)

    # Redistribute work and rebalance bank size across MPI ranks
    if serial:
        mpi.distribute_work(get_bank_size(simulation["bank_source"]), simulation)
    else:
        bank_rebalance(simulation)

    # Reset census bank
    set_bank_size(simulation["bank_census"], 0)

    # TIMER: bank management
    time_end = 0.0
    if master:
        with objmode(time_end="float64"):
            time_end = MPI.Wtime()
    time_spent += time_end
    if master:
        simulation["runtime_bank_management"] += time_spent


# ======================================================================================
# Bank size parallel rebalance
# ======================================================================================


@njit
def bank_rebalance(simulation):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(simulation["bank_source"], simulation)
    idx_end = idx_start + N_local
    mpi.distribute_work(N, simulation)

    # Abort if source bank is empty
    if N == 0:
        return

    # Rebalance not needed if there is only one rank
    if simulation["mpi_size"] <= 1:
        return

    # Some constants
    work_start = simulation["mpi_work_start"]
    work_end = work_start + simulation["mpi_work_size"]
    left = simulation["mpi_rank"] - 1
    right = simulation["mpi_rank"] + 1

    # Flags if need to receive from or sent to the neighbors
    send_to_left = idx_start < work_start
    receive_from_left = idx_start > work_start
    send_to_right = idx_end > work_end
    receive_from_right = idx_end < work_end

    # Flags if need to receive first
    receive_first = False
    if receive_from_left:
        receive_first = idx_start >= work_end
    if receive_from_right:
        receive_first = idx_end <= work_start

    # MPI nearest-neighbor send/receive
    buff = np.zeros(
        simulation["bank_source"]["particle_data"].shape[0], dtype=type_.particle_data
    )

    with objmode(size="int64"):
        # Create MPI-supported numpy object
        size = get_bank_size(simulation["bank_source"])
        bank = np.array(simulation["bank_source"]["particle_data"][:size])

        if receive_first:
            if receive_from_left:
                bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
                receive_from_left = False
            if receive_from_right:
                bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))
                receive_from_right = False

        if send_to_left:
            n = work_start - idx_start
            send_to_left_status = MPI.COMM_WORLD.isend(bank[:n], dest=left)
            bank = bank[n:]
        if send_to_right:
            n = idx_end - work_end
            send_to_right_status = MPI.COMM_WORLD.isend(bank[-n:], dest=right)
            bank = bank[:-n]

        if receive_from_left:
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
        if receive_from_right:
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))

        # Wait until sent massage is received
        if send_to_left:
            send_to_left_status.Wait()
        if send_to_right:
            send_to_right_status.Wait()

        # Set output buffer
        size = bank.shape[0]
        for i in range(size):
            buff[i] = bank[i]

    # Set source bank from buffer
    set_bank_size(simulation["bank_source"], size)
    for i in range(size):
        simulation["bank_source"]["particle_data"][i] = buff[i]


# ======================================================================================
# MPI collective operations
# ======================================================================================


@njit
def bank_scanning(bank, simulation):
    N_local = get_bank_size(bank)

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, simulation["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def bank_scanning_weight(bank, simulation):
    # Local weight CDF
    N_local = get_bank_size(bank)
    w_cdf = np.zeros(N_local + 1)
    for i in range(N_local):
        w_cdf[i + 1] = w_cdf[i] + bank["particle_data"][i]["w"]
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
        MPI.COMM_WORLD.Bcast(buff, simulation["mpi_size"] - 1)
    W_global = buff[0]

    return w_start, w_cdf, W_global


@njit
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for i in range(get_bank_size(bank)):
        bank["particle_data"][i]["w"] *= norm / W


@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(get_bank_size(bank)):
        W_local[0] += bank["particle_data"][i]["w"]

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
