import math
import numpy as np

from numba import literal_unroll, njit, objmode
from mpi4py import MPI

####

import mcdc.mcdc_set as mcdc_set
import mcdc.transport.particle_bank as particle_bank_module

from mcdc.constant import (
    GYRATION_RADIUS_ALL,
    GYRATION_RADIUS_INFINITE_X,
    GYRATION_RADIUS_INFINITE_Y,
    GYRATION_RADIUS_INFINITE_Z,
    GYRATION_RADIUS_ONLY_X,
    GYRATION_RADIUS_ONLY_Y,
    GYRATION_RADIUS_ONLY_Z,
)
from mcdc.print_ import print_structure

# ======================================================================================
# Reduce tally bins
# ======================================================================================


@njit
def reduce(simulation, data):
    for tally in simulation["tallies"]:
        _reduce(tally, simulation, data)


@njit
def _reduce(tally, simulation, data):
    N = tally["bin_length"]
    start = tally["bin_offset"]
    end = start + N

    # Normalize
    N_particle = simulation["settings"]["N_particle"]
    for i in range(N):
        data[start + i] /= N_particle

    # MPI Reduce
    buff = np.zeros(N)
    with objmode():
        MPI.COMM_WORLD.Reduce(data[start:end], buff, MPI.SUM, 0)
    data[start:end] = buff


# ======================================================================================
# Accumulate tally bins
# ======================================================================================


@njit
def accumulate(simulation, data):
    for tally in simulation["tallies"]:
        _accumulate(tally, data)


@njit
def _accumulate(tally, data):
    N_bin = tally["bin_length"]
    offset_bin = tally["bin_offset"]
    offset_sum = tally["bin_sum_offset"]
    offset_sum_square = tally["bin_sum_square_offset"]

    # Note: Three separate loops are employed to avoid cache miss due to potentially
    #       large N_bin

    # Sum of score
    for i in range(N_bin):
        score = data[offset_bin + i]
        data[offset_sum + i] += score

    # Sum of score squared
    for i in range(N_bin):
        score = data[offset_bin + i]
        data[offset_sum_square + i] += score * score

    # Reset score bin
    for i in range(N_bin):
        data[offset_bin + i] = 0.0


# ======================================================================================
# Finalize
# ======================================================================================


@njit
def finalize(simulation, data):
    for tally in simulation["tallies"]:
        _finalize(tally, simulation, data)


@njit
def _finalize(tally, simulation, data):
    N_history = simulation["settings"]["N_particle"]
    N_batch = simulation["settings"]["N_batch"]
    N_bin = tally["bin_length"]
    sum_start = tally["bin_sum_offset"]
    sum_sq_start = tally["bin_sum_square_offset"]
    sum_end = sum_start + N_bin
    sum_sq_end = sum_sq_start + N_bin

    if N_batch > 1:
        N_history = N_batch

    elif simulation["settings"]["neutron_eigenvalue_mode"]:
        N_history = simulation["settings"]["N_active"]

    else:
        # MPI Reduce
        buff = np.zeros(N_bin)
        buff_sq = np.zeros(N_bin)
        with objmode():
            MPI.COMM_WORLD.Reduce(data[sum_start:sum_end], buff, MPI.SUM, 0)
            MPI.COMM_WORLD.Reduce(data[sum_sq_start:sum_sq_end], buff_sq, MPI.SUM, 0)
        data[sum_start:sum_end] = buff
        data[sum_sq_start:sum_sq_end] = buff_sq

    # Calculate and store statistics
    #   sum --> mean
    #   sum_sq --> standard deviation
    N_bin = tally["bin_length"]
    offset_sum = tally["bin_sum_offset"]
    offset_sum_square = tally["bin_sum_square_offset"]
    for i in range(N_bin):
        data[offset_sum + i] = data[offset_sum + i] / N_history
        radicand = (
            data[offset_sum_square + i] / N_history - np.square(data[offset_sum + i])
        ) / (N_history - 1)

        # Check for round-off error (TODO: Check why this is needed.)
        if abs(radicand) < 1e-16:
            data[offset_sum_square + i] = 0.0
        if radicand < 0.0 and abs(radicand) < 1e-6:
            data[offset_sum_square + i] = 0.0
        else: 
            data[offset_sum_square + i] = math.sqrt(radicand)


# ======================================================================================
# Reset sum bins
# ======================================================================================


@njit
def reset_sum_bins(simulation, data):
    for tally in simulation["tallies"]:
        _reset_sum_bins(tally, data)


@njit
def _reset_sum_bins(tally, data):
    N_bin = tally["bin_length"]
    offset_sum = tally["bin_sum_offset"]
    offset_sum_square = tally["bin_sum_square_offset"]

    for i in range(N_bin):
        data[offset_sum + i] = 0.0
        data[offset_sum_square + i] = 0.0


# ======================================================================================
# Eigenvalue
# ======================================================================================


@njit
def eigenvalue_cycle(simulation, data):
    idx_cycle = simulation["idx_cycle"]
    N_particle = simulation["settings"]["N_particle"]

    # MPI Allreduce
    buff_nuSigmaF = np.zeros(1, np.float64)
    buff_n = np.zeros(1, np.float64)
    buff_nmax = np.zeros(1, np.float64)
    buff_C = np.zeros(1, np.float64)
    buff_Cmax = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(
            np.array(simulation["eigenvalue_tally_nuSigmaF"]), buff_nuSigmaF, MPI.SUM
        )
        if simulation["cycle_active"]:
            MPI.COMM_WORLD.Allreduce(
                np.array(simulation["eigenvalue_tally_n"]), buff_n, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(
                np.array([simulation["n_max"]]), buff_nmax, MPI.MAX
            )
            MPI.COMM_WORLD.Allreduce(
                np.array(simulation["eigenvalue_tally_C"]), buff_C, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(
                np.array([simulation["C_max"]]), buff_Cmax, MPI.MAX
            )

    # Update and store k_eff
    simulation["k_eff"] = buff_nuSigmaF[0] / N_particle
    mcdc_set.simulation.k_cycle(idx_cycle, simulation, data, value=simulation["k_eff"])

    # Normalize other eigenvalue/global tallies
    tally_n = buff_n[0] / N_particle
    tally_C = buff_C[0] / N_particle

    # Maximum densities
    simulation["n_max"] = buff_nmax[0]
    simulation["C_max"] = buff_Cmax[0]

    # Accumulate running average
    if simulation["cycle_active"]:
        simulation["k_avg"] += simulation["k_eff"]
        simulation["k_sdv"] += simulation["k_eff"] * simulation["k_eff"]
        simulation["n_avg"] += tally_n
        simulation["n_sdv"] += tally_n * tally_n
        simulation["C_avg"] += tally_C
        simulation["C_sdv"] += tally_C * tally_C

        N = 1 + simulation["idx_cycle"] - simulation["settings"]["N_inactive"]
        simulation["k_avg_running"] = simulation["k_avg"] / N
        if N == 1:
            simulation["k_sdv_running"] = 0.0
        else:
            simulation["k_sdv_running"] = math.sqrt(
                (simulation["k_sdv"] / N - simulation["k_avg_running"] ** 2) / (N - 1)
            )

    # Reset accumulators
    simulation["eigenvalue_tally_nuSigmaF"][0] = 0.0
    simulation["eigenvalue_tally_n"][0] = 0.0
    simulation["eigenvalue_tally_C"][0] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================

    if simulation["settings"]["use_gyration_radius"]:
        # Center of mass
        N_local = particle_bank_module.get_bank_size(simulation["bank_census"])
        total_local = np.zeros(4, np.float64)  # [x,y,z,W]
        total = np.zeros(4, np.float64)
        for i in range(N_local):
            P = simulation["bank_census"]["particle_data"][i]
            total_local[0] += P["x"] * P["w"]
            total_local[1] += P["y"] * P["w"]
            total_local[2] += P["z"] * P["w"]
            total_local[3] += P["w"]
        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(total_local, total, MPI.SUM)
        # COM
        W = total[3]
        com_x = total[0] / W
        com_y = total[1] / W
        com_z = total[2] / W

        # Distance RMS
        rms_local = np.zeros(1, np.float64)
        rms = np.zeros(1, np.float64)
        gr_type = simulation["settings"]["gyration_radius_type"]
        if gr_type == GYRATION_RADIUS_ALL:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += (
                    (P["x"] - com_x) ** 2
                    + (P["y"] - com_y) ** 2
                    + (P["z"] - com_z) ** 2
                ) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_X:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Y:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Z:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_X:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Y:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Z:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["z"] - com_z) ** 2) * P["w"]

        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(rms_local, rms, MPI.SUM)
        rms = math.sqrt(rms[0] / W)

        # Gyration radius
        mcdc_set.simulation.gyration_radius(idx_cycle, simulation, data, value=rms)


@njit
def eigenvalue_simulation(simulation):
    N = simulation["settings"]["N_active"]
    simulation["n_avg"] /= N
    simulation["C_avg"] /= N
    if N > 1:
        simulation["n_sdv"] = math.sqrt(
            (simulation["n_sdv"] / N - simulation["n_avg"] ** 2) / (N - 1)
        )
        simulation["C_sdv"] = math.sqrt(
            (simulation["C_sdv"] / N - simulation["C_avg"] ** 2) / (N - 1)
        )
    else:
        simulation["n_sdv"] = 0.0
        simulation["C_sdv"] = 0.0
