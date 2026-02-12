import numba as nb
import sys
from mpi4py import MPI
from colorama import Fore, Back, Style

master = MPI.COMM_WORLD.Get_rank() == 0


import numba as nb
import sys

from colorama import Fore, Style

import mcdc.mcdc_get as mcdc_get


def print_1d_array(arr):
    N = len(arr)
    if N > 5:
        return f"(size={len(arr)}): [{arr[0]:.5g}, {arr[1]:.5g}, ..., {arr[-2]:.5g}, {arr[-1]:.5g}]"
    else:
        text = f"(size={len(arr)}): ["
        for i in range(N):
            text += f"{arr[i]:.5g}, "
        if N > 0:
            text = text[:-2]
        text += "]"
        return text


def print_error(text):
    print(Fore.RED + f"[ERROR]: {text}\n")
    print(Style.RESET_ALL)
    sys.stdout.flush()
    sys.exit()


def print_warning(text):
    print(Fore.YELLOW + f"[WARNING]: {text}\n")
    print(Style.RESET_ALL)
    sys.stdout.flush()


def print_banner():
    print(
        "\n"
        + r"  __  __  ____  __ ____   ____ "
        + "\n"
        + r" |  \/  |/ ___|/ /_  _ \ / ___|"
        + "\n"
        + r" | |\/| | |   /_  / | | | |    "
        + "\n"
        + r" | |  | | |___ / /| |_| | |___ "
        + "\n"
        + r" |_|  |_|\____|// |____/ \____|"
        + "\n"
    )
    sys.stdout.flush()


def print_configuration():
    mode = "Python" if nb.config.DISABLE_JIT else "Numba"
    mpi_size = MPI.COMM_WORLD.Get_size()

    text = ""
    text += f"           Mode | {mode}\n"
    text += f"  MPI Processes | {mpi_size}\n"
    print(text)
    sys.stdout.flush()


def print_eigenvalue_header(mcdc):
    if mcdc["settings"]["use_gyration_radius"]:
        print("\n #     k        GyRad.  k (avg)            ")
        print(" ====  =======  ======  ===================")
    else:
        print("\n #     k        k (avg)            ")
        print(" ====  =======  ===================")
    sys.stdout.flush()


def print_batch_header(i, N):
    print(f"\nBatch {i}/{N}")
    sys.stdout.flush()


def print_time(tag, t, percent):
    if t >= 24 * 60 * 60:
        print("   %s | %.2f days (%.1f%%)" % (tag, t / 24 / 60 / 60), percent)
    elif t >= 60 * 60:
        print("   %s | %.2f hours (%.1f%%)" % (tag, t / 60 / 60, percent))
    elif t >= 60:
        print("   %s | %.2f minutes (%.1f%%)" % (tag, t / 60, percent))
    else:
        print("   %s | %.2f seconds (%.1f%%)" % (tag, t, percent))


def print_runtime(mcdc):
    total = mcdc["runtime_total"]
    preparation = mcdc["runtime_preparation"]
    simulation = mcdc["runtime_simulation"]
    output = mcdc["runtime_output"]
    print("\n Runtime report:")
    print_time("Total      ", total, 100)
    print_time("Preparation", preparation, preparation / total * 100)
    print_time("Simulation ", simulation, simulation / total * 100)
    print_time("Output     ", output, output / total * 100)
    print("\n")
    sys.stdout.flush()


def print_structure(struct):
    dtype = struct.dtype
    for name in dtype.names:
        print(f"{name} = {struct[name]}")


# TODO: below is not evaulated yet during the refactor


def print_msg(msg):
    if master:
        print(msg)
        sys.stdout.flush()


def print_error(msg):
    print("ERROR: %s\n" % msg)
    sys.stdout.flush()
    sys.exit()


def print_warning(msg):
    if master:
        print(Fore.RED + "Warning: %s\n" % msg)
        print(Style.RESET_ALL)
        sys.stdout.flush()


def print_progress(percent, mcdc):
    if master:
        sys.stdout.write("\r")
        if not mcdc["settings"]["eigenvalue_mode"]:
            if mcdc["settings"]["N_census"] == 1:
                sys.stdout.write(
                    " [%-28s] %d%%" % ("=" * int(percent * 28), percent * 100.0)
                )
            else:
                idx = mcdc["idx_census"] + 1
                N = mcdc["settings"]["N_census"]
                sys.stdout.write(
                    " Census %i/%i: [%-28s] %d%%"
                    % (idx, N, "=" * int(percent * 28), percent * 100.0)
                )
        else:
            if mcdc["settings"]["use_gyration_radius"]:
                sys.stdout.write(
                    " [%-40s] %d%%" % ("=" * int(percent * 40), percent * 100.0)
                )
            else:
                sys.stdout.write(
                    " [%-32s] %d%%" % ("=" * int(percent * 32), percent * 100.0)
                )
        sys.stdout.flush()


def print_header_eigenvalue(mcdc):
    if master:
        if mcdc["settings"]["use_gyration_radius"]:
            print("\n #     k        GyRad.  k (avg)            ")
            print(" ====  =======  ======  ===================")
        else:
            print("\n #     k        k (avg)            ")
            print(" ====  =======  ===================")


def print_header_batch(i, N):
    if master:
        print(f"\nBatch {i+1}/{N}")
        sys.stdout.flush()


def print_progress_eigenvalue(mcdc, data):
    if master:
        idx_cycle = mcdc["idx_cycle"]
        k_eff = mcdc["k_eff"]
        k_avg = mcdc["k_avg_running"]
        k_sdv = mcdc["k_sdv_running"]
        gr = mcdc_get.simulation.gyration_radius(idx_cycle, mcdc, data)
        if mcdc["settings"]["use_progress_bar"]:
            sys.stdout.write("\r")
            sys.stdout.write("\033[K")
        if mcdc["settings"]["use_gyration_radius"]:
            if not mcdc["cycle_active"]:
                print(" %-4i  %.5f  %6.2f" % (idx_cycle + 1, k_eff, gr))
            else:
                print(
                    " %-4i  %.5f  %6.2f  %.5f +/- %.5f"
                    % (idx_cycle + 1, k_eff, gr, k_avg, k_sdv)
                )
        else:
            if not mcdc["cycle_active"]:
                print(" %-4i  %.5f" % (idx_cycle + 1, k_eff))
            else:
                print(
                    " %-4i  %.5f  %.5f +/- %.5f" % (idx_cycle + 1, k_eff, k_avg, k_sdv)
                )


def print_runtime(mcdc):
    total = mcdc["runtime_total"]
    preparation = mcdc["runtime_preparation"]
    simulation = mcdc["runtime_simulation"]
    output = mcdc["runtime_output"]
    if master:
        print("\n Runtime report:")
        print_time("Total      ", total, 100)
        print_time("Preparation", preparation, preparation / total * 100)
        print_time("Simulation ", simulation, simulation / total * 100)
        print_time("Output     ", output, output / total * 100)
        print("\n")
        sys.stdout.flush()


def print_time(tag, t, percent):
    if t >= 24 * 60 * 60:
        print("   %s | %.2f days (%.1f%%)" % (tag, t / 24 / 60 / 60), percent)
    elif t >= 60 * 60:
        print("   %s | %.2f hours (%.1f%%)" % (tag, t / 60 / 60, percent))
    elif t >= 60:
        print("   %s | %.2f minutes (%.1f%%)" % (tag, t / 60, percent))
    else:
        print("   %s | %.2f seconds (%.1f%%)" % (tag, t, percent))


def print_bank(bank, show_content=False):
    tag = bank["tag"]
    size = bank["size"]
    particles = bank["particles"]

    print("\n=============")
    print("Particle bank")
    print("  tag  :", tag)
    print("  size :", size, "of", len(bank["particles"]))
    if show_content and size > 0:
        for i in range(size):
            print(" ", particles[i])
    print("\n")
