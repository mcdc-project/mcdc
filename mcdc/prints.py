import numba as nb
import sys

from colorama import Fore, Style
from mpi4py import MPI

####

import mcdc.objects as objects


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


def print_eigenvalue_header():
    if objects.settings.use_gyration_radius:
        print("\n #     k        GyRad.  k (avg)            ")
        print(" ====  =======  ======  ===================")
    else:
        print("\n #     k        k (avg)            ")
        print(" ====  =======  ===================")
    sys.stdout.flush()


def print_batch_header(i, N):
    print(f"\nBatch {i}/{N}")
    sys.stdout.flush()
