import argparse, os, sys
import importlib.metadata

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MC/DC: Monte Carlo Dynamic Code")
parser.add_argument(
    "--mode",
    type=str,
    help="Run mode",
    choices=["python", "numba", "numba_debug"],
    default="python",
)

parser.add_argument(
    "--target", type=str, help="Target", choices=["cpu", "gpu"], default="cpu"
)

parser.add_argument(
    "--gpu_strat",
    type=str,
    help="Strategy used in GPU execution (event or async).",
    choices=["async", "event"],
    default="event",
)

parser.add_argument(
    "--gpu_block_count",
    type=int,
    help="Number of blocks used in GPU execution.",
    default=240,
)

parser.add_argument(
    "--gpu_arena_size",
    type=int,
    help="Capacity of each intermediate data buffer used, as a particle count.",
    default=0x100000,
)

parser.add_argument(
    "--gpu_rocm_path",
    type=str,
    help="Path to ROCm installation for use in GPU execution.",
    default=None,
)

parser.add_argument(
    "--gpu_cuda_path",
    type=str,
    help="Path to CUDA installation for use in GPU execution.",
    default=None,
)

parser.add_argument(
    "--gpu_share_stride",
    type=int,
    help="Number of gpus that are shared across adjacent ranks.",
    default=1,
)


parser.add_argument("--N_particle", type=int, help="Number of particles")
parser.add_argument("--output", type=str, help="Output file name")
parser.add_argument("--progress_bar", default=True, action="store_true")
parser.add_argument("--no-progress_bar", dest="progress_bar", action="store_false")
parser.add_argument("--clear_cache", action="store_true")
parser.add_argument("--caching", action="store_true")
parser.add_argument("--no_caching", dest="caching", action="store_false")
parser.add_argument("--runtime_output", default=False, action="store_true")
parser.add_argument(
    "--delta_tracking",
    default=False,
    action="store_true",
    help="Set delta tracking to on",
)
parser.set_defaults(caching=False)
args, unargs = parser.parse_known_args()


mode = args.mode
target = args.target
caching = args.caching
clear_cache = args.clear_cache

from mpi4py import MPI

import shutil

src_path = os.path.dirname(os.path.abspath(__file__))
cache_path = f"{src_path}/__pycache__"

if ((caching == False) or (clear_cache == True)) and (MPI.COMM_WORLD.Get_rank() == 0):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists("__harmonize_cache__"):
        shutil.rmtree("__harmonize_cache__")

if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.Barrier()


from mcdc.card import UniverseCard
from mcdc.print_ import (
    print_banner,
    print_msg,
    print_runtime,
    print_header_eigenvalue,
    print_warning,
    print_error,
)
import numba as nb

if mode == "python":
    nb.config.DISABLE_JIT = True
elif mode == "numba":
    nb.config.DISABLE_JIT = False
    nb.config.NUMBA_DEBUG_CACHE = 1
    nb.config.THREADING_LAYER = "workqueue"
elif mode == "numba_debug":
    msg = "\n >> Entering numba debug mode\n >> will result in slower code and longer compile times\n >> to configure debug options see main.py"
    print_warning(msg)

    nb.config.DISABLE_JIT = False  # turns on the jitter
    nb.config.DEBUG = False  # turns on debugging options
    nb.config.NUMBA_FULL_TRACEBACKS = (
        1  # enables errors from sub-packages to be printed
    )
    nb.config.NUMBA_BOUNDSCHECK = 1  # checks bounds errors of vectors
    nb.config.NUMBA_COLOR_SCHEME = (
        "dark_bg"  # prints error messages for dark background terminals
    )
    nb.config.NUMBA_DEBUG_NRT = 1  # Numba run time (NRT) statistics counter
    nb.config.NUMBA_DEBUG_TYPEINFER = (
        1  # print out debugging information about type inference.
    )
    nb.config.NUMBA_ENABLE_PROFILING = 1  # enables profiler use
    nb.config.NUMBA_DUMP_CFG = 1  # prints out a control flow diagram
    nb.config.NUMBA_OPT = 0  # forums un optimized code from compilers
    nb.config.NUMBA_DEBUGINFO = 1  #
    nb.config.NUMBA_EXTEND_VARIABLE_LIFETIMES = (
        1  # allows for inspection of numba variables after end of compilation
    )
