"""Process-wide command-line and execution configuration for MC/DC.

Importing this module parses MC/DC's known command-line arguments, manages
generated-code caches, and configures Numba for the selected execution mode.
Simulation-specific command-line overrides are applied later, at the beginning
of :meth:`mcdc.Simulation.compile`.
"""

import argparse
import shutil
from pathlib import Path

from mpi4py import MPI

# ======================================================================================
# Command-line interface
# ======================================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Create the MC/DC command-line argument parser."""
    parser = argparse.ArgumentParser(description="MC/DC: Monte Carlo Dynamic Code")

    # Execution mode and target
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

    # Simulation-setting overrides
    parser.add_argument("--N_particle", type=int, help="Number of particles")
    parser.add_argument("--N_batch", type=int, help="Number of batches")
    parser.add_argument("--output", type=str, help="Output file name")
    parser.add_argument("--progress_bar", default=True, action="store_true")
    parser.add_argument("--no-progress_bar", dest="progress_bar", action="store_false")
    parser.add_argument("--runtime_output", default=False, action="store_true")

    # Numba compilation and cache behavior
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--caching", action="store_true", default=False)
    parser.add_argument("--no_caching", dest="caching", action="store_false")
    parser.add_argument(
        "-r",
        "--rebuild",
        action="store_true",
        help="Rebuild generated Numba support (for active object model development).",
    )

    # GPU execution
    parser.add_argument(
        "--gpu_state_storage",
        type=str,
        help="GPU state-storage strategy.",
        choices=["separate", "managed", "united"],
        default="separate",
    )
    parser.add_argument(
        "--gpu_strategy",
        type=str,
        help="GPU scheduling strategy.",
        choices=["async", "event"],
        default="event",
    )
    parser.add_argument(
        "--gpu_block_count",
        type=int,
        help="Number of GPU blocks.",
        default=240,
    )
    parser.add_argument(
        "--gpu_arena_size",
        type=int,
        help="Particle capacity of each intermediate GPU data buffer.",
        default=0x100000,
    )
    parser.add_argument(
        "--gpu_rocm_path",
        type=str,
        help="Path to the ROCm installation.",
        default=None,
    )
    parser.add_argument(
        "--gpu_cuda_path",
        type=str,
        help="Path to the CUDA installation.",
        default=None,
    )
    parser.add_argument(
        "--gpu_share_stride",
        type=int,
        help="Number of adjacent MPI ranks sharing each GPU.",
        default=1,
    )

    return parser


# Preserve these module-level names because execution and GPU modules consume them.
parser = _build_parser()
# Ignore unrelated arguments supplied by pytest, notebooks, or outer Python drivers.
args, unargs = parser.parse_known_args()

mode = args.mode
target = args.target
gpu_state_storage = args.gpu_state_storage
caching = args.caching
clear_cache = args.clear_cache


# ======================================================================================
# Simulation-setting overrides
# ======================================================================================


def override_settings(simulation) -> bool:
    """Apply command-line overrides before compiling a simulation snapshot."""
    settings = simulation.settings
    changed = False

    def set_setting(name, value):
        nonlocal changed
        if value is None or getattr(settings, name) == value:
            return
        setattr(settings, name, value)
        changed = True

    # These command-line options directly replace public Simulation settings.
    set_setting("N_particle", args.N_particle)
    set_setting("N_batch", args.N_batch)
    set_setting("output_name", args.output)
    set_setting("use_progress_bar", args.progress_bar)

    # GPU names are translated into the integer constants stored at runtime.
    if target == "gpu":
        from mcdc.constant import (
            GPU_STORAGE_MANAGED,
            GPU_STORAGE_SEPARATE,
            GPU_STORAGE_UNITED,
            GPU_STRATEGY_ASYNC,
            GPU_STRATEGY_EVENT,
        )

        strategy = {
            "async": GPU_STRATEGY_ASYNC,
            "event": GPU_STRATEGY_EVENT,
        }[args.gpu_strategy]
        storage = {
            "separate": GPU_STORAGE_SEPARATE,
            "managed": GPU_STORAGE_MANAGED,
            "united": GPU_STORAGE_UNITED,
        }[args.gpu_state_storage]

        set_setting("gpu_strategy", strategy)
        set_setting("gpu_storage", storage)

    return changed


def rebuild_numba_support_if_requested() -> None:
    """Rebuild generated Numba support during package initialization."""
    if not args.rebuild:
        return

    communicator = MPI.COMM_WORLD
    if communicator.Get_rank() == 0:
        from mcdc.code_factory.numba_layers_generator import rebuild_numba_support

        rebuild_numba_support()

    if communicator.Get_size() > 1:
        communicator.Barrier()


# ======================================================================================
# Process-wide initialization
# ======================================================================================


def _manage_runtime_caches() -> None:
    """Clear generated-code caches when caching is disabled or reset."""
    should_clear = not caching or clear_cache
    if should_clear and MPI.COMM_WORLD.Get_rank() == 0:
        cache_directories = (Path.cwd() / "__harmonize_cache__",)
        for cache_directory in cache_directories:
            if cache_directory.exists():
                shutil.rmtree(cache_directory)

    # Other ranks must not use a cache while the root rank is removing it.
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()


def _configure_numba() -> None:
    """Configure Numba for Python, compiled, or diagnostic execution."""
    import numba as nb

    if mode == "python":
        nb.config.DISABLE_JIT = True
        return

    nb.config.DISABLE_JIT = False

    if mode == "numba":
        nb.config.NUMBA_DEBUG_CACHE = 1
        nb.config.THREADING_LAYER = "workqueue"
        return

    from mcdc.print_ import print_warning

    print_warning(
        "\n >> Entering Numba debug mode"
        "\n >> This mode is slower and enables additional diagnostics"
    )

    # Runtime checks and diagnostic output
    nb.config.DEBUG = False
    nb.config.NUMBA_FULL_TRACEBACKS = 1
    nb.config.NUMBA_BOUNDSCHECK = 1
    nb.config.NUMBA_COLOR_SCHEME = "dark_bg"
    nb.config.NUMBA_DEBUG_NRT = 1
    nb.config.NUMBA_DEBUG_TYPEINFER = 1

    # Generated-code inspection and debugger support
    nb.config.NUMBA_ENABLE_PROFILING = 1
    nb.config.NUMBA_DUMP_CFG = 1
    nb.config.NUMBA_OPT = 0
    nb.config.NUMBA_DEBUGINFO = 1
    nb.config.NUMBA_EXTEND_VARIABLE_LIFETIMES = 1


_manage_runtime_caches()
_configure_numba()
