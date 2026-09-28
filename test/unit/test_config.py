from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mcdc
import mcdc.config as config
import mcdc.code_factory.numba_layers_generator as numba_layers_generator

from mcdc.config import _build_parser, override_settings


class SingleRankCommunicator:
    def __init__(self, size=1):
        self.size = size
        self.barrier_count = 0

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self.size

    def Barrier(self):
        self.barrier_count += 1


def test_parser_accepts_numba_support_rebuild():
    parser = _build_parser()

    assert parser.parse_args(["--rebuild"]).rebuild
    assert parser.parse_args(["-r"]).rebuild


def test_package_import_calls_configured_rebuild_gate():
    check_trigger = """
import sys
import types

config = types.ModuleType("mcdc.config")
config.rebuild_numba_support_if_requested = lambda: print("rebuild-gate-called")
sys.modules["mcdc.config"] = config

import mcdc
"""

    result = subprocess.run(
        [sys.executable, "-c", check_trigger],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rebuild-gate-called" in result.stdout


def test_requested_numba_support_rebuild_runs_on_master(monkeypatch):
    rebuild_count = 0
    communicator = SingleRankCommunicator(size=2)

    def rebuild():
        nonlocal rebuild_count
        rebuild_count += 1

    monkeypatch.setattr(config, "MPI", SimpleNamespace(COMM_WORLD=communicator))
    monkeypatch.setattr(config.args, "rebuild", True)
    monkeypatch.setattr(numba_layers_generator, "rebuild_numba_support", rebuild)

    config.rebuild_numba_support_if_requested()

    assert rebuild_count == 1
    assert communicator.barrier_count == 1


def test_unrequested_numba_support_rebuild_does_nothing(monkeypatch):
    communicator = SingleRankCommunicator(size=2)

    monkeypatch.setattr(config, "MPI", SimpleNamespace(COMM_WORLD=communicator))
    monkeypatch.setattr(config.args, "rebuild", False)

    config.rebuild_numba_support_if_requested()

    assert communicator.barrier_count == 0


def test_cache_cleanup_does_not_remove_python_cache(monkeypatch):
    removed = []
    communicator = SingleRankCommunicator()

    monkeypatch.setattr(config, "MPI", SimpleNamespace(COMM_WORLD=communicator))
    monkeypatch.setattr(config, "caching", False)
    monkeypatch.setattr(config, "clear_cache", False)
    monkeypatch.setattr(config.Path, "exists", lambda path: True)
    monkeypatch.setattr(config.shutil, "rmtree", removed.append)

    config._manage_runtime_caches()

    assert removed == [Path.cwd() / "__harmonize_cache__"]
    assert all(path.name != "__pycache__" for path in removed)


def test_compilation_applies_command_line_overrides(monkeypatch):
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    monkeypatch.setattr(config, "target", "cpu")
    monkeypatch.setattr(config.args, "N_particle", 100)
    monkeypatch.setattr(config.args, "N_batch", None)
    monkeypatch.setattr(config.args, "output", None)
    monkeypatch.setattr(
        config.args, "progress_bar", simulation.settings.use_progress_bar
    )

    simulation.compile()

    assert simulation.settings.N_particle == 100
    assert not override_settings(simulation)
