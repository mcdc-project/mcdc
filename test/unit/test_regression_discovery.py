"""Regression discovery ignores output folders left behind by branch changes."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_only_input_decks_are_discovered(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "regression" / "conftest.py"
    spec = spec_from_file_location("regression_discovery", path)
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "REGRESSION_DIR", tmp_path)

    for name in (
        "valid",
        "output_only",
        "empty",
        "__pycache__",
        runner.REGRESSION_DATA_NAME,
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "valid" / "input.py").write_text("# Input deck\n")
    (tmp_path / "output_only" / "output.h5").touch()
    (tmp_path / runner.REGRESSION_DATA_NAME / "input.py").touch()

    # A missing answer must still be reported when the actual input deck runs.
    assert runner.regression_cases() == [tmp_path / "valid"]
