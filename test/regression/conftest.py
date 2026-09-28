import fnmatch
import subprocess
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

REGRESSION_DIR = Path(__file__).parent
REGRESSION_DATA_NAME = "mcdc-regression_test_data"
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-14
NON_TEST_DIRS = {"__pycache__", REGRESSION_DATA_NAME}


def regression_cases():
    return sorted(
        path
        for path in REGRESSION_DIR.iterdir()
        if path.is_dir() and path.name not in NON_TEST_DIRS
    )


def selected_cases(config):
    name_pattern = config.getoption("--name")
    skip_pattern = config.getoption("--skip")
    target = config.getoption("--target")

    cases = []
    for case in regression_cases():
        if name_pattern != "ALL" and not fnmatch.fnmatch(case.name, name_pattern):
            continue
        if skip_pattern != "NONE" and fnmatch.fnmatch(case.name, skip_pattern):
            continue
        if target == "gpu" and ("iqmc" in case.name or "eigenvalue" in case.name):
            continue
        cases.append(case)
    return cases


def pytest_configure(config):
    name_pattern = config.getoption("--name")
    if name_pattern == "ALL":
        return
    if not any(fnmatch.fnmatch(case.name, name_pattern) for case in regression_cases()):
        raise pytest.UsageError(
            f"--name={name_pattern} did not match any regression case"
        )


def pytest_generate_tests(metafunc):
    if "case_path" in metafunc.fixturenames:
        cases = selected_cases(metafunc.config)
        metafunc.parametrize("case_path", cases, ids=[case.name for case in cases])


@pytest.fixture(scope="session")
def regression_data():
    data_path = REGRESSION_DIR / REGRESSION_DATA_NAME
    if data_path.is_dir():
        result = subprocess.run(
            ["git", "pull"],
            cwd=data_path,
            capture_output=True,
            text=True,
        )
        # An out-of-date clone is still usable (e.g. when running offline), so warn
        # rather than skip.
        if result.returncode != 0:
            warnings.warn(
                "Could not update regression data repository.\n"
                + format_subprocess_output(result.stdout, result.stderr)
            )
        return data_path

    result = subprocess.run(
        ["git", "clone", f"https://github.com/mcdc-project/{REGRESSION_DATA_NAME}.git"],
        cwd=REGRESSION_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "Could not clone regression data repository.\n"
            + format_subprocess_output(result.stdout, result.stderr)
        )
    return data_path


@pytest.fixture
def run_regression_case(pytestconfig, regression_data):
    def run(case_path):
        run_case(pytestconfig, case_path)

    return run


def run_case(config, case_path):
    input_path = case_path / "input.py"
    answer_path = case_path / "answer.h5"
    output_path = case_path / "output.h5"

    if not input_path.exists():
        pytest.fail("input.py is missing")
    if not answer_path.exists():
        pytest.fail("answer.h5 is missing")

    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        build_command(config),
        cwd=case_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Run failed with return code {result.returncode}.\n"
            + format_subprocess_output(result.stdout, result.stderr)
        )

    if not output_path.exists():
        pytest.fail(
            "Run did not produce output.h5.\n"
            + format_subprocess_output(result.stdout, result.stderr)
        )

    compare_outputs(output_path, answer_path, config.getoption("--target"))


def build_command(config):
    mode = config.getoption("--mode")
    target = config.getoption("--target")
    mpiexec = config.getoption("--mpiexec")
    srun = config.getoption("--srun")
    command = [
        sys.executable,
        "input.py",
        f"--mode={mode}",
        f"--target={target}",
        "--output=output",
        "--no-progress-bar",
    ]

    if mpiexec > 1:
        prefix = ["mpiexec", "-n", str(mpiexec)]
        if target == "gpu":
            prefix.append("--gpus-per-task=1")
        return [*prefix, *command]

    if srun > 1:
        prefix = ["srun", "-n", str(srun)]
        if target == "gpu":
            prefix.append("--gpus-per-task=1")
        return [*prefix, *command]

    return command


def compare_outputs(output_path, answer_path, target):
    errors = []
    with h5py.File(output_path, "r") as output, h5py.File(answer_path, "r") as answer:
        if "iqmc" in output.keys():
            compare_iqmc(output, answer, errors)
        else:
            compare_tallies(output, answer, target, errors)
            compare_k_results(output, answer, errors)

    if errors:
        pytest.fail("\n\n".join(errors))


def compare_tallies(output, answer, target, errors):
    name_root = "tallies"
    for tally in answer[name_root].keys():
        name_tally = f"{name_root}/{tally}"
        for score in answer[name_tally].keys():
            if score == "grid":
                continue
            name_score = f"{name_tally}/{score}"
            for result in answer[name_score].keys():
                if "uq_var" in result and target == "gpu":
                    continue
                name = f"{name_score}/{result}"
                assert_allclose(output[name][()], answer[name][()], name, errors)


def compare_k_results(output, answer, errors):
    for name in ["k_mean", "k_sdev", "k_cycle", "k_eff"]:
        if name in output.keys():
            assert_allclose(output[name][()], answer[name][()], name, errors)


def compare_iqmc(output, answer, errors):
    for score in output["iqmc/tally/"].keys():
        name = f"iqmc/tally/{score}/mean"
        assert_allclose(
            np.squeeze(output[name][()]),
            np.squeeze(answer[name][()]),
            name,
            errors,
        )


def assert_allclose(actual, expected, name, errors):
    try:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
    except AssertionError as error:
        errors.append(f"Differences in {name}\n{error}")


def format_subprocess_output(stdout, stderr):
    return f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
