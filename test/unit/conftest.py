import os


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        choices=["python", "numba"],
        default="python",
        help="MCDC execution mode for unit tests.",
    )


def pytest_configure(config):
    mode = config.getoption("--mode")
    if mode == "python":
        os.environ["NUMBA_DISABLE_JIT"] = "1"
    else:
        os.environ.pop("NUMBA_DISABLE_JIT", None)


def pytest_report_header(config):
    return f"MCDC mode: {config.getoption('--mode')}"
