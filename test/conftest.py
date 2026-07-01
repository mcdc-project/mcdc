def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        choices=["python", "numba"],
        default="python",
        help="MCDC execution mode.",
    )
    parser.addoption(
        "--target",
        choices=["cpu", "gpu"],
        default="cpu",
        help="MCDC regression target.",
    )
    parser.addoption(
        "--mpiexec",
        type=int,
        default=0,
        help="Run regression cases with mpiexec and the given number of ranks.",
    )
    parser.addoption(
        "--srun",
        type=int,
        default=0,
        help="Run regression cases with srun and the given number of ranks.",
    )
    parser.addoption(
        "--name",
        default="ALL",
        help="Regression case name pattern to run.",
    )
    parser.addoption(
        "--skip",
        default="NONE",
        help="Regression case name pattern to skip.",
    )
