import numpy as np
import time
import cProfile, pstats

import mcdc


def build_model(N_particle):
    # materials
    m1 = mcdc.material(capture=np.array([1.0]))
    m2 = mcdc.material(capture=np.array([1.5]))
    m3 = mcdc.material(capture=np.array([2.0]))

    # surfaces
    s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
    s2 = mcdc.surface("plane-z", z=2.0)
    s3 = mcdc.surface("plane-z", z=4.0)
    s4 = mcdc.surface("plane-z", z=6.0, bc="vacuum")

    # cells
    mcdc.cell(+s1 & -s2, m2)
    mcdc.cell(+s2 & -s3, m3)
    mcdc.cell(+s3 & -s4, m1)

    # source
    mcdc.source(z=[0.0, 6.0], isotropic=True)

    # tally
    mcdc.tally.mesh_tally(
        scores=["flux"],
        z=np.linspace(0.0, 6.0, 61),
        mu=np.linspace(-1.0, 1.0, 32 + 1),
    )

    # settings
    mcdc.setting(N_particle=N_particle)


def run_once():
    t0 = time.perf_counter()
    mcdc.run()
    return time.perf_counter() - t0


if __name__ == "__main__":
    # warm-up JIT compiler
    build_model(200)
    _ = run_once()

    # use warmed-up JIT compiler
    profiling = False
    if profiling:
        build_model(1e4)

        pr = cProfile.Profile()
        pr.enable()
        wall = run_once()
        pr.disable()

        # output top 25 cumulative time offenders from results of cProfile
        p = pstats.Stats(pr).sort_stats("cumulative")
        p.print_stats(25)
    else:
        runs = 20
        runtimes = []
        for _ in range(runs):
            build_model(1e4)
            runtimes.append(run_once())
        print(
            f"mean +/- stdev runtime for {runs} runs: {np.mean(runtimes):.3f} +/- {np.std(runtimes):.3f}"
        )

        # 1e3
        # new, no Numba: 0.712s +/- 0.056s
        # new, Numba: 0.302s +/- 0.077s
        # old, no Numba: 2.072s +/- 0.046s
        # old, Numba: 0.329s +/- 0.074

        # 1e4
        # new, no Numba: 6.125s +/- 0.088s
        # new, Numba: 1.989s +/- 0.106s
        # old, no Numba: 20.278s +/- 1.141s
        # old, Numba: 2.167s +/- 0.088s
