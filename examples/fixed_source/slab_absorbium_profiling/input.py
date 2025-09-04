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
    build_model(1e3)
    pr = cProfile.Profile()
    pr.enable()
    wall = run_once()
    pr.disable()
    print(f"Wall time (profiled run): {wall:.3f} s")

    # output top 25 cumulative time offenders from results of cProfile
    p = pstats.Stats(pr).sort_stats("cumulative")
    p.print_stats(25)
