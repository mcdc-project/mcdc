"""Exercise neutron mcdc-data with seven independent material samples."""

import numpy as np
import mcdc
import os

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/data_check/"

# ======================================================================================
# Representative nuclides and temperatures
# ======================================================================================

CASES = (
    ("H2", 293.6),
    ("H3", 293.6),
    ("Co58m1", 293.6),
    ("Ta182", 1200.0),
    ("Ra223", 293.6),
    ("Si32", 293.6),
    ("U240", 293.6),
)


def build_simulation():
    """Build isolated slabs with sources spanning thermal through fast energies."""
    simulation = mcdc.Simulation("Seven-nuclide mcdc-data")
    cells, sources = [], []
    for i, (name, temperature) in enumerate(CASES):
        material = mcdc.Material(
            name=name, nuclide_composition={name: 0.02}, temperature=temperature
        )
        left = mcdc.Surface.PlaneX(x=10.0 * i, boundary_condition="vacuum")
        right = mcdc.Surface.PlaneX(x=10.0 * i + 5.0, boundary_condition="vacuum")
        cells.append(mcdc.Cell(+left & -right, fill=material))
        sources.append(
            mcdc.Source(
                position=[10.0 * i + 2.5, 0.0, 0.0],
                isotropic=True,
                discrete_energy=([0.0253, 1.0e3, 1.0e6, 14.0e6, 19.0e6], [1.0] * 5),
            )
        )
    simulation.set_model(cells)
    simulation.set_sources(sources)
    simulation.set_tallies(
        [
            mcdc.Tally(
                scores=["flux", "fission"],
                mesh=mcdc.MeshStructured(x=np.arange(0.0, 71.0, 5.0)),
                energy=np.geomspace(1.0e-5, 20.0e6, 81),
            )
        ]
    )
    simulation.settings.N_particle = 7000
    simulation.settings.N_batch = 2
    simulation.settings.time_boundary = 10.0
    simulation.settings.active_bank_buffer = 1000
    return simulation


if __name__ == "__main__":
    build_simulation().run()
