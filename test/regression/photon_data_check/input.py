"""Exercise photon mcdc-data with three independent element samples."""

import numpy as np
import mcdc
import os

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/data_check/"

# ======================================================================================
# Representative elements and photon energies
# ======================================================================================

CASES = ("H", "Mg", "Si")
ENERGIES = [100.0, 1300.0, 1320.0, 1830.0, 1860.0, 1.0e5, 3.0e6, 20.0e6]


def build_simulation():
    """Build isolated slabs with sources around absorption edges and at MeV energies."""
    simulation = mcdc.Simulation("Three-element photon mcdc-data")
    cells, sources = [], []
    for i, name in enumerate(CASES):
        material = mcdc.Material(name=name, element_composition={name: 0.05})
        left = mcdc.Surface.PlaneX(x=10.0 * i, boundary_condition="vacuum")
        right = mcdc.Surface.PlaneX(x=10.0 * i + 5.0, boundary_condition="vacuum")
        cells.append(mcdc.Cell(+left & -right, fill=material))
        sources.append(
            mcdc.Source(
                particle_type="photon",
                position=[10.0 * i + 2.5, 0.0, 0.0],
                isotropic=True,
                discrete_energy=(ENERGIES, [1.0] * len(ENERGIES)),
            )
        )
    simulation.set_model(cells)
    simulation.set_sources(sources)
    simulation.set_tallies(
        [
            mcdc.Tally(
                scores=[score],
                particle_type="photon",
                mesh=mcdc.MeshStructured(x=np.arange(0.0, 31.0, 5.0)),
                energy=np.geomspace(1.0, 21.0e6, 81),
            )
            for score in ("flux", "energy_deposition")
        ]
    )
    simulation.settings.N_particle = 3000
    simulation.settings.N_batch = 2
    simulation.settings.time_boundary = 10.0
    simulation.settings.active_bank_buffer = 1000
    return simulation


if __name__ == "__main__":
    build_simulation().run()
