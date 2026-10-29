"""ACE LAW 4 splits discrete masses and continuum densities before conversion."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from mcdc.object_.hdf5_reader import read_distribution


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("interpolation", [1, 2])
def test_ace_law4_retains_discrete_probabilities_and_energy_units(
    tmp_path, pure, interpolation
):
    ACEtk = pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import write_energy_distribution

    tables = []
    for incident, line, mass in [
        (1.0, 2e-6, 1.0 if pure else 0.2),
        (3.0, 8e-6, 1.0 if pure else 0.6),
    ]:
        tables.append(
            ACEtk.continuous.TabulatedEnergyDistribution(
                incident,
                interpolation,
                [line] if pure else [line, 0.0, 1e-5],
                [mass] if pure else [mass, (1 - mass) / 1e-5, (1 - mass) / 1e-5],
                [mass] if pure else [mass, mass, 1.0],
                1,
            )
        )
    source = ACEtk.continuous.OutgoingEnergyDistributionData(tables)
    with h5py.File(tmp_path / "law4.h5", "w") as group:
        write_energy_distribution(group, source)
        np.testing.assert_allclose(group["grid"], [1e6, 3e6])
        np.testing.assert_allclose(group["tables/discrete-value"], [2, 8])
        np.testing.assert_allclose(group["tables/pmf"], [1, 1] if pure else [0.2, 0.6])
        assert group["tables/discrete-value"].attrs["unit"] == "eV"
        assert "unit" not in group["tables/pmf"].attrs
        if not pure:
            np.testing.assert_allclose(group["tables/pdf"], [0.08, 0.08, 0.04, 0.04])
            assert group["tables/pdf"].attrs["unit"] == "1/eV"
        result = read_distribution(group)
    np.testing.assert_allclose(
        result.continuous_probability, [0, 0] if pure else [0.8, 0.4]
    )
