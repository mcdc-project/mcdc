"""LAW 66 preserves dimensionless phase space and its mass parameters."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from mcdc.object_.hdf5_reader import read_distribution


@pytest.mark.parametrize("law,density", [(1, [1.0, 1.0, 0.0]), (2, [0.0, 2.0, 0.0])])
def test_ace_nbody_conversion_retains_fraction_units_and_parameters(
    tmp_path, law, density
):
    ACEtk = pytest.importorskip("ACEtk")
    from data_library_generator.common.ace_converter import write_energy_distribution

    source = ACEtk.continuous.NBodyPhaseSpaceDistribution(
        0.0, 20.0, 3, 12.0, law, [0.0, 0.5, 1.0], density, [0.0, 0.5, 1.0]
    )
    with h5py.File(tmp_path / "nbody.h5", "w") as group:
        write_energy_distribution(group, source)
        assert group.attrs["type"] == "n-body"
        assert group["number-particles"][()] == 3
        assert group["total-mass-ratio"][()] == 12.0
        np.testing.assert_array_equal(group["value"], [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(group["pdf"], density)
        assert group["interpolation/laws"].asstr()[()].tolist() == [
            "histogram" if law == 1 else "linear"
        ]
        for name in ("number-particles", "total-mass-ratio", "value", "pdf"):
            assert "unit" not in group[name].attrs
        result = read_distribution(group)
    assert result.number_particles == 3
    assert result.total_mass_ratio == 12.0
    np.testing.assert_array_equal(result.pdf.x, [0, 0.5, 1])
    np.testing.assert_array_equal(result.pdf.y, density)
