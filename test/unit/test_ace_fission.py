"""Fission preparation preserves precursor identities and converts time units."""

from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ACEtk = pytest.importorskip("ACEtk")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from data_library_generator.common.ace_converter import write_fission


def test_precursor_groups_retain_their_fractions_spectra_and_decay_rates(tmp_path):
    """Distinct groups keep their own spectra and use inverse seconds in HDF5."""
    polynomial = ACEtk.continuous.PolynomialFissionMultiplicity
    spectra = [
        ACEtk.continuous.OutgoingEnergyDistributionData(
            [
                ACEtk.continuous.TabulatedEnergyDistribution(
                    incident, 2, [0.0, upper], [1.0 / upper] * 2, [0.0, 1.0]
                )
                for incident in (1.0, 20.0)
            ]
        )
        for upper in (1.0, 2.0)
    ]
    ace = SimpleNamespace(
        fission_multiplicity_block=ACEtk.continuous.FissionMultiplicityBlock(
            polynomial([2.4]), polynomial([2.5])
        ),
        delayed_fission_multiplicity_block=SimpleNamespace(
            multiplicity=polynomial([0.1])
        ),
        delayed_neutron_precursor_block=ACEtk.continuous.DelayedNeutronPrecursorBlock(
            [
                ACEtk.continuous.DelayedNeutronPrecursorData(
                    i + 1, rate, [0.0, 20.0], [fraction] * 2
                )
                for i, (rate, fraction) in enumerate([(1e-10, 0.25), (1e-9, 0.75)])
            ]
        ),
        delayed_neutron_energy_distribution_block=SimpleNamespace(
            energy_distribution_data=lambda i: spectra[i - 1]
        ),
    )
    with h5py.File(tmp_path / "fission.h5", "w") as file:
        write_fission(file, ace)
        delayed = file["delayed-neutrons"]
        np.testing.assert_allclose(delayed["fractions"], [0.25, 0.75])
        np.testing.assert_allclose(delayed["decay-rates"], [0.01, 0.1])
        assert delayed["decay-rates"].attrs["unit"] == "1/s"
        for i, upper in enumerate([1e6, 2e6]):
            np.testing.assert_allclose(
                delayed[f"energy-distribution-{i}/tables/value"], [0.0, upper] * 2
            )
