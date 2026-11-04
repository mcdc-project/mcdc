"""Synthetic neutron inputs for proton tests that also enable neutron transport."""

import h5py
import pytest

from mcdc.constant import BOLTZMANN_K


@pytest.fixture
def neutron_library(tmp_path):
    """Provide both natural-hydrogen isotopes in the mcdc-data format.

    These tests call proton kernels directly, so the neutron reaction groups
    are empty. The files allow secondary-neutron transport to be configured
    without depending on a downloaded data library.
    """
    for mass in (1, 2):
        with h5py.File(tmp_path / f"neutron-H{mass}-293.6K.h5", "w") as file:
            file.attrs["format"] = "mcdc-data"
            file.attrs["format-version"] = "0.1.0"
            file["atomic-number"] = 1
            file["mass-number"] = mass
            file["atomic-weight-ratio"] = 1.0
            file["excitation-level"] = 0
            file["fissionable"] = False
            file["temperature"] = 293.6
            file["temperature"].attrs["unit"] = "K"
            file["kT"] = 293.6 * BOLTZMANN_K
            file["kT"].attrs["unit"] = "eV"
            file["xs-energy-grid"] = [0.1e6, 1.0e6, 10.0e6]
            file["xs-energy-grid"].attrs["unit"] = "eV"
            file.create_group("reactions")
