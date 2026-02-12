import h5py
import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.data import DataBase, DataPolynomial, DataTable
from mcdc.object_.distribution import DistributionBase
from mcdc.object_.reaction import (
    ReactionNeutronCapture,
    ReactionNeutronElasticScattering,
    ReactionNeutronFission,
    ReactionNeutronInelasticScattering,
    decode_type,
    set_energy_distribution,
)
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Nuclide
# ======================================================================================


class Nuclide(ObjectNonSingleton):
    # Annotations for Numba mode
    label: str = "nuclide"
    #
    name: str
    temperature: float
    atomic_weight_ratio: float
    fissionable: bool
    excitation_level: int
    xs_energy_grid: NDArray[float64]
    total_xs: NDArray[float64]
    elastic_xs: NDArray[float64]
    capture_xs: NDArray[float64]
    inelastic_xs: NDArray[float64]
    fission_xs: NDArray[float64]
    elastic_scattering_reactions: list[ReactionNeutronElasticScattering]
    capture_reactions: list[ReactionNeutronCapture]
    inelastic_scattering_reactions: list[ReactionNeutronInelasticScattering]
    fission_reactions: list[ReactionNeutronFission]
    fission_prompt_multiplicity: DataBase
    fission_delayed_multiplicity: DataBase
    N_fission_delayed_precursor: int
    fission_delayed_fractions: NDArray[float64]
    fission_delayed_decay_rates: NDArray[float64]
    fission_delayed_spectra: list[DistributionBase]

    def __init__(self, nuclide_name, temperature):
        super().__init__()

        self.name = nuclide_name
        self.temperature = temperature

        # Set attributes from the hdf5 file
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{nuclide_name}-{temperature}K.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        # Basic properties
        self.atomic_weight_ratio = file["atomic_weight_ratio"][()]
        self.fissionable = bool(file["fissionable"][()])
        self.excitation_level = int(file["excitation_level"][()])

        # The reactions
        rx_names = [
            "elastic_scattering",
            "capture",
            "inelastic_scattering",
            "fission",
        ]

        # The reaction MTs
        MTs = {}
        for name in rx_names:
            if name not in file["neutron_reactions"]:
                MTs[name] = []
                continue

            MTs[name] = [
                x for x in file[f"neutron_reactions/{name}"] if x.startswith("MT")
            ]

        # ==========================================================================
        # Reaction XS
        # ==========================================================================

        # Energy grid
        xs_energy = file["neutron_reactions/xs_energy_grid"][()] * 1e6  # MeV to eV
        self.xs_energy_grid = xs_energy

        # The total XS
        self.total_xs = np.zeros_like(self.xs_energy_grid)
        self.elastic_xs = np.zeros_like(self.xs_energy_grid)
        self.capture_xs = np.zeros_like(self.xs_energy_grid)
        self.inelastic_xs = np.zeros_like(self.xs_energy_grid)
        self.fission_xs = np.zeros_like(self.xs_energy_grid)

        xs_containers = [
            self.elastic_xs,
            self.capture_xs,
            self.inelastic_xs,
            self.fission_xs,
        ]
        for xs_container, rx_name in list(zip(xs_containers, rx_names)):
            for MT in MTs[rx_name]:
                xs = file[f"neutron_reactions/{rx_name}/{MT}/xs"]
                xs_container[xs.attrs["offset"] :] += xs[()]

        self.total_xs = (
            self.elastic_xs + self.capture_xs + self.inelastic_xs + self.fission_xs
        )

        # ==========================================================================
        # The reactions
        # ==========================================================================

        self.elastic_scattering_reactions = []
        self.capture_reactions = []
        self.inelastic_scattering_reactions = []
        self.fission_reactions = []

        rx_containers = [
            self.elastic_scattering_reactions,
            self.capture_reactions,
            self.inelastic_scattering_reactions,
            self.fission_reactions,
        ]
        rx_classes = [
            ReactionNeutronElasticScattering,
            ReactionNeutronCapture,
            ReactionNeutronInelasticScattering,
            ReactionNeutronFission,
        ]
        for rx_container, rx_name, rx_class in list(
            zip(rx_containers, rx_names, rx_classes)
        ):
            for MT in MTs[rx_name]:
                h5_group = file[f"neutron_reactions/{rx_name}/{MT}"]
                reaction = rx_class.from_h5_group(h5_group)
                rx_container.append(reaction)

        # ==============================================================================
        # Fission nuclide attributes
        # ==============================================================================

        if not self.fissionable:
            self.fission_prompt_multiplicity = simulation.data[0]
            self.fission_delayed_multiplicity = simulation.data[0]
            self.N_fission_delayed_precursor = 0
            self.fission_delayed_fractions = np.zeros(0)
            self.fission_delayed_decay_rates = np.zeros(0)
            self.fission_delayed_spectra = []
        else:
            fission_group = file["neutron_reactions/fission"]

            # Multiplicities
            self.fission_prompt_multiplicity = set_fission_multiplicity(
                fission_group["prompt_multiplicity"]
            )
            self.fission_delayed_multiplicity = set_fission_multiplicity(
                fission_group["delayed_multiplicity"]
            )

            # Delayed fractions and decay rates
            self.fission_delayed_fractions = fission_group[
                "delayed_neutron_precursors/fractions"
            ][()]
            self.fission_delayed_decay_rates = fission_group[
                "delayed_neutron_precursors/decay_rates"
            ][()]
            self.N_fission_delayed_precursor = len(self.fission_delayed_fractions)

            # Delayed spectra
            self.fission_delayed_spectra = []
            spectrum_names = [
                x
                for x in fission_group["delayed_neutron_precursors"]
                if x.startswith("energy_spectrum-")
            ]
            for spectrum_name in spectrum_names:
                self.fission_delayed_spectra.append(
                    set_energy_distribution(
                        fission_group[f"delayed_neutron_precursors/{spectrum_name}"]
                    )
                )

        file.close()

    def __repr__(self):
        text = "\n"
        text += f"Nuclide\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        text += f"  - Reaction MTs\n"
        text += f"    - Elastic scattering: {[int(x.MT) for x in self.elastic_scattering_reactions]}\n"
        text += f"    - Capture: {[int(x.MT) for x in self.capture_reactions]}\n"
        text += f"    - Inelastic scattering: {[int(x.MT) for x in self.inelastic_scattering_reactions]}\n"
        if self.fissionable:
            text += f"    - Fission: {[int(x.MT) for x in self.fission_reactions]}\n"
        text += f"  - Reaction cross-sections (eV, barns)\n"
        text += f"    - Energy grid {print_1d_array(self.xs_energy_grid)}\n"
        text += f"    - Total {print_1d_array(self.total_xs)}\n"
        text += f"    - Elastic scattering {print_1d_array(self.elastic_xs)}\n"
        text += f"    - Capture {print_1d_array(self.capture_xs)}\n"
        text += f"    - Inelastic scattering {print_1d_array(self.inelastic_xs)}\n"
        if self.fissionable:
            text += f"    - Fission {print_1d_array(self.fission_xs)}\n"
        return text


# ======================================================================================
# Helper functions
# ======================================================================================


def set_fission_multiplicity(h5_group):
    multiplicity_type = h5_group.attrs["type"]

    if multiplicity_type == "tabulated":
        x = h5_group["energy"][()] * 1e6  # MeV to eV
        y = h5_group["value"][()]
        multiplicity = DataTable(x, y)

    elif multiplicity_type == "polynomial":
        coefficient = h5_group["coefficient"][()]

        # MeV-based to eV-based
        for l in range(len(coefficient)):
            coefficient[l] /= 1e6**l

        multiplicity = DataPolynomial(coefficient)
    else:
        print_error(f"Unsupported multiplicity of type {multiplicity_type}")

    return multiplicity
