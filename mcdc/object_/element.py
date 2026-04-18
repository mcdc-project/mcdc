import h5py
import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.electron_reaction import (
    ElectronReactionBremsstrahlung,
    ElectronReactionElasticScattering,
    ElectronReactionExcitation,
    ElectronReactionIonization,
)


class Element(ObjectNonSingleton):
    # Annotations for Numba mode
    label: str = "element"
    #
    name: str
    atomic_weight_ratio: float
    atomic_number: int
    #
    electron_xs_energy_grid: NDArray[float64]
    electron_total_xs: NDArray[float64]
    electron_ionization_xs: NDArray[float64]
    electron_elastic_xs: NDArray[float64]
    electron_excitation_xs: NDArray[float64]
    electron_bremsstrahlung_xs: NDArray[float64]
    #
    electron_ionization_reactions: list[ElectronReactionIonization]
    electron_elastic_scattering_reactions: list[ElectronReactionElasticScattering]
    electron_excitation_reactions: list[ElectronReactionExcitation]
    electron_bremsstrahlung_reactions: list[ElectronReactionBremsstrahlung]
    #
    electron_ionization_subshell_binding_energy: NDArray[float64]

    def __init__(self, element_name: str):
        super().__init__()

        self.name = element_name

        # Basic properties
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{element_name}.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")
        self.atomic_weight_ratio = float(file["atomic_weight_ratio"][()])
        self.atomic_number = int(file["atomic_number"][()])
        file.close()

    def set_electron_data(self):
        element_name = self.name

        # Load data library
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{element_name}.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        # The reactions
        rx_names = [
            "elastic_scattering",
            "excitation",
            "bremsstrahlung",
            "ionization",
        ]

        # The reaction MTs
        MTs = {}
        for name in rx_names:
            if name not in file["electron_reactions"]:
                MTs[name] = []
                continue

            MTs[name] = [
                x for x in file[f"electron_reactions/{name}"] if x.startswith("MT")
            ]

        # ==========================================================================
        # Reaction XS
        # ==========================================================================

        self.electron_xs_energy_grid = file["electron_reactions/xs_energy_grid"][()]
        def load_xs_dataset(dataset):
            values = np.zeros_like(self.electron_xs_energy_grid)
            offset = int(dataset.attrs.get("offset", 0))
            data = dataset[()]
            values[offset : offset + len(data)] = data
            return values

        def load_reaction_total_xs(rx_name):
            total_path = f"electron_reactions/{rx_name}/total/xs"
            if total_path in file:
                return load_xs_dataset(file[total_path])

            values = np.zeros_like(self.electron_xs_energy_grid)
            for MT in MTs[rx_name]:
                xs = file[f"electron_reactions/{rx_name}/{MT}/xs"]
                offset = int(xs.attrs.get("offset", 0))
                data = xs[()]
                values[offset : offset + len(data)] += data
            return values

        self.electron_elastic_xs = load_reaction_total_xs("elastic_scattering")
        self.electron_excitation_xs = load_reaction_total_xs("excitation")
        self.electron_bremsstrahlung_xs = load_reaction_total_xs("bremsstrahlung")
        self.electron_ionization_xs = load_reaction_total_xs("ionization")

        total_path = "electron_reactions/total/xs"
        if total_path in file:
            self.electron_total_xs = load_xs_dataset(file[total_path])
        else:
            self.electron_total_xs = (
                self.electron_elastic_xs
                + self.electron_excitation_xs
                + self.electron_bremsstrahlung_xs
                + self.electron_ionization_xs
            )

        # ==========================================================================
        # The reactions
        # ==========================================================================

        self.electron_elastic_scattering_reactions = []
        self.electron_excitation_reactions = []
        self.electron_bremsstrahlung_reactions = []
        self.electron_ionization_reactions = []

        rx_containers = [
            self.electron_elastic_scattering_reactions,
            self.electron_excitation_reactions,
            self.electron_bremsstrahlung_reactions,
            self.electron_ionization_reactions,
        ]
        rx_classes = [
            ElectronReactionElasticScattering,
            ElectronReactionExcitation,
            ElectronReactionBremsstrahlung,
            ElectronReactionIonization,
        ]
        for rx_container, rx_name, rx_class in list(
            zip(rx_containers, rx_names, rx_classes)
        ):
            for MT in MTs[rx_name]:
                h5_group = file[f"electron_reactions/{rx_name}/{MT}"]
                rx_container.append(rx_class.from_h5_group(h5_group))

        # ==========================================================================
        # Ionization element attributes
        # ==========================================================================

        binding_energy = []
        if len(MTs["ionization"]) > 0:
            h5_group = file[f"electron_reactions/ionization/{MTs['ionization'][0]}"]
            for name in h5_group["subshells"]:
                subshell = h5_group[f"subshells/{name}"]
                binding_energy.append(float(subshell["binding_energy"][()]))

        self.electron_ionization_subshell_binding_energy = np.asarray(binding_energy)

        file.close()

    def __repr__(self):
        text = "\n"
        text += f"Element\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic number: {self.atomic_number}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        return text
