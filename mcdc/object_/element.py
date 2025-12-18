import h5py
import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.reaction import (
    ReactionElectronElasticScattering,
    ReactionElectronExcitation,
    ReactionElectronBremsstrahlung,
    ReactionElectronIonization,
)
from mcdc.print_ import print_1d_array, print_error


# ======================================================================================
# Element
# ======================================================================================


class Element(ObjectNonSingleton):
    # Annotations for Numba mode
    label: str = "element"
    #
    name: str
    atomic_weight_ratio: float
    atomic_number: int
    # XS on xs_energy_grid
    xs_energy_grid: NDArray[float64]
    total_xs: NDArray[float64]
    ionization_xs: NDArray[float64]
    elastic_xs: NDArray[float64]
    excitation_xs: NDArray[float64]
    bremsstrahlung_xs: NDArray[float64]
    # Reactions
    ionization_reactions: list[ReactionElectronIonization]
    elastic_scattering_reactions: list[ReactionElectronElasticScattering]
    excitation_reactions: list[ReactionElectronExcitation]
    bremsstrahlung_reactions: list[ReactionElectronBremsstrahlung]
    # Ionization element attributes
    ionization_subshell_binding_energy: NDArray[float64]

    def __init__(self, element_name: str):
        super().__init__()

        self.name = element_name

        # Set attributes from the hdf5 file
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{element_name}.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        # Basic properties
        self.atomic_weight_ratio = float(file["atomic_weight_ratio"][()])
        self.atomic_number = int(file["atomic_number"][()])

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

        # Energy grid
        xs_energy = file["electron_reactions/xs_energy_grid"][()]
        self.xs_energy_grid = xs_energy

        # The total XS
        self.total_xs = np.zeros_like(self.xs_energy_grid)
        self.elastic_xs = np.zeros_like(self.xs_energy_grid)
        self.excitation_xs = np.zeros_like(self.xs_energy_grid)
        self.bremsstrahlung_xs = np.zeros_like(self.xs_energy_grid)
        self.ionization_xs = np.zeros_like(self.xs_energy_grid)

        xs_containers = [
            self.elastic_xs,
            self.excitation_xs,
            self.bremsstrahlung_xs,
            self.ionization_xs,
        ]
        for xs_container, rx_name in list(zip(xs_containers, rx_names)):
            for MT in MTs[rx_name]:
                xs = file[f"electron_reactions/{rx_name}/{MT}/xs"]
                xs_container[xs.attrs["offset"] :] += xs[()]

        # Total = sum of xs'
        self.total_xs = (
            self.elastic_xs
            + self.excitation_xs
            + self.bremsstrahlung_xs
            + self.ionization_xs
        )

        # ==========================================================================
        # The reactions
        # ==========================================================================

        self.elastic_scattering_reactions = []
        self.excitation_reactions = []
        self.bremsstrahlung_reactions = []
        self.ionization_reactions = []

        rx_containers = [
            self.elastic_scattering_reactions,
            self.excitation_reactions,
            self.bremsstrahlung_reactions,
            self.ionization_reactions,
        ]
        rx_classes = [
            ReactionElectronElasticScattering,
            ReactionElectronExcitation,
            ReactionElectronBremsstrahlung,
            ReactionElectronIonization,
        ]

        for rx_container, rx_name, rx_class in list(zip(rx_containers, rx_names, rx_classes)):
            for MT in MTs[rx_name]:
                h5_group = file[f"electron_reactions/{rx_name}/{MT}"]
                reaction = rx_class.from_h5_group(h5_group)
                rx_container.append(reaction)

        # ==========================================================================
        # Ionization element attributes
        # ==========================================================================

        be_list = []
        if len(MTs["ionization"]) > 0:
            g = file[f"electron_reactions/ionization/{MTs['ionization'][0]}"]
            subshells = g["subshells"]
            subshell_names = list(subshells.keys())
            for name in subshell_names:
                sh = subshells[name]
                be_list.append(float(sh["binding_energy"][()]))

        self.ionization_subshell_binding_energy = np.asarray(be_list)

        file.close()

    def __repr__(self):
        text = "\n"
        text += f"Element\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic number: {self.atomic_number}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        text += f"  - Reaction MTs\n"
        text += f"    - Ionization: {[int(x.MT) for x in self.ionization_reactions]}\n"
        text += f"    - Elastic scattering: {[int(x.MT) for x in self.elastic_scattering_reactions]}\n"
        text += f"    - Excitation: {[int(x.MT) for x in self.excitation_reactions]}\n"
        text += f"    - Bremsstrahlung: {[int(x.MT) for x in self.bremsstrahlung_reactions]}\n"
        text += f"  - Reaction cross-sections (eV, barns)\n"
        text += f"    - Energy grid {print_1d_array(self.xs_energy_grid)}\n"
        text += f"    - Total {print_1d_array(self.total_xs)}\n"
        text += f"    - Ionization {print_1d_array(self.ionization_xs)}\n"
        text += f"    - Elastic scattering {print_1d_array(self.elastic_xs)}\n"
        text += f"    - Excitation {print_1d_array(self.excitation_xs)}\n"
        text += f"    - Bremsstrahlung {print_1d_array(self.bremsstrahlung_xs)}\n"
        return text