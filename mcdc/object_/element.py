import h5py
import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.object_.base import MCDCObject
from mcdc.object_.electron_reaction import (
    ElectronReactionBremsstrahlung,
    ElectronReactionElasticScattering,
    ElectronReactionExcitation,
    ElectronReactionIonization,
    read_energy,
)
from mcdc.object_.photon_reaction import (
    PhotonReactionCoherent,
    PhotonReactionIncoherent,
    PhotonReactionPairProduction,
    PhotonReactionPhotoelectric,
)
from mcdc.print_ import print_error


class Element(MCDCObject):
    """Element definition for the MC/DC HDF5 library.

    Parameters
    ----------
    element_name : str
        Chemical symbol used to locate ``<element_name>.h5`` under ``MCDC_LIB``.

    Notes
    -----
    Construction records the element identity without accessing the data
    library. Basic properties are loaded when the element is compiled into a
    simulation. Electron reaction cross sections and secondary distributions
    are loaded later by :meth:`set_electron_data`, and the photon ones by
    :meth:`set_photon_data`.
    """

    # MC/DC framework metadata
    label = "element"

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
    #
    photon_xs_energy_grid: NDArray[float64]
    photon_total_xs: NDArray[float64]
    photon_coherent_xs: NDArray[float64]
    photon_incoherent_xs: NDArray[float64]
    photon_photoelectric_xs: NDArray[float64]
    photon_pair_production_xs: NDArray[float64]
    #
    photon_coherent_reactions: list[PhotonReactionCoherent]
    photon_incoherent_reactions: list[PhotonReactionIncoherent]
    photon_photoelectric_reactions: list[PhotonReactionPhotoelectric]
    photon_pair_production_reactions: list[PhotonReactionPairProduction]
    #
    photon_photoelectric_subshell_binding_energy: NDArray[float64]
    #
    photon_relaxation_transition_energy: NDArray[float64]
    photon_relaxation_transition_probability: NDArray[float64]
    photon_relaxation_transition_radiative: NDArray[float64]
    photon_relaxation_transition_origin: NDArray[float64]
    photon_relaxation_subshell_start: NDArray[float64]
    photon_relaxation_subshell_count: NDArray[float64]
    photon_relaxation_subshell_designator: NDArray[float64]

    def __init__(self, element_name: str):
        super().__init__()

        self.name = element_name

        # Every annotated field must be present for the Numba layer generator to
        # pack this object, but which loader runs depends on which particle
        # species is active: set_electron_data and set_photon_data each populate
        # only their own half. Initialize both halves empty so an element is
        # packable whichever subset ran.
        self._set_empty_electron_data()
        self._set_empty_photon_data()

    def _set_empty_electron_data(self):
        """Initialize the electron half to empty, pending set_electron_data."""
        self.electron_xs_energy_grid = np.zeros(0)
        self.electron_total_xs = np.zeros(0)
        self.electron_ionization_xs = np.zeros(0)
        self.electron_elastic_xs = np.zeros(0)
        self.electron_excitation_xs = np.zeros(0)
        self.electron_bremsstrahlung_xs = np.zeros(0)

        self.electron_ionization_reactions = []
        self.electron_elastic_scattering_reactions = []
        self.electron_excitation_reactions = []
        self.electron_bremsstrahlung_reactions = []

        self.electron_ionization_subshell_binding_energy = np.zeros(0)

    def _set_empty_photon_data(self):
        """Initialize the photon half to empty, pending set_photon_data."""
        self.photon_xs_energy_grid = np.zeros(0)
        self.photon_total_xs = np.zeros(0)
        self.photon_coherent_xs = np.zeros(0)
        self.photon_incoherent_xs = np.zeros(0)
        self.photon_photoelectric_xs = np.zeros(0)
        self.photon_pair_production_xs = np.zeros(0)

        self.photon_coherent_reactions = []
        self.photon_incoherent_reactions = []
        self.photon_photoelectric_reactions = []
        self.photon_pair_production_reactions = []

        self.photon_photoelectric_subshell_binding_energy = np.zeros(0)

        self.photon_relaxation_transition_energy = np.zeros(0)
        self.photon_relaxation_transition_probability = np.zeros(0)
        self.photon_relaxation_transition_radiative = np.zeros(0)
        self.photon_relaxation_transition_origin = np.zeros(0)
        self.photon_relaxation_subshell_start = np.zeros(0)
        self.photon_relaxation_subshell_count = np.zeros(0)
        self.photon_relaxation_subshell_designator = np.zeros(0)

    def _compile_into_simulation(self, simulation) -> bool:
        """Load basic properties and register with the owning simulation."""
        if self.compile_ID == simulation.compile_ID:
            return False

        dir_name = os.getenv("MCDC_LIB")
        if dir_name is None:
            print_error("Environment variable MCDC_LIB is not set")

        file_name = f"{self.name}.h5"
        file_path = os.path.join(dir_name, file_name)
        if not os.path.isfile(file_path):
            print_error(f"Element {self.name} is not available in the library")

        with h5py.File(file_path, "r") as file:
            self.atomic_weight_ratio = float(file["atomic_weight_ratio"][()])
            self.atomic_number = int(file["atomic_number"][()])

        return super()._compile_into_simulation(simulation)

    def set_electron_data(self, simulation):
        """Load and register electron reaction data from ``MCDC_LIB``."""
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

        self.electron_xs_energy_grid = read_energy(
            file["electron_reactions/xs_energy_grid"]
        )
        self.electron_total_xs = np.zeros_like(self.electron_xs_energy_grid)
        self.electron_elastic_xs = np.zeros_like(self.electron_xs_energy_grid)
        self.electron_excitation_xs = np.zeros_like(self.electron_xs_energy_grid)
        self.electron_bremsstrahlung_xs = np.zeros_like(self.electron_xs_energy_grid)
        self.electron_ionization_xs = np.zeros_like(self.electron_xs_energy_grid)

        xs_containers = [
            self.electron_elastic_xs,
            self.electron_excitation_xs,
            self.electron_bremsstrahlung_xs,
            self.electron_ionization_xs,
        ]
        for xs_container, rx_name in list(zip(xs_containers, rx_names)):
            for MT in MTs[rx_name]:
                xs = file[f"electron_reactions/{rx_name}/{MT}/xs"]
                xs_container[xs.attrs["offset"] :] += xs[()]

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
                binding_energy.append(float(read_energy(subshell["binding_energy"])))

        self.electron_ionization_subshell_binding_energy = np.asarray(binding_energy)

        file.close()

        # Register data loaded during object-model finalization.
        for reaction_container in rx_containers:
            for reaction in reaction_container:
                reaction._compile_into_simulation(simulation)

    def set_photon_data(self, simulation):
        """Load and register photon reaction data from ``MCDC_LIB``."""
        element_name = self.name

        # Load data library
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{element_name}.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        if "photon_reactions" not in file:
            file.close()
            print_error(
                f"Element {element_name} carries no photon_reactions group; the "
                "library was not built for photon transport"
            )

        # The reactions, named as the library groups them
        rx_names = [
            "coherent_scattering",
            "incoherent_scattering",
            "photoelectric",
            "pair_production",
        ]

        # The reaction MTs. Reaction groups are spelled "MTnnn" without a
        # hyphen, unlike the photoelectric subshells inside them, which are
        # "MT-nnn". Both spellings start with "MT".
        MTs = {}
        for name in rx_names:
            if name not in file["photon_reactions"]:
                MTs[name] = []
                continue

            MTs[name] = [
                x for x in file[f"photon_reactions/{name}"] if x.startswith("MT")
            ]

        # ==========================================================================
        # Reaction XS
        # ==========================================================================

        self.photon_xs_energy_grid = read_energy(
            file["photon_reactions/xs_energy_grid"]
        )
        self.photon_coherent_xs = np.zeros_like(self.photon_xs_energy_grid)
        self.photon_incoherent_xs = np.zeros_like(self.photon_xs_energy_grid)
        self.photon_photoelectric_xs = np.zeros_like(self.photon_xs_energy_grid)
        self.photon_pair_production_xs = np.zeros_like(self.photon_xs_energy_grid)

        xs_containers = [
            self.photon_coherent_xs,
            self.photon_incoherent_xs,
            self.photon_photoelectric_xs,
            self.photon_pair_production_xs,
        ]
        for xs_container, rx_name in list(zip(xs_containers, rx_names)):
            for MT in MTs[rx_name]:
                xs = file[f"photon_reactions/{rx_name}/{MT}/xs"]
                xs_container[xs.attrs["offset"] :] += xs[()]

        self.photon_total_xs = (
            self.photon_coherent_xs
            + self.photon_incoherent_xs
            + self.photon_photoelectric_xs
            + self.photon_pair_production_xs
        )

        # ==========================================================================
        # The reactions
        # ==========================================================================

        self.photon_coherent_reactions = []
        self.photon_incoherent_reactions = []
        self.photon_photoelectric_reactions = []
        self.photon_pair_production_reactions = []

        rx_containers = [
            self.photon_coherent_reactions,
            self.photon_incoherent_reactions,
            self.photon_photoelectric_reactions,
            self.photon_pair_production_reactions,
        ]
        rx_classes = [
            PhotonReactionCoherent,
            PhotonReactionIncoherent,
            PhotonReactionPhotoelectric,
            PhotonReactionPairProduction,
        ]
        for rx_container, rx_name, rx_class in list(
            zip(rx_containers, rx_names, rx_classes)
        ):
            for MT in MTs[rx_name]:
                h5_group = file[f"photon_reactions/{rx_name}/{MT}"]
                rx_container.append(rx_class.from_h5_group(h5_group))

        # ==========================================================================
        # Photoelectric element attributes
        # ==========================================================================

        binding_energy = []
        subshell_names = []
        if len(MTs["photoelectric"]) > 0:
            h5_group = file[f"photon_reactions/photoelectric/{MTs['photoelectric'][0]}"]
            if "subshells" in h5_group:
                for name in h5_group["subshells"]:
                    subshell = h5_group[f"subshells/{name}"]
                    subshell_names.append(name)
                    binding_energy.append(
                        float(read_energy(subshell["binding_energy"]))
                    )

        self.photon_photoelectric_subshell_binding_energy = np.asarray(binding_energy)

        # ==========================================================================
        # Atomic relaxation
        # ==========================================================================
        # One transition table per subshell, flattened into shared arrays and
        # indexed by the photoelectric subshell order above, so that the shell a
        # photoelectric event ionizes selects its own de-excitation table. Keyed
        # by subshell MT name rather than by position, because the relaxation and
        # photoelectric subshell sets need not agree.
        #
        # secondary_designator distinguishes the two de-excitation channels: zero
        # is radiative, so a characteristic X-ray is emitted and transported;
        # non-zero is non-radiative (Auger), whose electron is not transported and
        # therefore deposits locally.
        #
        # origin_designator names the subshell the filling electron came from, so
        # it is where the vacancy moves next. Paired with each subshell's own
        # designator it lets the de-excitation cascade be followed one further
        # step, which is what emits the second (L) fluorescence photon.

        transition_energy = []
        transition_probability = []
        transition_radiative = []
        transition_origin = []
        subshell_start = []
        subshell_count = []
        subshell_designator = []

        relaxation = (
            file["atomic_relaxation/subshells"]
            if (
                "atomic_relaxation" in file and "subshells" in file["atomic_relaxation"]
            )
            else None
        )

        for name in subshell_names:
            subshell_start.append(len(transition_energy))

            if relaxation is None or name not in relaxation:
                subshell_count.append(0)
                subshell_designator.append(0)
                continue

            subshell = relaxation[name]
            subshell_designator.append(
                int(subshell["designator"][()]) if "designator" in subshell else 0
            )
            if "transitions" not in subshell:
                subshell_count.append(0)
                continue

            transitions = subshell["transitions"]
            energies = read_energy(transitions["energy"])
            probabilities = transitions["probability"][()]
            secondary = transitions["secondary_designator"][()]
            origin = (
                transitions["origin_designator"][()]
                if "origin_designator" in transitions
                else np.zeros(len(energies), dtype=np.int64)
            )

            subshell_count.append(len(energies))
            for index in range(len(energies)):
                transition_energy.append(float(energies[index]))
                transition_probability.append(float(probabilities[index]))
                transition_radiative.append(1.0 if int(secondary[index]) == 0 else 0.0)
                transition_origin.append(float(int(origin[index])))

        self.photon_relaxation_transition_energy = np.asarray(
            transition_energy, dtype=float64
        )
        self.photon_relaxation_transition_probability = np.asarray(
            transition_probability, dtype=float64
        )
        self.photon_relaxation_transition_radiative = np.asarray(
            transition_radiative, dtype=float64
        )
        self.photon_relaxation_transition_origin = np.asarray(
            transition_origin, dtype=float64
        )
        self.photon_relaxation_subshell_start = np.asarray(
            subshell_start, dtype=float64
        )
        self.photon_relaxation_subshell_count = np.asarray(
            subshell_count, dtype=float64
        )
        self.photon_relaxation_subshell_designator = np.asarray(
            subshell_designator, dtype=float64
        )

        file.close()

        # Register data loaded during object-model finalization.
        for reaction_container in rx_containers:
            for reaction in reaction_container:
                reaction._compile_into_simulation(simulation)

    def __repr__(self):
        text = "\n"
        text += f"Element\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic number: {self.atomic_number}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        return text
