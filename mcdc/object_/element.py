import h5py
import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.object_.base import MCDCObject
from mcdc.object_.hdf5_reader import (
    check_format,
    load_target_properties,
    read_xs_energy_grid,
    text,
    xs_is_linear,
)
from mcdc.object_.electron_reaction import (
    ElectronReactionBremsstrahlung,
    ElectronReactionElasticScattering,
    ElectronReactionExcitation,
    ElectronReactionIonization,
    read_energy,
)
from mcdc.object_.photon_reaction import (
    PhotonReactionBase,
    PhotonReactionCoherentScattering,
    ordered_subshells,
    PhotonReactionIncoherentScattering,
    PhotonReactionPairProduction,
    PhotonReactionPhotoelectric,
)
from mcdc.object_.util import atomic_number_from_element_name

# ======================================================================================
# Element
# ======================================================================================


class Element(MCDCObject):
    """Element definition for the MC/DC HDF5 library.

    Parameters
    ----------
    element_name : str
        Chemical symbol identifying the target in projectile and atomic files.

    Notes
    -----
    Construction records the element identity without accessing the data
    library. Projectile loaders resolve basic properties during compilation.
    Electron reaction cross sections and secondary distributions
    are loaded later by :meth:`set_electron_data`, and the photon ones by
    :meth:`set_photon_data`.
    Isotopes and temperature-specific nuclides share one element object per
    chemical symbol within a simulation.
    """

    # MC/DC framework metadata
    label = "element"

    name: str
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
    photon_xs_is_linear: bool
    photon_reactions: list[PhotonReactionBase]
    photon_xs_energy_grid: NDArray[float64]
    photon_total_xs: NDArray[float64]
    photon_coherent_xs: NDArray[float64]
    photon_incoherent_xs: NDArray[float64]
    photon_photoelectric_xs: NDArray[float64]
    photon_pair_production_xs: NDArray[float64]
    #
    photon_coherent_scattering_reactions: list[PhotonReactionCoherentScattering]
    photon_incoherent_scattering_reactions: list[PhotonReactionIncoherentScattering]
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
        self.atomic_number = atomic_number_from_element_name(element_name)
        self._property_sources = {}

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
        self.photon_xs_is_linear = True
        self.photon_reactions = []
        self.photon_xs_energy_grid = np.zeros(0)
        self.photon_total_xs = np.zeros(0)
        self.photon_coherent_xs = np.zeros(0)
        self.photon_incoherent_xs = np.zeros(0)
        self.photon_photoelectric_xs = np.zeros(0)
        self.photon_pair_production_xs = np.zeros(0)

        self.photon_coherent_scattering_reactions = []
        self.photon_incoherent_scattering_reactions = []
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
        """Register this target and its shared references with the simulation."""
        if self.compile_ID == simulation.compile_ID:
            return False

        self._property_sources = {}
        return super()._compile_into_simulation(simulation)

    def _load_properties(self, group, field="atomic-number"):
        """Load element properties while checking their chemical identity."""
        expected = atomic_number_from_element_name(self.name)
        if not expected and "atomic_number" in self._property_sources:
            expected = self.atomic_number
        load_target_properties(
            self,
            group,
            (field,),
            identity={"atomic_number": expected} if expected else None,
        )

    def set_electron_data(self, simulation):
        """Load and register electron reaction data from ``MCDC_LIB``."""
        element_name = self.name

        # Load data library
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{element_name}.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        self._load_properties(
            file, field="atomic_number" if "atomic_number" in file else "atomic-number"
        )

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
        file_name = f"photon-{element_name}.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        check_format(file)
        self._load_properties(file)
        photon = file

        # The reaction categories
        rx_names = [
            "coherent-scattering",
            "incoherent-scattering",
            "photoelectric",
            "pair-production",
        ]

        # Group reaction names by their declared transport category.
        reaction_names = {name: [] for name in rx_names}
        for name, group in photon["reactions"].items():
            category = text(group.attrs["category"])
            rx_name = category
            if rx_name not in reaction_names:
                raise ValueError(
                    f"{group.name}: unknown photon reaction category {category!r}"
                )
            reaction_names[rx_name].append(name)

        # ==========================================================================
        # Reaction XS
        # ==========================================================================

        self.photon_xs_energy_grid = read_xs_energy_grid(photon)
        grid = self.photon_xs_energy_grid
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
            for name in reaction_names[rx_name]:
                xs = photon[f"reactions/{name}/xs"]
                offset = int(xs["offset"][()])
                if offset < 0 or offset + len(xs["value"]) != len(grid):
                    raise ValueError(
                        f"{xs.name}: cross-section segment does not end at the shared grid boundary"
                    )
                xs_container[offset:] += xs["value"][()]

        self.photon_total_xs = (
            self.photon_coherent_xs
            + self.photon_incoherent_xs
            + self.photon_photoelectric_xs
            + self.photon_pair_production_xs
        )

        # ==========================================================================
        # The reactions
        # ==========================================================================

        self.photon_reactions = []
        self.photon_xs_is_linear = True
        self.photon_coherent_scattering_reactions = []
        self.photon_incoherent_scattering_reactions = []
        self.photon_photoelectric_reactions = []
        self.photon_pair_production_reactions = []

        rx_containers = [
            self.photon_coherent_scattering_reactions,
            self.photon_incoherent_scattering_reactions,
            self.photon_photoelectric_reactions,
            self.photon_pair_production_reactions,
        ]
        rx_classes = [
            PhotonReactionCoherentScattering,
            PhotonReactionIncoherentScattering,
            PhotonReactionPhotoelectric,
            PhotonReactionPairProduction,
        ]
        for rx_container, rx_name, rx_class in list(
            zip(rx_containers, rx_names, rx_classes)
        ):
            for name in reaction_names[rx_name]:
                h5_group = photon[f"reactions/{name}"]
                reaction = rx_class.from_h5_group(h5_group)
                rx_container.append(reaction)
                self.photon_reactions.append(reaction)
                self.photon_xs_is_linear &= xs_is_linear(reaction)

        # ==========================================================================
        # Photoelectric element attributes
        # ==========================================================================

        binding_energy = []
        if len(reaction_names["photoelectric"]) > 0:
            h5_group = photon[f"reactions/{reaction_names['photoelectric'][0]}"]
            if "subshells" in h5_group:
                for subshell in ordered_subshells(h5_group["subshells"]):
                    binding_energy.append(float(subshell["binding-energy"][()]))

        self.photon_photoelectric_subshell_binding_energy = np.asarray(binding_energy)

        file.close()
        self.set_atomic_data(simulation)

        # Register data loaded during object-model finalization.
        for reaction_container in rx_containers:
            for reaction in reaction_container:
                reaction._compile_into_simulation(simulation)

    def set_atomic_data(self, simulation):
        """Load this element's EADL record when atoms.h5 supplies one.

        Atomic data belongs to the shared Element. Missing records leave empty
        transition arrays, while malformed records or inconsistent identities
        raise errors. Only the requested element is read from the collection.
        """
        self._set_atomic_relaxation({})
        path = os.path.join(os.environ["MCDC_LIB"], "atoms.h5")
        if not os.path.isfile(path):
            return
        with h5py.File(path, "r") as file:
            check_format(file)
            if self.name in file:
                record = file[self.name]
                self._load_properties(record)
                self._set_atomic_relaxation(record)

    def _set_atomic_relaxation(self, record):
        """Flatten atomic transition tables using their explicit shell designators."""
        # ==========================================================================
        # Atomic relaxation
        # ==========================================================================
        # Keep all relaxation shells, including ones reached only by a cascade.
        # Explicit designators connect photoelectric vacancies and shell transitions.
        # A zero secondary designator identifies a radiative transition.

        transition_energy = []
        transition_probability = []
        transition_radiative = []
        transition_origin = []
        subshell_start = []
        subshell_count = []
        subshell_designator = []

        relaxation = record.get("atomic-relaxation/subshells", {})
        subshells = ordered_subshells(relaxation)
        designators = {int(subshell["designator"][()]) for subshell in subshells}

        for subshell in subshells:
            subshell_start.append(len(transition_energy))
            subshell_designator.append(int(subshell["designator"][()]))
            transitions = subshell["transitions"]
            energies = transitions["energy"][()]
            probabilities = transitions["probability"][()]
            secondary = transitions["secondary-designator"][()]
            origin = transitions["origin-designator"][()]
            if (
                any(
                    array.ndim != 1
                    for array in (energies, probabilities, secondary, origin)
                )
                or not (
                    len(energies) == len(probabilities) == len(secondary) == len(origin)
                )
                or not np.all(np.isfinite(energies))
                or np.any(energies < 0)
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or (
                    len(probabilities)
                    and not np.isclose(sum(probabilities), 1.0, rtol=1e-5, atol=1e-8)
                )
                or not set(origin) <= designators
                or not (set(secondary) - {0}) <= designators
            ):
                raise ValueError(
                    f"{transitions.name}: invalid atomic-relaxation transitions"
                )
            if len(probabilities):
                probabilities = probabilities / sum(probabilities)

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

    def __repr__(self):
        text = "\n"
        text += f"Element\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic number: {self.atomic_number}\n"
        return text


# ======================================================================================
# Element lookup
# ======================================================================================


def get_or_create_element(element_name, simulation, candidate=None):
    """Reuse the simulation's element object for a chemical symbol.

    A candidate preserves an existing object's identity when the symbol has
    not been registered yet. Compilation registers the returned object and
    loads its basic properties independently of projectile reaction data.
    """
    for element in simulation.elements:
        if element.name == element_name:
            return element
    return candidate if candidate is not None else Element(element_name)
