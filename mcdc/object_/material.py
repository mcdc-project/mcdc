from types import NoneType
from typing import Self

import numpy as np
from numpy import float64
from numpy.typing import ArrayLike, NDArray
import h5py
import os

from mcdc.constant import INTERPOLATION_LINEAR
from mcdc.object_.data import DataBase, DataNone, DataTable
from mcdc.object_.base import MCDCObject
from mcdc.object_.element import Element
from mcdc.object_.transport_model_data import (
    NeutronMultigroupData,
    PhotonConstantXSData,
)
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.util import ISOTOPIC_ABUNDANCE, element_symbol_from_nuclide_name
from mcdc.print_ import print_error

# ======================================================================================
# Material
# ======================================================================================


class Material(MCDCObject):
    """Particle-interaction properties assigned to simulation cells.

    Parameters
    ----------
    name : str, optional
        User-facing material name.
    nuclide_composition : dict of str to float, optional
        Nuclide names and atomic densities in atoms/(barn cm).
    element_composition : dict of str to float, optional
        Element symbols and atomic densities in atoms/(barn cm).
    temperature : float, optional
        Material temperature in kelvin. Each nuclide uses the closest
        temperature available in the data library.
    photon_constant_xs : PhotonConstantXSData, optional
        Energy-independent photon cross sections for this material. Supplying
        it selects the constant-cross-section photon treatment in place of the
        tabulated per-element data.
    neutron_multigroup : NeutronMultigroupData, optional
        Groupwise macroscopic cross sections and related data for neutron
        multigroup transport, where neutron energy is represented by discrete
        groups. When supplied with a
        :ref:`native composition <user_native_transport>`, its explicit
        ``energy_grid`` defines the energy range where the multigroup treatment
        applies. In hybrid transport, native neutron data is used outside that
        range when present; without native composition, the material has zero
        interaction cross section.

    Notes
    -----
    A nuclide or element composition connects the material to MC/DC's native
    transport physics through its data libraries. Particle-specific data can
    augment native interaction data or support specialized and reduced
    transport treatments.
    ``NeutronMultigroupData`` describes discrete neutron energy groups and
    their macroscopic interaction, production, and timing data. It can be used
    alone or alongside a native composition.

    Examples
    --------
    Define uranium dioxide from nuclide atomic densities:

    >>> import mcdc
    >>> fuel = mcdc.Material(
    ...     name="UO2",
    ...     nuclide_composition={"U235": 5.0e-4, "U238": 2.2e-2, "O16": 4.5e-2},
    ...     temperature=293.6,
    ... )

    Define a one-group multigroup material:

    >>> import numpy as np
    >>> absorber = mcdc.Material.multigroup(
    ...     name="Absorber", capture=np.array([1.0])
    ... )

    Attach native data and multigroup neutron data to the same material:

    >>> hybrid_fuel = mcdc.Material(
    ...     name="Hybrid fuel",
    ...     nuclide_composition={"U235": 5.0e-4, "U238": 2.2e-2},
    ...     neutron_multigroup=mcdc.NeutronMultigroupData(
    ...         capture=np.array([0.10]),
    ...         fission=np.array([0.20]),
    ...         nu_p=np.array([2.50]),
    ...         energy_grid=np.array([1.0e-5, 20.0e6]),
    ...     ),
    ... )
    """

    # MC/DC framework metadata
    label = "material"
    non_numba = ["nuclide_composition", "element_composition"]

    name: str
    temperature: float
    fissionable: bool
    has_neutron_multigroup: bool
    has_photon_constant_xs: bool

    nuclide_composition: dict[Nuclide, float]  # Non-Numba
    element_composition: dict[Element, float]  # Non-Numba
    neutron_multigroup: NeutronMultigroupData
    photon_constant_xs: PhotonConstantXSData

    nuclides: list[Nuclide]
    elements: list[Element]
    nuclide_densities: NDArray[float64]
    element_densities: NDArray[float64]

    stopping_power_provided: bool = False
    stopping_power: DataBase

    radiation_length: float = 0.0
    radiation_length_provided: bool = False

    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict[str, float] | NoneType = None,
        element_composition: dict[str, float] | NoneType = None,
        temperature: float = 293.6,
        neutron_multigroup: NeutronMultigroupData | NoneType = None,
        photon_constant_xs: PhotonConstantXSData | NoneType = None,
    ) -> None:
        super().__init__()

        # Normalize optional compositions without mutable argument defaults
        nuclide_composition = nuclide_composition or {}
        element_composition = element_composition or {}

        # Require one valid material-data representation
        if nuclide_composition and element_composition:
            print_error(
                "Cannot specify both nuclide_composition and element_composition."
            )
        if (
            not nuclide_composition
            and not element_composition
            and neutron_multigroup is None
            and photon_constant_xs is None
        ):
            print_error(
                "Material requires nuclide_composition, element_composition, "
                "neutron_multigroup, or photon_constant_xs."
            )
        if photon_constant_xs is not None and not isinstance(
            photon_constant_xs, PhotonConstantXSData
        ):
            print_error("photon_constant_xs must be a PhotonConstantXSData object.")
        if photon_constant_xs is not None and photon_constant_xs.total <= 0.0:
            print_error(
                "Material photon_constant_xs must define a positive total cross "
                "section."
            )
        if neutron_multigroup is not None and not isinstance(
            neutron_multigroup, NeutronMultigroupData
        ):
            print_error("neutron_multigroup must be a NeutronMultigroupData object.")
        if neutron_multigroup is not None and neutron_multigroup.G == 0:
            print_error(
                "Material neutron_multigroup must define at least one energy group."
            )
        if (
            (nuclide_composition or element_composition)
            and neutron_multigroup is not None
            and neutron_multigroup.G > 0
            and not np.any(neutron_multigroup.energy_grid)
        ):
            print_error(
                "Material requires an explicit neutron multigroup energy_grid "
                "when a nuclide or element composition is supplied."
            )
        # Initialize shared material state
        self.name = name or "(Unnamed material)"
        self.temperature = float(temperature)

        # Use a zero-group placeholder until compilation can select ID 0
        self.neutron_multigroup = (
            neutron_multigroup
            if neutron_multigroup is not None
            else NeutronMultigroupData()
        )
        self.has_neutron_multigroup = self.neutron_multigroup.G > 0
        self.fissionable = self.neutron_multigroup.fissionable

        # Use a zero-cross-section placeholder until compilation can select ID 0
        self.photon_constant_xs = (
            photon_constant_xs
            if photon_constant_xs is not None
            else PhotonConstantXSData()
        )
        self.has_photon_constant_xs = self.photon_constant_xs.total > 0.0

        # Create lightweight native-composition objects without loading data
        nearest_temperature = _get_supported_temperature(self.temperature)
        self.nuclide_composition = {
            Nuclide(name, nearest_temperature): float(density)
            for name, density in nuclide_composition.items()
        }
        self.element_composition = {
            Element(name): float(density)
            for name, density in element_composition.items()
        }

        # Initialize the packed native-composition representation
        self.nuclides = list(self.nuclide_composition)
        self.elements = list(self.element_composition)
        self.nuclide_densities = np.asarray(
            list(self.nuclide_composition.values()), dtype=float64
        )
        self.element_densities = np.asarray(
            list(self.element_composition.values()), dtype=float64
        )

        self.stopping_power = DataNone()

    @classmethod
    def multigroup(
        cls,
        *,
        name: str = "",
        capture: ArrayLike | NoneType = None,
        scatter: ArrayLike | NoneType = None,
        fission: ArrayLike | NoneType = None,
        nu_s: ArrayLike | NoneType = None,
        nu_p: ArrayLike | NoneType = None,
        nu_d: ArrayLike | NoneType = None,
        chi_p: ArrayLike | NoneType = None,
        chi_d: ArrayLike | NoneType = None,
        speed: ArrayLike | NoneType = None,
        decay_rate: ArrayLike | NoneType = None,
        energy_grid: ArrayLike | NoneType = None,
        energy_representation: str | int = "midpoint",
    ) -> Self:
        """Create a material for neutron multigroup transport.

        The supplied arguments define groupwise macroscopic interaction and
        production data. This helper stores them in
        :class:`NeutronMultigroupData` and attaches the data to the material.
        Macroscopic cross sections use ``cm^-1``, group speeds use cm/s,
        precursor decay rates use ``s^-1``, and explicit physical energy
        boundaries use eV. Omitting ``energy_grid`` creates a zero-valued
        placeholder that is valid only when every material with neutron
        multigroup data also omits its energy grid.
        """
        # Build the neutron multigroup data while preserving Material as the sole type
        neutron_multigroup = NeutronMultigroupData(
            capture=capture,
            scatter=scatter,
            fission=fission,
            nu_s=nu_s,
            nu_p=nu_p,
            nu_d=nu_d,
            chi_p=chi_p,
            chi_d=chi_d,
            speed=speed,
            decay_rate=decay_rate,
            energy_grid=energy_grid,
            energy_representation=energy_representation,
        )
        return cls(name=name, neutron_multigroup=neutron_multigroup)

    def _compile_into_simulation(self, simulation) -> bool:
        """Canonicalize owned data and register the unified material."""
        # Skip canonicalization when this material is already compiled
        if self.compile_ID == simulation.compile_ID:
            return False

        # Resolve native composition objects before generic member traversal
        nuclide_composition = {}
        element_composition = {}

        for element, density in self.element_composition.items():
            element = _get_or_create_element(element.name, simulation, element)
            element_composition[element] = density

        for nuclide, density in self.nuclide_composition.items():
            nuclide = _get_or_create_nuclide(
                nuclide.name, nuclide.temperature, simulation, nuclide
            )
            nuclide_composition[nuclide] = density

        # Synchronize packed fields with canonical native objects
        self.nuclide_composition = nuclide_composition
        self.element_composition = element_composition
        self.nuclides = list(nuclide_composition)
        self.elements = list(element_composition)
        self.nuclide_densities = np.asarray(
            list(nuclide_composition.values()), dtype=float64
        )
        self.element_densities = np.asarray(
            list(element_composition.values()), dtype=float64
        )

        # Point absent multigroup data to the reserved neutron model
        if self.neutron_multigroup.G == 0:
            self.neutron_multigroup = simulation.neutron_multigroup_data[0]
        self.has_neutron_multigroup = self.neutron_multigroup.G > 0

        if self.photon_constant_xs.total == 0.0:
            self.photon_constant_xs = simulation.photon_constant_xs_data[0]
        self.has_photon_constant_xs = self.photon_constant_xs.total > 0.0

        # Register the material, then compile only canonical owned members
        if not super()._compile_into_simulation(simulation):
            return False

        # Resolve fissionability from every available neutron representation
        self.fissionable = self.neutron_multigroup.fissionable or any(
            nuclide.fissionable for nuclide in self.nuclides
        )

        return True

    def add_stopping_power(
        self,
        stopping_power_filename: str = "",
    ):

        self.stopping_power_provided = True

        dir_name = os.getenv("MCDC_LIB")
        file_name = stopping_power_filename
        file = h5py.File(f"{dir_name}/{file_name}.h5", "r")

        self.stopping_power = DataTable(
            file["stopping_power"]["energy"][()],
            file["stopping_power"]["total_stopping_power"][()],
            INTERPOLATION_LINEAR,
        )
        if file["radiation_length"]["radiation_length"][()]:
            self.radiation_length = file["radiation_length"]["radiation_length"][()]

        file.close()

    def __repr__(self) -> str:
        text = super().__repr__()
        text += f"  - Name: {self.name}\n"
        text += f"  - Fissionable: {self.fissionable}\n"
        text += f"  - Temperature: {self.temperature} K\n"

        if self.nuclide_composition:
            text += "  - Nuclide composition [atoms/barn-cm]\n"
            for nuclide, density in self.nuclide_composition.items():
                text += f"    - {nuclide.name:<5} | {density}\n"

        if self.element_composition:
            text += "  - Element composition [atoms/barn-cm]\n"
            for element, density in self.element_composition.items():
                text += f"    - {element.name:<5} | {density}\n"

        if self.neutron_multigroup.G > 0:
            text += "  - Neutron multigroup model\n"
            text += f"    - G: {self.neutron_multigroup.G}\n"
            text += f"    - J: {self.neutron_multigroup.J}\n"
        return text


# Currently supported temperatures
TEMPERATURES = [0.0, 0.1, 233.15, 273.15, 293.6, 600.0, 900.0, 1200.0, 2500.0]

# ======================================================================================
# Native-composition helpers
# ======================================================================================


def set_nuclides_from_elements(material, simulation):
    """Expand an elemental composition and register its natural isotopes."""
    material.nuclides = []
    material.nuclide_composition = {}
    nuclide_densities = []

    # Select the nearest supported native-data temperature
    nearest_temperature = _get_supported_temperature(material.temperature)

    # Expand each natural element into its normalized isotopic composition
    for element, element_density in material.element_composition.items():
        norm = sum(ISOTOPIC_ABUNDANCE[element.name].values())

        for nuclide_name, abundance in ISOTOPIC_ABUNDANCE[element.name].items():
            nuclide = _get_or_create_nuclide(
                nuclide_name, nearest_temperature, simulation
            )
            nuclide._compile_into_simulation(simulation)

            nuclide_density = element_density * abundance / norm
            material.nuclides.append(nuclide)
            nuclide_densities.append(nuclide_density)
            material.nuclide_composition[nuclide] = nuclide_density

    material.nuclide_densities = np.asarray(nuclide_densities, dtype=float64)


def set_elements_from_nuclides(material, simulation):
    """Collapse a nuclide composition and register its elements."""
    # Accumulate nuclide densities by exact element symbol (e.g., C12 -> C, Cr52 -> Cr)
    element_densities = {}
    for nuclide, nuclide_density in material.nuclide_composition.items():
        element_name = element_symbol_from_nuclide_name(nuclide.name)
        element_densities[element_name] = (
            element_densities.get(element_name, 0.0) + nuclide_density
        )

    # Register each element's canonical object
    material.elements = []
    material.element_composition = {}
    for element_name, density in element_densities.items():
        element = _get_or_create_element(element_name, simulation)
        element._compile_into_simulation(simulation)
        material.elements.append(element)
        material.element_composition[element] = density

    material.element_densities = np.asarray(
        list(element_densities.values()), dtype=float64
    )


def _get_supported_temperature(temperature):
    return min(TEMPERATURES, key=lambda value: abs(value - temperature))


def _get_or_create_element(element_name, simulation, candidate=None):
    for element in simulation.elements:
        if element.name == element_name:
            return element
    return candidate or Element(element_name)


def _get_or_create_nuclide(nuclide_name, temperature, simulation, candidate=None):
    for nuclide in simulation.nuclides:
        if nuclide.name == nuclide_name and nuclide.temperature == temperature:
            return nuclide
    return candidate or Nuclide(nuclide_name, temperature)


def update_fissionable_from_nuclides(material):
    """Update fissionability from native and multigroup neutron data."""
    material.fissionable = material.neutron_multigroup.fissionable or any(
        nuclide.fissionable for nuclide in material.nuclides
    )


def update_radiation_length_from_nuclides(material):
    """Update mixture radiation length from nuclide mass fractions."""
    total_mass = 0.0
    X0_weighted_mass = 0.0
    for nuclide, density in material.nuclide_composition.items():
        nuclide_mass = nuclide.mass_number
        nuclide_X0 = nuclide.radiation_length
        total_mass += nuclide_mass * density
        X0_weighted_mass += nuclide_mass * density / nuclide_X0

        material.radiation_length = total_mass / X0_weighted_mass
