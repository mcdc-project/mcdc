from types import NoneType
from typing import Self

import numpy as np
from numpy import float64
from numpy.typing import ArrayLike, NDArray
import h5py
import os

from mcdc.constant import (
    AMU_PER_NEUTRON_MASS,
    AVOGADRO_BARN_CM,
    INTERPOLATION_LINEAR,
)
from mcdc.object_.data import DataBase, DataNone, DataTable
from mcdc.object_.base import MCDCObject
from mcdc.object_.element import Element
from mcdc.object_.transport_model_data import NeutronMultigroupData
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.util import (
    ISOTOPIC_ABUNDANCE,
    element_symbol_from_nuclide_name,
    is_nuclide_name,
)
from mcdc.print_ import print_error, print_warning

# ======================================================================================
# Material
# ======================================================================================


class Material(MCDCObject):
    """Particle-interaction properties assigned to simulation cells.

    Parameters
    ----------
    name : str, optional
        User-facing material name.
    number_density : dict of str to float, optional
        Nuclide or element names and atomic densities in atoms/(barn cm).
    nuclide_composition : dict of str to float, optional
        Nuclide names and atomic densities in atoms/(barn cm). Alias for number_density.
    element_composition : dict of str to float, optional
        Element symbols and atomic densities in atoms/(barn cm). Alias for number_density.
    temperature : float, optional
        Material temperature in kelvin. Each nuclide uses the closest
        temperature available in the data library (elements do not have temperature).
    neutron_multigroup : NeutronMultigroupData, optional
        Groupwise macroscopic cross sections and related data for neutron
        multigroup transport, where neutron energy is represented by discrete
        groups. When supplied with a
        :ref:`native composition <user_native_transport>`, its explicit
        ``energy_grid`` defines the energy range where the multigroup treatment
        applies. In hybrid transport, native neutron data is used outside that
        range when present; without native composition, the material has zero
        interaction cross section.
    mass_density : dict of str to float, optional
        Nuclide or element names and mass densities in g/cm^3. Converted to
        atomic densities when the simulation is compiled, using atomic weights
        from the data library (see Notes).
    mass_fractions : dict of str to float, optional
        Nuclide or element names and mass fractions. Requires ``density``. The
        fractions are normalized to sum to one, with a warning if their sum
        differs from one by more than 0.01.
    density : float, optional
        Total material density in g/cm^3, distributed over ``mass_fractions``.
        Only valid together with ``mass_fractions``.

    Notes
    -----
    A nuclide or element composition connects the material to MC/DC's native
    transport physics through its data libraries. Particle-specific data can
    augment native interaction data or support specialized and reduced
    transport treatments.

    ``NeutronMultigroupData`` describes discrete neutron energy groups and
    their macroscopic interaction, production, and timing data. It can be used
    alone or alongside a native composition.

    ``number_density`` allows nuclides and elements to be mixed in a single dict.
    ``element_composition`` and ``nuclide_composition`` are aliases for
    ``number_density``. Specify only one of ``number_density``,
    ``nuclide_composition``, ``element_composition``, ``mass_density``, and
    ``mass_fractions``; the first three have the same behavior.

    Atomic symbols ("U", "Si") are interpreted as elements, and atomic symbols with
    mass numbers ("U235", "O16") are interpreted as nuclides. For neutrons, elements
    are expanded into nuclides using their natural abundance; for photons and
    electrons, element data is used. If a nuclide is specified in a problem with
    photons or electrons, the element is inferred from the nuclide (e.g. "Al27"
    uses the electron data for "Al").

    Element/nuclide names are not checked when the Material is created, so custom
    elements or nuclides can be used; a name missing from the data library raises
    an error when the simulation is compiled.

    If an element and an isotope of that element (e.g. "U" and "U235") are both
    specified, the densities are summed (the provided density for U235 is added
    to the density obtained by expanding U).

    ``mass_density`` and ``mass_fractions`` are converted to atomic densities as
    :math:`N = \\rho N_A / A`, where :math:`A` is the atomic weight. A nuclide uses
    its library atomic weight. An element uses the abundance-weighted average of
    its natural isotopes' weights, so the isotope files of that element must be
    in the data library. After compilation, ``number_density`` holds the
    resulting atomic densities.

    Examples
    --------
    Define uranium dioxide from nuclide/element atomic densities:

    >>> import mcdc
    >>> fuel = mcdc.Material(
    ...     name="UO2",
    ...     number_density={
    ...         "U235": 5.0e-4,
    ...         "U238": 2.2e-2,
    ...         "O": 4.5e-2  # natural O abundance
    ...     },
    ...     temperature=293.6,
    ... )

    Define the same material from mass fractions and a total density:

    >>> fuel = mcdc.Material(
    ...     name="UO2",
    ...     mass_fractions={"U235": 0.0193, "U238": 0.862, "O": 0.1184},
    ...     density=10.09,
    ... )

    Define the same material from per-isotope/per-element mass densities:

    >>> fuel = mcdc.Material(
    ...     name="UO2",
    ...     mass_density={"U235": 0.1947, "U238": 8.6976, "O": 1.1947},
    ... )

    Define a one-group multigroup material:

    >>> import numpy as np
    >>> absorber = mcdc.Material.multigroup(
    ...     name="Absorber", capture=np.array([1.0])
    ... )

    Attach native data and multigroup neutron data to the same material:

    >>> hybrid_fuel = mcdc.Material(
    ...     name="Hybrid fuel",
    ...     number_density={"U235": 5.0e-4, "U238": 2.2e-2},
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
    non_numba = [
        "number_density",
        "nuclide_composition",
        "element_composition",
        "mass_density",
    ]

    name: str
    temperature: float
    fissionable: bool
    has_neutron_multigroup: bool

    number_density: dict[str, float]  # Non-Numba
    mass_density: dict[str, float]  # Non-Numba
    nuclide_composition: dict[Nuclide, float]  # Non-Numba
    element_composition: dict[Element, float]  # Non-Numba
    neutron_multigroup: NeutronMultigroupData

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
        number_density: dict[str, float] | NoneType = None,
        nuclide_composition: dict[str, float] | NoneType = None,
        element_composition: dict[str, float] | NoneType = None,
        temperature: float = 293.6,
        neutron_multigroup: NeutronMultigroupData | NoneType = None,
        mass_density: dict[str, float] | NoneType = None,
        mass_fractions: dict[str, float] | NoneType = None,
        density: float | NoneType = None,
    ) -> None:
        super().__init__()

        # Normalize optional compositions without mutable argument defaults
        number_density = number_density or {}
        nuclide_composition = nuclide_composition or {}
        element_composition = element_composition or {}
        mass_density = mass_density or {}
        mass_fractions = mass_fractions or {}

        # Require one valid material-data representation
        if (
            bool(nuclide_composition)
            + bool(element_composition)
            + bool(number_density)
            + bool(mass_density)
            + bool(mass_fractions)
        ) > 1:
            print_error(
                "Cannot specify more than one of nuclide_composition, "
                "element_composition, number_density, mass_density, or mass_fractions."
            )
        if bool(mass_fractions) != (density is not None):
            print_error("mass_fractions and density must be specified together.")
        if density is not None and density <= 0.0:
            print_error("density must be positive.")
        if any(
            value < 0.0 for value in (*mass_density.values(), *mass_fractions.values())
        ):
            print_error("mass_density and mass_fractions must be non-negative.")
        if any(
            value < 0.0
            for value in (
                *number_density.values(),
                *nuclide_composition.values(),
                *element_composition.values(),
            )
        ):
            print_error("Atomic densities must be non-negative.")

        # Reduce mass fractions to per-name mass densities
        if mass_fractions:
            total = sum(mass_fractions.values())
            if total <= 0.0:
                print_error("mass_fractions must have a positive sum.")
            if abs(total - 1.0) > 0.01:
                print_warning(
                    f"mass_fractions sum to {total:.4g} rather than 1; "
                    "they have been normalized."
                )
            mass_density = {
                name: density * fraction / total
                for name, fraction in mass_fractions.items()
            }

        if (
            not nuclide_composition
            and not element_composition
            and not number_density
            and not mass_density
            and neutron_multigroup is None
        ):
            print_error(
                "Material requires nuclide_composition, element_composition, "
                "number_density, mass_density, mass_fractions, or neutron_multigroup."
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
            (
                nuclide_composition
                or element_composition
                or number_density
                or mass_density
            )
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

        if nuclide_composition:
            number_density = nuclide_composition
        if element_composition:
            number_density = element_composition
        if mass_density:
            # Atomic densities need atomic weights, which are read at compile time
            number_density = dict.fromkeys(mass_density, 0.0)
        self.mass_density = {name: float(rho) for name, rho in mass_density.items()}

        # Create lightweight native-composition objects without loading data
        nearest_temperature = _get_supported_temperature(self.temperature)
        self.nuclide_composition = {
            Nuclide(name, nearest_temperature): float(density)
            for name, density in number_density.items()
            if is_nuclide_name(name)
        }
        self.element_composition = {
            Element(name): float(density)
            for name, density in number_density.items()
            if not is_nuclide_name(name)
        }
        self.number_density = {
            name: float(density) for name, density in number_density.items()
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
        converted = {}

        for element, density in self.element_composition.items():
            element = _get_or_create_element(element.name, simulation, element)
            if self.mass_density:
                density = _number_density_from_mass(
                    self.mass_density[element.name],
                    _element_weight(element.name, self.temperature, simulation),
                )
                converted[element.name] = density
            element_composition[element] = density

        for nuclide, density in self.nuclide_composition.items():
            nuclide = _get_or_create_nuclide(
                nuclide.name, nuclide.temperature, simulation, nuclide
            )
            if self.mass_density:
                density = _number_density_from_mass(
                    self.mass_density[nuclide.name],
                    _nuclide_weight(nuclide, simulation),
                )
                converted[nuclide.name] = density
            nuclide_composition[nuclide] = density

        # Replace the placeholders with the converted atomic densities
        if self.mass_density:
            self.number_density = {name: converted[name] for name in self.mass_density}

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
    """Build the nuclide view: explicit nuclides plus natural isotopes of elements."""
    # return early if no elements
    if all(is_nuclide_name(name) for name in material.number_density):
        return

    # Select the nearest supported native-data temperature
    nearest_temperature = _get_supported_temperature(material.temperature)

    # Collect (nuclide name, density) pairs in input order
    pairs = []
    for name, density in material.number_density.items():
        if is_nuclide_name(name):
            pairs.append((name, density))
            continue

        # Expand a natural element into its normalized isotopic composition
        if name not in ISOTOPIC_ABUNDANCE:
            print_error(
                f"No natural isotopic abundance data for element {name}; "
                "specify its nuclides directly."
            )
        norm = sum(ISOTOPIC_ABUNDANCE[name].values())
        for nuclide_name, abundance in ISOTOPIC_ABUNDANCE[name].items():
            pairs.append((nuclide_name, density * abundance / norm))

    # Register canonical nuclides, summing the density of any repeated nuclide
    material.nuclide_composition = {}
    for nuclide_name, density in pairs:
        nuclide = _get_or_create_nuclide(nuclide_name, nearest_temperature, simulation)
        nuclide._compile_into_simulation(simulation)
        material.nuclide_composition[nuclide] = (
            material.nuclide_composition.get(nuclide, 0.0) + density
        )

    material.nuclides = list(material.nuclide_composition)
    material.nuclide_densities = np.asarray(
        list(material.nuclide_composition.values()), dtype=float64
    )


def set_elements_from_nuclides(material, simulation):
    """Build the element view: explicit elements plus the elements of nuclides."""
    # Nothing to collapse without nuclides
    if not any(is_nuclide_name(name) for name in material.number_density):
        return

    # Accumulate densities by exact element symbol (e.g., C12 -> C, Cr52 -> Cr)
    element_densities = {}
    for name, density in material.number_density.items():
        element_name = (
            element_symbol_from_nuclide_name(name) if is_nuclide_name(name) else name
        )
        element_densities[element_name] = (
            element_densities.get(element_name, 0.0) + density
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


def _number_density_from_mass(mass_density, weight):
    """Convert g/cm^3 to atoms/(barn cm) for an atomic weight in amu."""
    return mass_density * AVOGADRO_BARN_CM / weight


def _nuclide_weight(nuclide, simulation):
    """Atomic weight of a nuclide in amu, read from the data library."""
    nuclide._compile_into_simulation(simulation)
    return nuclide.atomic_weight_ratio * AMU_PER_NEUTRON_MASS


def _element_weight(element_name, temperature, simulation):
    """Atomic weight of a natural element in amu."""
    if element_name not in ISOTOPIC_ABUNDANCE:
        print_error(
            f"No natural isotopic abundance data for element {element_name}; "
            "specify its nuclides directly."
        )

    # Average the isotope weights by natural abundance. This needs the isotope
    # files, even for problems that never transport neutrons; reading the
    # element's own atomic_weight_ratio from its library file might be preferable.
    nearest_temperature = _get_supported_temperature(temperature)
    abundances = ISOTOPIC_ABUNDANCE[element_name]
    weighted = sum(
        abundance
        * _nuclide_weight(
            _get_or_create_nuclide(isotope, nearest_temperature, simulation),
            simulation,
        )
        for isotope, abundance in abundances.items()
    )
    return weighted / sum(abundances.values())


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
