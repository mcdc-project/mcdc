import h5py
import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import INTERPOLATION_LINEAR
from mcdc.object_.base import MCDCObject
from mcdc.object_.data import DataBase, DataNone, DataTable
from mcdc.object_.hdf5_reader import (
    check_format,
    read_xs_energy_grid,
    text,
    xs_is_linear,
)
from mcdc.object_.element import Element, get_or_create_element
from mcdc.object_.util import element_symbol_from_nuclide_name
from mcdc.object_.neutron_reaction import (
    NeutronReactionBase,
    NeutronReactionCapture,
    NeutronReactionElasticScattering,
    NeutronReactionFission,
    NeutronReactionInelasticScattering,
)
from mcdc.object_.proton_reaction import (
    ProtonReactionElasticScattering,
    ProtonReactionInelasticScattering,
    ProtonReactionCapture,
)
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Nuclide
# ======================================================================================


class Nuclide(MCDCObject):
    """Temperature-specific nuclide definition for the MC/DC HDF5 library.

    Parameters
    ----------
    nuclide_name : str
        Nuclide identifier, such as ``"U235"``.
    temperature : float
        Library temperature in kelvin.

    Notes
    -----
    Construction records the nuclide identity without accessing the data
    library. Basic properties are loaded when the nuclide is compiled into a
    simulation. Neutron cross sections, reactions, multiplicities, and
    delayed-neutron data are loaded later by :meth:`set_neutron_data`.
    Element-wide properties are accessed through the shared :attr:`element`
    reference, which is packed as ``element_ID`` for transport.
    """

    # MC/DC framework metadata
    label = "nuclide"

    name: str
    temperature: float
    element: Element
    mass_number: int
    atomic_weight_ratio: float
    fissionable: bool
    excitation_level: int
    radiation_length: float
    #
    neutron_xs_is_linear: bool
    neutron_reactions: list[NeutronReactionBase]
    neutron_xs_energy_grid: NDArray[float64]
    neutron_total_xs: NDArray[float64]
    neutron_elastic_xs: NDArray[float64]
    neutron_capture_xs: NDArray[float64]
    neutron_inelastic_xs: NDArray[float64]
    neutron_fission_xs: NDArray[float64]
    #
    proton_xs_energy_grid: NDArray[float64]
    proton_total_xs: NDArray[float64]
    proton_elastic_xs: NDArray[float64]
    proton_capture_xs: NDArray[float64]
    proton_inelastic_xs: NDArray[float64]
    #
    neutron_elastic_scattering_reactions: list[NeutronReactionElasticScattering]
    neutron_capture_reactions: list[NeutronReactionCapture]
    neutron_inelastic_scattering_reactions: list[NeutronReactionInelasticScattering]
    neutron_fission_reactions: list[NeutronReactionFission]
    #
    proton_elastic_scattering_reactions: list[ProtonReactionElasticScattering]
    proton_capture_reactions: list[ProtonReactionCapture]
    proton_inelastic_scattering_reactions: list[ProtonReactionInelasticScattering]
    #
    stopping_power: DataBase

    def __init__(self, nuclide_name, temperature):
        super().__init__()

        self.name = nuclide_name
        self.temperature = temperature
        self.element = Element(element_symbol_from_nuclide_name(nuclide_name))

        # Initialize all attributes
        # Neutron XS
        self.neutron_xs_energy_grid = np.zeros(0)
        self.neutron_total_xs = np.zeros(0)
        self.neutron_elastic_xs = np.zeros(0)
        self.neutron_capture_xs = np.zeros(0)
        self.neutron_inelastic_xs = np.zeros(0)
        self.neutron_fission_xs = np.zeros(0)
        # Proton XS
        self.proton_xs_energy_grid = np.zeros(0)
        self.proton_total_xs = np.zeros(0)
        self.proton_elastic_xs = np.zeros(0)
        self.proton_inelastic_xs = np.zeros(0)
        self.proton_capture_xs = np.zeros(0)
        # Reactions
        self.neutron_elastic_scattering_reactions = []
        self.neutron_reactions = []
        self.neutron_capture_reactions = []
        self.neutron_inelastic_scattering_reactions = []
        self.neutron_fission_reactions = []
        self.proton_elastic_scattering_reactions = []
        self.proton_inelastic_scattering_reactions = []
        self.proton_capture_reactions = []
        self.neutron_xs_is_linear = True
        # Stopping Power
        self.stopping_power = DataNone()
        self.radiation_length = 0.0

    def _compile_into_simulation(self, simulation) -> bool:
        """Load basic properties and register with the owning simulation."""
        if self.compile_ID == simulation.compile_ID:
            return False

        dir_name = os.getenv("MCDC_LIB")
        if dir_name is None:
            print_error("Environment variable MCDC_LIB is not set")

        file_name = f"{self.name}-{self.temperature}K.h5"
        file_path = os.path.join(dir_name, file_name)
        if not os.path.isfile(file_path):
            print_error(
                f"Nuclide {self.name} at temperature {self.temperature} K "
                "is not available in the library"
            )

        with h5py.File(file_path, "r") as file:
            if "format" in file.attrs:
                check_format(file)
                stored_temperature = float(file["temperature"][()])
                if not np.isclose(stored_temperature, self.temperature):
                    raise ValueError(
                        f"{file_path}: temperature does not match its filename"
                    )
                with h5py.File(
                    os.path.join(dir_name, f"{self.name}.h5"), "r"
                ) as nuclide:
                    check_format(nuclide)
                    self.mass_number = int(nuclide["mass-number"][()])
                    self.atomic_weight_ratio = float(nuclide["atomic-weight-ratio"][()])
                    self.fissionable = bool(nuclide["fissionable"][()])
                    self.excitation_level = int(nuclide["excitation-level"][()])
            else:
                # TODO: Remove this fallback once proton libraries use mcdc-data.
                # Proton libraries carry their identity in the temperature file.
                self.mass_number = int(file["mass_number"][()])
                self.atomic_weight_ratio = float(file["atomic_weight_ratio"][()])
                self.fissionable = bool(file["fissionable"][()])
                self.excitation_level = int(file["excitation_level"][()])
        self.element = get_or_create_element(
            self.element.name, simulation, self.element
        )
        return super()._compile_into_simulation(simulation)

    def set_neutron_data(self, simulation):
        """Load neutron reactions by category and register their owned data.

        Cross sections remain on the projectile's shared energy grid. Cached
        category sums support linear tables; nonlinear tables are evaluated
        reaction by reaction during transport so their interpolation is exact.
        Fission yields and delayed groups belong to each fission reaction.
        """
        path = os.path.join(
            os.environ["MCDC_LIB"], f"{self.name}-{self.temperature}K.h5"
        )
        categories = {
            "elastic-scattering": (
                "elastic",
                "elastic_scattering",
                NeutronReactionElasticScattering,
            ),
            "capture": ("capture", "capture", NeutronReactionCapture),
            "inelastic-scattering": (
                "inelastic",
                "inelastic_scattering",
                NeutronReactionInelasticScattering,
            ),
            "fission": ("fission", "fission", NeutronReactionFission),
        }
        with h5py.File(path, "r") as file:
            check_format(file)
            neutron = file["projectiles/neutron"]
            grid = read_xs_energy_grid(neutron)
            self.neutron_xs_energy_grid = grid
            self.neutron_xs_is_linear = True
            self.neutron_reactions = []
            for xs_name, rx_name, _ in categories.values():
                setattr(self, f"neutron_{xs_name}_xs", np.zeros_like(grid))
                setattr(self, f"neutron_{rx_name}_reactions", [])
            for group in neutron["reactions"].values():
                category = text(group.attrs["category"])
                if category not in categories:
                    raise ValueError(
                        f"{group.name}: unknown neutron reaction category {category!r}"
                    )
                xs_name, rx_name, cls = categories[category]
                reaction = cls.from_h5_group(group, simulation)
                if reaction.xs_offset_ + len(reaction.xs) != len(grid):
                    raise ValueError(
                        f"{group.name}: cross-section segment does not end at the shared grid boundary"
                    )
                getattr(self, f"neutron_{xs_name}_xs")[
                    reaction.xs_offset_ :
                ] += reaction.xs
                getattr(self, f"neutron_{rx_name}_reactions").append(reaction)
                self.neutron_reactions.append(reaction)
                self.neutron_xs_is_linear &= xs_is_linear(reaction)
            self.neutron_total_xs = sum(
                getattr(self, f"neutron_{name}_xs")
                for name in ("elastic", "capture", "inelastic", "fission")
            )

        # Register the complete reaction trees after the library file is closed.
        for _, name, _ in categories.values():
            for reaction in getattr(self, f"neutron_{name}_reactions"):
                reaction._compile_into_simulation(simulation)

    def set_proton_data(self, simulation):
        nuclide_name = self.name
        temperature = self.temperature

        # Load data library
        dir_name = os.getenv("MCDC_LIB")
        file_name = f"{nuclide_name}-{temperature}K.h5"
        file = h5py.File(f"{dir_name}/{file_name}", "r")

        # Radiation length
        self.radiation_length = float(file["radiation_length"][()])

        # ==========================================================================
        # Stopping power for protons
        # ==========================================================================

        if "stopping_power" in file:
            self.stopping_power = DataTable(
                file["stopping_power"]["energy"][()],
                file["stopping_power"]["total_stopping_power"][()],
                INTERPOLATION_LINEAR,
            )
            # Register data loaded during finalization.
            self.stopping_power._compile_into_simulation(simulation)
        elif simulation.settings.condensed_interactions.proton:
            raise ValueError(
                f"Proton condensed history requires stopping power for nuclide {self.name}"
            )

        # Only condensed history data available - no nuclear rxn xs
        if "proton_reactions" not in file:
            # Zero out all xs arrays
            xs_energy = np.array([0, 1.0e10])
            self.proton_xs_energy_grid = xs_energy

            self.proton_total_xs = np.zeros_like(self.proton_xs_energy_grid)
            self.proton_elastic_xs = np.zeros_like(self.proton_xs_energy_grid)
            self.proton_inelastic_xs = np.zeros_like(self.proton_xs_energy_grid)

            file.close()
            return

        rx_names = [
            "elastic_scattering",
            "inelastic_scattering",
            "capture",
        ]

        # The reaction MTs
        MTs = {}
        for name in rx_names:
            if name not in file["proton_reactions"]:
                MTs[name] = []
                continue

            MTs[name] = [
                x for x in file[f"proton_reactions/{name}"] if x.startswith("MT")
            ]

        # ==========================================================================
        # Reaction XS
        # ==========================================================================

        # Energy grid
        xs_energy = file["proton_reactions/xs_energy_grid"][()] * 1e6  # MeV to eV
        self.proton_xs_energy_grid = xs_energy

        # The total XS
        self.proton_total_xs = np.zeros_like(self.proton_xs_energy_grid)
        self.proton_elastic_xs = np.zeros_like(self.proton_xs_energy_grid)
        self.proton_inelastic_xs = np.zeros_like(self.proton_xs_energy_grid)
        self.proton_capture_xs = np.zeros_like(self.proton_xs_energy_grid)

        xs_containers = [
            self.proton_elastic_xs,
            self.proton_inelastic_xs,
            self.proton_capture_xs,
        ]

        for xs_container, rx_name in list(zip(xs_containers, rx_names)):
            for MT in MTs[rx_name]:
                xs = file[f"proton_reactions/{rx_name}/{MT}/xs"]
                xs_container[xs.attrs["offset"] :] += xs[()]

        self.proton_total_xs = (
            self.proton_elastic_xs + self.proton_inelastic_xs + self.proton_capture_xs
        )

        # ==========================================================================
        # The reactions
        # ==========================================================================

        self.proton_elastic_scattering_reactions = []
        self.proton_inelastic_scattering_reactions = []
        self.proton_capture_reactions = []

        rx_containers = [
            self.proton_elastic_scattering_reactions,
            self.proton_inelastic_scattering_reactions,
            self.proton_capture_reactions,
        ]
        rx_classes = [
            ProtonReactionElasticScattering,
            ProtonReactionInelasticScattering,
            ProtonReactionCapture,
        ]
        for rx_container, rx_name, rx_class in list(
            zip(rx_containers, rx_names, rx_classes)
        ):
            for MT in MTs[rx_name]:
                h5_group = file[f"proton_reactions/{rx_name}/{MT}"]
                reaction = rx_class.from_h5_group(h5_group, simulation)
                rx_container.append(reaction)

        file.close()

        # Register reactions and their data loaded during finalization.
        for reaction_container in rx_containers:
            for reaction in reaction_container:
                reaction._compile_into_simulation(simulation)

    def __repr__(self):
        text = "\n"
        text += f"Nuclide\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Element: {self.element.name} [ID: {self.element.ID}]\n"
        text += f"  - Mass number: {self.mass_number}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        text += f"  - Reaction MTs\n"
        text += f"    - Elastic scattering: {[int(x.MT) for x in self.neutron_elastic_scattering_reactions]}\n"
        text += (
            f"    - Capture: {[int(x.MT) for x in self.neutron_capture_reactions]}\n"
        )
        text += f"    - Inelastic scattering: {[int(x.MT) for x in self.neutron_inelastic_scattering_reactions]}\n"
        if self.fissionable:
            text += f"    - Fission: {[int(x.MT) for x in self.neutron_fission_reactions]}\n"
        text += f"  - Reaction cross-sections (eV, barns)\n"
        text += f"    - Energy grid {print_1d_array(self.neutron_xs_energy_grid)}\n"
        text += f"    - Total {print_1d_array(self.neutron_total_xs)}\n"
        text += f"    - Elastic scattering {print_1d_array(self.neutron_elastic_xs)}\n"
        text += f"    - Capture {print_1d_array(self.neutron_capture_xs)}\n"
        text += (
            f"    - Inelastic scattering {print_1d_array(self.neutron_inelastic_xs)}\n"
        )
        if self.fissionable:
            text += f"    - Fission {print_1d_array(self.neutron_fission_xs)}\n"
        return text
