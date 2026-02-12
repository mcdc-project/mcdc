import numpy as np
import os

from numpy import float64
from numpy.typing import NDArray
from types import NoneType
from typing import Annotated

####

from mcdc.constant import MATERIAL, MATERIAL_MG
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Material base class
# ======================================================================================


class MaterialBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "material"
    #
    name: str
    fissionable: bool

    def __init__(self, type_, name):
        super().__init__(type_)

        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.child_ID}"

        self.fissionable = False

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Fissionable: {self.fissionable}\n"
        return text


def decode_type(type_):
    if type_ == MATERIAL:
        return "Material"
    elif type_ == MATERIAL_MG:
        return "Multigroup material"


# ======================================================================================
# Native material
# ======================================================================================


class Material(MaterialBase):
    # Annotations for Numba mode
    label: str = "native_material"
    non_numba: list[str] = ["nuclide_composition"]
    #
    nuclide_composition: dict[Nuclide, float]
    #
    nuclides: list[Nuclide]
    nuclide_densities: NDArray[float64]

    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict[str, float] = {},
        temperature: float = 293.6,
    ):
        type_ = MATERIAL
        super().__init__(type_, name)

        # Temperature
        self.temperature = temperature

        # Dictionary connecting nuclides to respective densities
        self.nuclide_composition = {}

        # Numba representation of nuclide_composition
        self.nuclides = []
        self.nuclide_densities = np.zeros(len(nuclide_composition))

        # Check if library directory is set
        lib_dir = os.getenv("MCDC_LIB")
        if lib_dir is None:
            print_error("Environment variable MCDC_LIB is not set")

        # Loop over the items in the composition
        for i, (key, value) in enumerate(nuclide_composition.items()):
            nuclide_name = key
            nuclide_density = value

            # Get supported temperature
            nearest_temperature = min(TEMPERATURES, key=lambda x: abs(x - temperature))

            # Check if nuclide-temperature is available in the library
            file_name = f"{nuclide_name}-{nearest_temperature}K.h5"
            if not file_name in os.listdir(lib_dir):
                print_error(
                    f"Nuclide {nuclide_name} at temperature {nearest_temperature} K is not available in the library"
                )

            # Check if nuclide is already created
            found = False
            for nuclide in simulation.nuclides:
                if (
                    nuclide.name == nuclide_name
                    and nearest_temperature == nuclide.temperature
                ):
                    found = True
                    break

            # Create the nuclide to objects if needed
            if not found:
                nuclide = Nuclide(nuclide_name, nearest_temperature)

            # Register the nuclide composition
            self.nuclides.append(nuclide)
            self.nuclide_densities[i] = nuclide_density
            self.nuclide_composition[nuclide] = nuclide_density

            # Some flags
            if nuclide.fissionable:
                self.fissionable = True

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Temperature: {self.temperature} K\n"
        text += f"  - Nuclide composition [atoms/barn-cm]\n"
        for nuclide in self.nuclide_composition.keys():
            text += f"    - {nuclide.name:<5} | {self.nuclide_composition[nuclide]}\n"
        return text


# Currently supported temperatures
TEMPERATURES = [0.1, 233.15, 273.15, 293.6, 600.0, 900.0, 1200.0, 2500.0]


# ======================================================================================
# Multigroup material
# ======================================================================================


class MaterialMG(MaterialBase):
    # Annotations for Numba mode
    label: str = "multigroup_material"
    #
    G: int
    J: int
    mgxs_speed: Annotated[NDArray[float64], ("G",)]
    mgxs_decay_rate: Annotated[NDArray[float64], ("J",)]
    mgxs_capture: Annotated[NDArray[float64], ("G",)]
    mgxs_scatter: Annotated[NDArray[float64], ("G",)]
    mgxs_fission: Annotated[NDArray[float64], ("G",)]
    mgxs_total: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_s: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_p: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_d: Annotated[NDArray[float64], ("G", "J")]
    mgxs_nu_d_total: Annotated[NDArray[float64], ("G",)]
    mgxs_nu_f: Annotated[NDArray[float64], ("G",)]
    mgxs_chi_s: Annotated[NDArray[float64], ("G", "G")]
    mgxs_chi_p: Annotated[NDArray[float64], ("G", "G")]
    mgxs_chi_d: Annotated[NDArray[float64], ("J", "G")]

    def __init__(
        self,
        name: str = "",
        capture: NDArray[float64] | NoneType = None,
        scatter: NDArray[float64] | NoneType = None,
        fission: NDArray[float64] | NoneType = None,
        nu_s: NDArray[float64] | NoneType = None,
        nu_p: NDArray[float64] | NoneType = None,
        nu_d: NDArray[float64] | NoneType = None,
        chi_p: NDArray[float64] | NoneType = None,
        chi_d: NDArray[float64] | NoneType = None,
        speed: NDArray[float64] | NoneType = None,
        decay_rate: NDArray[float64] | NoneType = None,
    ):
        type_ = MATERIAL_MG
        super().__init__(type_, name)

        # Energy group size
        if capture is not None:
            G = len(capture)
        elif scatter is not None:
            G = len(scatter)
        elif fission is not None:
            G = len(fission)
        else:
            print_error("Need to supply capture, scatter, or fission for MaterialMG")
        self.G = G

        # Delayed group size
        J = 0
        if nu_d is not None:
            J = len(nu_d)
        self.J = J

        # Allocate the attributes
        self.mgxs_speed = np.ones(G)
        self.mgxs_decay_rate = np.ones(J) * np.inf
        self.mgxs_capture = np.zeros(G)
        self.mgxs_scatter = np.zeros(G)
        self.mgxs_fission = np.zeros(G)
        self.mgxs_total = np.zeros(G)
        self.mgxs_nu_s = np.ones(G)
        self.mgxs_nu_p = np.zeros(G)
        self.mgxs_nu_d = np.zeros([G, J])
        self.mgxs_nu_d_total = np.zeros([G])
        self.mgxs_nu_f = np.zeros(G)
        self.mgxs_chi_s = np.zeros([G, G])
        self.mgxs_chi_p = np.zeros([G, G])
        self.mgxs_chi_d = np.zeros([J, G])

        # Speed (vector of size G)
        if speed is not None:
            self.mgxs_speed = speed

        # Decay constant (vector of size J)
        if decay_rate is not None:
            self.mgxs_decay_rate = decay_rate

        # Cross-sections (vector of size G)
        if capture is not None:
            self.mgxs_capture = capture
        if scatter is not None:
            self.mgxs_scatter = np.sum(scatter, 0)
        if fission is not None:
            self.mgxs_fission = fission
            self.fissionable = True
        self.mgxs_total = self.mgxs_capture + self.mgxs_scatter + self.mgxs_fission

        # Scattering multiplication (vector of size G)
        if nu_s is not None:
            self.mgxs_nu_s = nu_s

        # Check if nu_p or nu_d is not provided, give fission
        if fission is not None:
            if nu_p is None and nu_d is None:
                print_error("Need to supply nu_p or nu_d for fissionable MaterialMG")

        # Prompt fission production (vector of size G)
        if nu_p is not None:
            self.mgxs_nu_p = nu_p

        # Delayed fission production (matrix of size GxJ)
        if nu_d is not None:
            # Transpose: [dg, gin] -> [gin, dg]
            self.mgxs_nu_d = np.swapaxes(nu_d, 0, 1)[:, :]
        self.mgxs_nu_d_total = np.sum(self.mgxs_nu_d, axis=1)

        # Total fission production (vector of size G)
        self.mgxs_nu_f = np.zeros_like(self.mgxs_nu_p)
        self.mgxs_nu_f += self.mgxs_nu_p
        for j in range(J):
            self.mgxs_nu_f += self.mgxs_nu_d[:, j]

        # Scattering spectrum (matrix of size GxG)
        if scatter is not None:
            # Transpose: [gout, gin] -> [gin, gout]
            self.mgxs_chi_s = np.swapaxes(scatter, 0, 1)[:, :]
            for g in range(G):
                if self.mgxs_scatter[g] > 0.0:
                    self.mgxs_chi_s[g, :] /= self.mgxs_scatter[g]

        # Prompt fission spectrum (matrix of size GxG)
        if nu_p is not None:
            if G == 1:
                self.mgxs_chi_p[:, :] = np.array([[1.0]])
            elif chi_p is None:
                print_error("Need to supply chi_p if nu_p is provided and G > 1")
            else:
                # Convert 1D spectrum to 2D
                if chi_p.ndim == 1:
                    tmp = np.zeros((G, G))
                    for g in range(G):
                        tmp[:, g] = chi_p
                    chi_p = tmp
                # Transpose: [gout, gin] -> [gin, gout]
                self.mgxs_chi_p[:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
                # Normalize
                for g in range(G):
                    if np.sum(self.mgxs_chi_p[g, :]) > 0.0:
                        self.mgxs_chi_p[g, :] /= np.sum(self.mgxs_chi_p[g, :])

        # Delayed fission spectrum (matrix of size JxG)
        if nu_d is not None:
            if G == 1:
                self.mgxs_chi_d = np.ones([J, G])
            else:
                if chi_d is None:
                    print_error("Need to supply chi_d if nu_d is provided and G > 1")
                # Transpose: [gout, dg] -> [dg, gout]
                self.mgxs_chi_d = np.swapaxes(chi_d, 0, 1)[:, :]
            # Normalize
            for dg in range(J):
                if np.sum(self.mgxs_chi_d[dg, :]) > 0.0:
                    self.mgxs_chi_d[dg, :] /= np.sum(self.mgxs_chi_d[dg, :])

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Multigroup data\n"
        text += f"    - G: {self.G}\n"
        text += f"    - J: {self.J}\n"
        text += f"    - Sigma_c {print_1d_array(self.mgxs_capture)}\n"
        text += f"    - Sigma_s {print_1d_array(self.mgxs_scatter)}\n"
        text += f"    - Sigma_f {print_1d_array(self.mgxs_fission)}\n"
        text += f"    - nu_s {print_1d_array(self.mgxs_nu_s)}\n"
        text += f"    - nu_p {print_1d_array(self.mgxs_nu_p)}\n"
        text += f"    - nu_d {print_1d_array(self.mgxs_nu_d.flatten())}\n"
        text += f"    - chi_s {print_1d_array(self.mgxs_chi_s.flatten())}\n"
        text += f"    - chi_fp {print_1d_array(self.mgxs_chi_p.flatten())}\n"
        text += f"    - chi_fd {print_1d_array(self.mgxs_chi_d.flatten())}\n"
        text += f"    - speed {print_1d_array(self.mgxs_speed)}\n"
        text += f"    - lambda {print_1d_array(self.mgxs_decay_rate)}\n"
        return text
