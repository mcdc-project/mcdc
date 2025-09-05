import numpy as np

from numpy import float64
from numpy.typing import NDArray

####

import mcdc.objects as objects

from mcdc.constant import MATERIAL, MATERIAL_MG, MATERIAL_ELEMENTAL
from mcdc.nuclide import Nuclide
from mcdc.element import Element
from mcdc.objects import ObjectOverriding
from mcdc.prints import print_1d_array, print_error

# ======================================================================================
# Material classes
# ======================================================================================


class MaterialBase(ObjectOverriding):
    def __init__(self, type_, name):
        super().__init__("material", type_)
        self.name = name
        self.fissionable = False

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Fissionable: {self.fissionable}\n"
        return text


class MaterialMG(MaterialBase):
    def __init__(
        self,
        name: str = "",
        capture: NDArray[float64] = None,
        scatter: NDArray[float64] = None,
        fission: NDArray[float64] = None,
        nu_s: NDArray[float64] = None,
        nu_p: NDArray[float64] = None,
        nu_d: NDArray[float64] = None,
        chi_p: NDArray[float64] = None,
        chi_d: NDArray[float64] = None,
        speed: NDArray[float64] = None,
        decay_rate: NDArray[float64] = None,
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


class Material(MaterialBase):
    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict = {},
    ):
        type_ = MATERIAL
        super().__init__(type_, name)

        self.nuclides = []
        self.atomic_densities = np.zeros(len(nuclide_composition))

        # Helper dictionary connecting nuclides to respective atomic densities
        self.nuclide_composition = {}

        # Loop over the items in the composition
        for i, (key, value) in enumerate(nuclide_composition.items()):
            nuclide_name = key
            atomic_density = value

            # Check if nuclide is already created
            found = False
            for nuclide in objects.nuclides:
                if nuclide.name == nuclide_name:
                    found = True
                    break

            # Create the nuclide to objects if needed
            if not found:
                nuclide = Nuclide(nuclide_name)

            # Register the nuclide composition
            self.nuclides.append(nuclide)
            self.atomic_densities[i] = atomic_density
            self.nuclide_composition[nuclide] = atomic_density

            # Some flags
            if nuclide.fissionable:
                self.fissionable = True

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Nuclide composition [atoms/barn-cm]\n"
        for nuclide in self.nuclide_composition.keys():
            text += f"    - {nuclide.name:<5} | {self.nuclide_composition[nuclide]}\n"
        return text
    

class MaterialElemental(MaterialBase):
    def __init__(
        self,
        name: str = "",
        element_composition: dict = {},
    ):
        type_ = MATERIAL_ELEMENTAL
        super().__init__(type_, name)

        self.elements = []
        self.atomic_densities = np.zeros(len(element_composition))

        # Helper dictionary connecting elements to respective atomic densities
        self.element_composition = {}

        # Loop over the items in the composition
        for i, (key, value) in enumerate(element_composition.items()):
            element_name = key
            atomic_density = value

            # Check if element is already created
            found = False
            for element in objects.elements:
                if element.name == element_name:
                    found = True
                    break

            # Create the element to objects if needed
            if not found:
                element = Element(element_name)

            # Register the element composition
            self.elements.append(element)
            self.atomic_densities[i] = atomic_density
            self.element_composition[element] = atomic_density

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Element composition [atoms/barn-cm]\n"
        for element in self.element_composition.keys():
            text += f"    - {element.name:<5} | {self.element_composition[element]}\n"
        return text


def decode_type(type_):
    if type_ == MATERIAL:
        return "Material"
    elif type_ == MATERIAL_MG:
        return "Multigroup material"
    if type_ == MATERIAL_ELEMENTAL:
        return "Elemental material"