import numpy as np

from numpy import float64
from numpy.typing import NDArray

####

import mcdc.objects as objects

from mcdc.constant import MATERIAL, MATERIAL_MG
from mcdc.nuclide import Nuclide
from mcdc.objects import ObjectPolymorphic
from mcdc.prints import print_1d_array, print_error


class MaterialBase(ObjectPolymorphic):
    def __init__(self, label, type_, name):
        super().__init__(label, type_)
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
        decay_rate: NDArray[float64] = None
    ):
        label = "mg_material"
        type_ = MATERIAL_MG
        super().__init__(label, type_, name)
       
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
        self.speed = np.ones(G)
        self.decay_rate = np.ones(J) * np.inf
        self.capture = np.zeros(G)
        self.scatter = np.zeros(G)
        self.fission = np.zeros(G)
        self.total = np.zeros(G)
        self.nu_s = np.ones(G)
        self.nu_p = np.zeros(G)
        self.nu_d = np.zeros([G, J])
        self.nu_f = np.zeros(G)
        self.chi_s = np.zeros([G, G])
        self.chi_p = np.zeros([G, G])
        self.chi_d = np.zeros([J, G])

        # Speed (vector of size G)
        if speed is not None:
            self.speed = speed

        # Decay constant (vector of size J)
        if decay_rate is not None:
            self.decay_rate = decay_rate

        # Cross-sections (vector of size G)
        if capture is not None:
            self.capture = capture
        if scatter is not None:
            self.scatter = np.sum(scatter, 0)
        if fission is not None:
            self.fission = fission
            self.fissionable = True
        self.total = self.capture + self.scatter + self.fission

        # Scattering multiplication (vector of size G)
        if nu_s is not None:
            self.nu_s = nu_s

        # Check if nu_p or nu_d is not provided, give fission
        if fission is not None:
            if nu_p is None and nu_d is None:
                print_error("Need to supply nu_p or nu_d for fissionable MaterialMG")

        # Prompt fission production (vector of size G)
        if nu_p is not None:
            self.nu_p = nu_p

        # Delayed fission production (matrix of size GxJ)
        if nu_d is not None:
            # Transpose: [dg, gin] -> [gin, dg]
            self.nu_d = np.swapaxes(nu_d, 0, 1)[:, :]

        # Total fission production (vector of size G)
        self.nu_f = np.zeros_like(self.nu_p)
        self.nu_f += self.nu_p
        for j in range(J):
            self.nu_f += self.nu_d[:, j]

        # Scattering spectrum (matrix of size GxG)
        if scatter is not None:
            # Transpose: [gout, gin] -> [gin, gout]
            self.chi_s = np.swapaxes(scatter, 0, 1)[:, :]
            for g in range(G):
                if self.scatter[g] > 0.0:
                    self.chi_s[g, :] /= self.scatter[g]

        # Prompt fission spectrum (matrix of size GxG)
        if nu_p is not None:
            if G == 1:
                self.chi_p[:, :] = np.array([[1.0]])
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
                self.chi_p[:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
                # Normalize
                for g in range(G):
                    if np.sum(self.chi_p[g, :]) > 0.0:
                        self.chi_p[g, :] /= np.sum(self.chi_p[g, :])

        # Delayed fission spectrum (matrix of size JxG)
        if nu_d is not None:
            if G == 1:
                self.chi_d = np.ones([J, G])
            else:
                if chi_d is None:
                    print_error("Need to supply chi_d if nu_d is provided and G > 1")
                # Transpose: [gout, dg] -> [dg, gout]
                self.chi_d = np.swapaxes(chi_d, 0, 1)[:, :]
            # Normalize
            for dg in range(J):
                if np.sum(self.chi_d[dg, :]) > 0.0:
                    self.chi_d[dg, :] /= np.sum(self.chi_d[dg, :])
        
        # Register the material
        self.ID = len(objects.materials)
        objects.materials.append(self)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Multigroup data\n"
        text += f"    - G: {self.G}\n"
        text += f"    - J: {self.J}\n"
        text += f"    - Sigma_c {print_1d_array(self.capture)}\n"
        text += f"    - Sigma_s {print_1d_array(self.scatter)}\n"
        text += f"    - Sigma_f {print_1d_array(self.fission)}\n"
        text += f"    - nu_s {print_1d_array(self.nu_s)}\n"
        text += f"    - nu_p {print_1d_array(self.nu_p)}\n"
        text += f"    - nu_d {print_1d_array(self.nu_d.flatten())}\n"
        text += f"    - chi_s {print_1d_array(self.chi_s.flatten())}\n"
        text += f"    - chi_fp {print_1d_array(self.chi_p.flatten())}\n"
        text += f"    - chi_fd {print_1d_array(self.chi_d.flatten())}\n"
        text += f"    - speed {print_1d_array(self.speed)}\n"
        text += f"    - lambda {print_1d_array(self.decay_rate)}\n"
        return text


class Material(MaterialBase):
    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict = {},
    ):
        label = "material"
        type_ = MATERIAL
        super().__init__(label, type_, name)

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

            # Create and register the nuclide to objects if needed
            if not found:
                nuclide = Nuclide(nuclide_name)
                nuclide.ID = len(objects.nuclides)
                objects.nuclides.append(nuclide)

            # Register the nuclide composition
            self.nuclides.append(nuclide)
            self.atomic_densities[i] = atomic_density
            self.nuclide_composition[nuclide] = atomic_density

        # Register the material
        self.ID = len(objects.materials)
        objects.materials.append(self)
        

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Nuclide composition [atoms/barn-cm]\n"
        for nuclide in self.nuclide_composition.keys():
            text += f"    - {nuclide.name:<5} | {self.nuclide_composition[nuclide]}\n"
        return text


def decode_type(type_):
    if type_ == MATERIAL:
        return "Material"
    elif type_ == MATERIAL_MG:
        return "Multigroup material"
