import numpy as np

from numpy import float64
from numpy.typing import NDArray

import mcdc.objects

from mcdc.nuclide import Nuclide
from mcdc.objects import ObjectBase


class Material(ObjectBase):
    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict = {},
        capture: NDArray[float64] = np.array([0.0]),
        scatter: NDArray[float64] = np.array([[0.0]]),
        fission: NDArray[float64] = np.array([0.0]),
        nu_s: NDArray[float64] = np.array([0.0]),
        nu_p: NDArray[float64] = np.array([0.0]),
        nu_d: NDArray[float64] = np.array([[0.0]]),
        chi_p: NDArray[float64] = np.array([[0.0]]),
        chi_d: NDArray[float64] = np.array([[0.0]]),
        speed: NDArray[float64] = np.array([0.0]),
        decay_rate: NDArray[float64] = np.array([0.0])
    ):
        super().__init__("material")

        self.name = name
        self.multigroup = False if len(nuclide_composition) == 0 else True

        # ==============================================================================
        # Continuous-energy material setup
        # ==============================================================================

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
            for nuclide in mcdc.objects.nuclides:
                if nuclide.name == nuclide_name:
                    found = True
                    break

            # Create and register the nuclide to objects if needed
            if not found:
                mcdc.objects.nuclides.append(Nuclide(nuclide_name))
                nuclide = mcdc.objects.nuclides[-1]

            # Register the nuclide composition
            self.nuclides.append(nuclide)
            self.atomic_densities[i] = atomic_density
            self.nuclide_composition[nuclide] = atomic_density

        # Register the material
        mcdc.objects.materials.append(self)
        
        # ==============================================================================
        # Multigroup material setup
        # ==============================================================================

        self.capture = capture
        self.scatter = scatter
        self.fission = fission
        self.nu_s = nu_s
        self.nu_p = nu_p
        self.nu_d = nu_d
        self.chi_p = chi_p
        self.chi_d = chi_d
        self.speed = speed
        self.decay_rate = decay_rate


    def __repr__(self):
        text = "\n"
        text += f"Material\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Nuclide composition [atoms/barn-cm]\n"
        for nuclide in self.nuclide_composition.keys():
            text += f"    - {nuclide.name:<5} | {self.nuclide_composition[nuclide]}\n"
        return text
