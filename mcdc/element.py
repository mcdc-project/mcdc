import h5py
import numpy as np
import os

####

import mcdc.objects as objects

from mcdc.reaction import (
    ReactionElectronBremsstrahlung,
    ReactionElectronExcitation,
    ReactionElectronElasticScattering,
    ReactionElectronIonization,
    decode_type,
)
from mcdc.objects import ObjectNonSingleton
from mcdc.prints import print_1d_array

# ======================================================================================
# Element class
# ======================================================================================


class Element(ObjectNonSingleton):
    def __init__(self, element_name):
        label = "element"
        super().__init__(label)

        # Set attributes from the hdf5 file
        dir_name = os.getenv("MCDC_ELECTRON_XSLIB")
        file_name = f"{element_name}.h5"
        with h5py.File(f"{dir_name}/{file_name}", "r") as f:
            self.name = f["element_name"][()].decode()
            self.atomic_number = f["atomic_number"][()]
            self.xs_energy_grid = f["electron_reactions/xs_energy_grid"][()]

            self.reactions = []
            self.total_xs = np.zeros_like(self.xs_energy_grid)

            for reaction_type in f["electron_reactions"]:
                if reaction_type == "xs_energy_grid":
                    continue

                if reaction_type == "bremsstrahlung":
                    ReactionClass = ReactionElectronBremsstrahlung

                elif reaction_type == "excitation":
                    ReactionClass = ReactionElectronExcitation

                elif reaction_type == "elastic_scattering":
                    ReactionClass = ReactionElectronElasticScattering

                elif reaction_type == "ionization":
                    ReactionClass = ReactionElectronIonization

                reaction = ReactionClass(f[f"electron_reactions/{reaction_type}"])
                self.reactions.append(reaction)

                # Accumulate total XS
                self.total_xs += reaction.xs

    def __repr__(self):
        text = "\n"
        text += "Element\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic number (Z): {self.atomic_number}\n"
        text += f"  - XS energy grid {print_1d_array(self.xs_energy_grid)} eV\n"
        text += "  - Reactions\n"
        for reaction in self.reactions:
            text += f"    - {decode_type(reaction.type)}\n"
        return text
