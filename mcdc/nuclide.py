import h5py
import numpy as np
import os

####

from mcdc.reaction import (
    ReactionNeutronCapture,
    ReactionNeutronElasticScattering,
    ReactionNeutronFission,
    decode_type,
)
from mcdc.objects import ObjectNonSingleton
from mcdc.prints import print_1d_array

# ======================================================================================
# Nuclide class
# ======================================================================================


class Nuclide(ObjectNonSingleton):
    def __init__(self, nuclide_name):
        label = "nuclide"
        super().__init__(label)

        # Set attributes from the hdf5 file
        dir_name = os.getenv("MCDC_XSLIB")
        file_name = f"{nuclide_name}.h5"
        with h5py.File(f"{dir_name}/{file_name}", "r") as f:
            self.name = f["nuclide_name"][()].decode()
            self.atomic_weight_ratio = f["atomic_weight_ratio"][()]
            self.xs_energy_grid = f["neutron_reactions/xs_energy_grid"][()]

            self.fissionable = False
            self.reactions = []
            self.total_xs = np.zeros_like(self.xs_energy_grid)

            for reaction_type in f["neutron_reactions"]:
                if reaction_type == "xs_energy_grid":
                    continue

                if reaction_type == "capture":
                    ReactionClass = ReactionNeutronCapture

                elif reaction_type == "elastic_scattering":
                    ReactionClass = ReactionNeutronElasticScattering

                elif reaction_type == "fission":
                    self.fissionable = True
                    ReactionClass = ReactionNeutronFission

                reaction = ReactionClass.from_h5_group(f[f"neutron_reactions/{reaction_type}"])
                self.reactions.append(reaction)

                # Accumulate total XS
                self.total_xs += reaction.xs

    def __repr__(self):
        text = "\n"
        text += f"Nuclide\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        text += f"  - XS energy grid {print_1d_array(self.xs_energy_grid)} eV\n"
        text += f"  - Reactions\n"
        for reaction in self.reactions:
            text += f"    - {decode_type(reaction.type)}\n"
        return text
