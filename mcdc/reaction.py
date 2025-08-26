import numpy as np

####

from mcdc.constant import (
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_EXCITATION,
)
from mcdc.objects import ObjectPolymorphic
from mcdc.prints import print_1d_array


class ReactionBase(ObjectPolymorphic):
    def __init__(self, label, type_, h5_group):
        super().__init__(label, type_)
        self.xs = h5_group["xs"][()]

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - XS {print_1d_array(self.xs)}\n"
        return text


# ======================================================================================
# Neutron reactions
# ======================================================================================


class ReactionNeutronCapture(ReactionBase):
    def __init__(self, h5_group):
        label = "neutron_capture_reaction"
        type_ = REACTION_NEUTRON_CAPTURE
        super().__init__(label, type_, h5_group)


class ReactionNeutronElasticScattering(ReactionBase):
    def __init__(self, h5_group):
        label = "neutron_elastic_scattering_reaction"
        type_ = REACTION_NEUTRON_ELASTIC_SCATTERING
        super().__init__(label, type_, h5_group)

        # Scattering cosine
        base = "scattering_cosine"
        self.mu_energy_grid = h5_group[f"{base}/energy_grid"][()]
        self.mu_energy_offset = h5_group[f"{base}/energy_offset"][()]
        self.mu = h5_group[f"{base}/value"][()]
        self.mu_PDF = h5_group[f"{base}/PDF"][()]

        # CDF
        mu = h5_group[f"{base}/value"][()]
        PDF = h5_group[f"{base}/PDF"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        self.mu_CDF = np.zeros_like(PDF)
        for i in range(len(offset)):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(PDF)
            for idx in range(start, end - 2):
                self.mu_CDF[idx + 1] = (
                    self.mu_CDF[idx]
                    + (PDF[idx] + PDF[idx + 1]) * (mu[idx + 1] - mu[idx]) * 0.5
                )
            # Ensure it ends at one
            self.mu_CDF[end - 1] = 1.0

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Scattering cosine\n"
        text += f"    - mu_energy_grid {print_1d_array(self.mu_energy_grid)}\n"
        text += f"    - mu_energy_offset {print_1d_array(self.mu_energy_offset)}\n"
        text += f"    - mu {print_1d_array(self.mu)}\n"
        text += f"    - mu_PDF {print_1d_array(self.mu_PDF)}\n"
        return text


# ======================================================================================
# Electron reactions
# ======================================================================================


def decode_type(type_):
    if type_ == REACTION_NEUTRON_CAPTURE:
        return "Neutron capture"
    elif type_ == REACTION_NEUTRON_ELASTIC_SCATTERING:
        return "Neulastic scattering"
    elif type_ == REACTION_NEUTRON_FISSION:
        return "Neutron fission"
    elif type_ == REACTION_ELECTRON_BREMSSTRAHLUNG:
        return "Electron bremsstrahlung"
    elif type_ == REACTION_ELECTRON_EXCITATION:
        return "Electron excitation"

class ReactionElectronBremsstrahlung(ReactionBase):
    def __init__(self, h5_group):
        label = "electron_bremsstrahlung_reaction"
        type_ = REACTION_ELECTRON_BREMSSTRAHLUNG
        super().__init__(label, type_, h5_group)

        # Energy loss
        base = "energy_loss"
        self.eloss_energy_grid = h5_group[f"{base}/energy_grid"][()]
        self.eloss_energy_offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss = h5_group[f"{base}/value"][()]
        self.eloss_PDF = h5_group[f"{base}/PDF"][()]

        # CDF
        eloss = h5_group[f"{base}/value"][()]
        PDF = h5_group[f"{base}/PDF"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss_CDF = np.zeros_like(PDF)
        for i in range(len(offset)):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(PDF)
            for idx in range(start, end-2):
                self.eloss_CDF[idx + 1] = (
                    self.eloss_CDF[idx]
                    + (PDF[idx] + PDF[idx + 1]) * (eloss[idx + 1] - eloss[idx]) * 0.5
                )
            # Ensure it ends at one
            self.eloss_CDF[end - 1] = 1.0

    def __repr__(self):
        text =  super().__repr__()
        text += " - Energy loss\n"
        text += f"   - eloss_energy_grid {print_1d_array(self.eloss_energy_grid)}\n"
        text += f"   - eloss_energy_offset {print_1d_array(self.eloss_energy_offset)}\n"
        text += f"   - eloss {print_1d_array(self.eloss)}\n"
        text += f"   - eloss_PDF {print_1d_array(self.eloss_PDF)}\n"
        return text

class ReactionElectronExcitation(ReactionBase):
    def __init__(self, h5_group):
        label = "electron_excitation_reaction"
        type_ = REACTION_ELECTRON_EXCITATION
        super().__init__(label, type_, h5_group)

        # Energy loss
        base = "energy_loss"
        self.eloss_energy_grid = h5_group[f"{base}/energy_grid"][()]
        self.eloss_energy_offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss = h5_group[f"{base}/value"][()]
        self.eloss_PDF = h5_group[f"{base}/PDF"][()]

        # CDF
        eloss = h5_group[f"{base}/value"][()]
        PDF = h5_group[f"{base}/PDF"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss_CDF = np.zeros_like(PDF)
        for i in range(len(offset)):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(PDF)
            for idx in range(start, end-2):
                self.eloss_CDF[idx + 1] = (
                    self.eloss_CDF[idx]
                    + (PDF[idx] + PDF[idx + 1]) * (eloss[idx + 1] - eloss[idx]) * 0.5
                )
            # Ensure it ends at one
            self.eloss_CDF[end - 1] = 1.0

    def __repr__(self):
        text =  super().__repr__()
        text += " - Energy loss\n"
        text += f"   - eloss_energy_grid {print_1d_array(self.eloss_energy_grid)}\n"
        text += f"   - eloss_energy_offset {print_1d_array(self.eloss_energy_offset)}\n"
        text += f"   - eloss {print_1d_array(self.eloss)}\n"
        text += f"   - eloss_PDF {print_1d_array(self.eloss_PDF)}\n"
        return text
