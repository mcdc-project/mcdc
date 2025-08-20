import numpy as np

from mcdc.constant import REACTION_CAPTURE, REACTION_ELASTIC_SCATTERING
from mcdc.objects import ObjectBase
from mcdc.prints import print_1d_array

class Reaction(ObjectBase):
    def __init__(self, type_, h5_group):
        super().__init__(type_, derived_class=True)

        self.xs = h5_group["xs"][()]

        if type_ == "capture_reaction":
            self.type_enum = REACTION_CAPTURE
        elif type_ == "elastic_scattering_reaction":
            self.type_enum = REACTION_ELASTIC_SCATTERING


class CaptureReaction(Reaction):
    def __init__(self, h5_group):
        super().__init__("capture_reaction", h5_group)

    def __repr__(self):
        text = "\n"
        text += f"Capture\n"
        text += f"  - XS {print_1d_array(self.xs)}\n"
        return text


class ElasticScatteringReaction(Reaction):
    def __init__(self, h5_group):
        super().__init__("elastic_scattering_reaction", h5_group)

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
                    self.mu_CDF[idx] + (PDF[idx] + PDF[idx + 1]) * (mu[idx + 1] - mu[idx]) * 0.5
                )
            # Ensure it ends at one
            self.mu_CDF[end - 1] = 1.0


    def __repr__(self):
        text = "\n"
        text += f"Elastic scattering\n"
        text += f"  - XS {print_1d_array(self.xs)}\n"
        text += f"  - Scattering cosine\n"
        text += f"    - mu_energy_grid {print_1d_array(self.mu_energy_grid)}\n"
        text += f"    - mu_energy_offset {print_1d_array(self.mu_energy_offset)}\n"
        text += f"    - mu {print_1d_array(self.mu)}\n"
        text += f"    - mu_PDF {print_1d_array(self.mu_PDF)}\n"
        return text


def decode_type_enum(type_enum):
    if type_enum == REACTION_CAPTURE:
        return "Capture"
    elif type_enum == REACTION_ELASTIC_SCATTERING:
        return "Elastic scattering"
