from typing import Annotated
import numpy as np
from numpy import float64
from numpy.typing import NDArray

####

import mcdc.object_.distribution as distribution

from mcdc.constant import (
    ANGLE_ISOTROPIC,
    ANGLE_ENERGY_CORRELATED,
    ANGLE_DISTRIBUTED,
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOG,
    PROTON_REACTION_ELASTIC_SCATTERING,
    PROTON_REACTION_CAPTURE,
    PROTON_REACTION_INELASTIC_SCATTERING,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
    PARTICLE_NEUTRON,
    PARTICLE_PROTON,
)
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.distribution import (
    DistributionBase,
    DistributionMultiTable,
    DistributionLevelScattering,
    DistributionEvaporation,
    DistributionMaxwellian,
    DistributionKalbachMann,
    DistributionTabulatedEnergyAngle,
    DistributionNBody,
)
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Proton reaction base class
# ======================================================================================


class ProtonReactionBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "proton_reaction"
    #
    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" ir reserved for "xs"
    reference_frame: int
    q_value: float64

    def __init__(self, type_, MT, xs, xs_offset, reference_frame, q_value):
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.reference_frame = reference_frame
        self.q_value = q_value
        super().__init__(type_)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - MT: {self.MT}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
        text += f"  - Reference frame: {decode_reference_frame(self.reference_frame)}\n"
        text += f"  - Q-value: {self.q_value}\n"
        return text


def decode_type(type_):
    if type_ == PROTON_REACTION_ELASTIC_SCATTERING:
        return "Proton elastic scattering"
    elif type_ == PROTON_REACTION_INELASTIC_SCATTERING:
        return "Proton inelastic scattering"
    elif type_ == PROTON_REACTION_CAPTURE:
        return "Proton capture"


def decode_reference_frame(type_):
    if type_ == REFERENCE_FRAME_LAB:
        return "Laboratory"
    elif type_ == REFERENCE_FRAME_COM:
        return "Center of mass"


# ======================================================================================
# Proton elastic scattering
# ======================================================================================


class ProtonReactionElasticScattering(ProtonReactionBase):
    # Annotations for Numba mode
    label: str = "proton_elastic_scattering_reaction"
    #
    mu_table: DistributionBase

    def __init__(self, MT, xs, xs_offset, reference_frame, mu):
        type_ = PROTON_REACTION_ELASTIC_SCATTERING
        self.mu_table = mu
        super().__init__(type_, MT, xs, xs_offset, reference_frame, 0.0)

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame, _ = set_basic_properties(h5_group)
        _, mu = set_angular_distribution(h5_group["angular_cosine_distribution"])
        return cls(MT, xs, xs_offset, reference_frame, mu)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Scattering cosine: {distribution.decode_type(self.mu_table.type)} [ID: {self.mu_table.ID}]\n"
        return text


# ======================================================================================
# Proton inelastic scattering
# ======================================================================================


class ProtonReactionInelasticScattering(ProtonReactionBase):
    # Annotations for Numba mode
    label: str = "proton_inelastic_scattering_reaction"
    #
    multiplicity: int
    angle_type: int
    mu: DistributionBase
    N_spectrum_probability_bin: int
    N_spectrum: int
    spectrum_probability_grid: NDArray[float64]
    spectrum_probability: Annotated[
        NDArray[float64], ("N_spectrum_probability_bin", "N_spectrum")
    ]
    energy_spectra: list[DistributionBase]

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        q_value,
        multiplicity,
        angle_type,
        mu,
        spectrum_probability_grid,
        spectrum_probability,
        energy_spectra,
    ):
        type_ = PROTON_REACTION_INELASTIC_SCATTERING
        super().__init__(type_, MT, xs, xs_offset, reference_frame, q_value)

        self.multiplicity = multiplicity
        self.angle_type = angle_type
        self.mu = mu
        self.N_spectrum_probability_bin = len(spectrum_probability_grid) - 1
        self.N_spectrum = len(energy_spectra)
        self.spectrum_probability_grid = spectrum_probability_grid
        self.spectrum_probability = spectrum_probability
        self.energy_spectra = energy_spectra

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame, q_value = set_basic_properties(h5_group)
        multiplicity = int(h5_group["multiplicity"][()])

        angle_type, mu = set_angular_distribution(
            h5_group["angular_cosine_distribution"]
        )

        # Energy spectra
        spectrum_probability_grid = (
            h5_group[f"spectrum_probability_grid"][()] * 1e6
        )  # MeV to eV
        spectrum_probability = h5_group[f"spectrum_probability"][()]
        energy_spectra = []
        spectrum_names = [x for x in h5_group if x.startswith("energy_spectrum-")]
        for spectrum_name in spectrum_names:
            energy_spectra.append(set_energy_distribution(h5_group[f"{spectrum_name}"]))

        return cls(
            MT,
            xs,
            xs_offset,
            reference_frame,
            q_value,
            multiplicity,
            angle_type,
            mu,
            spectrum_probability_grid,
            spectrum_probability,
            energy_spectra,
        )

    def __repr__(self):
        text = super().__repr__()
        if self.angle_type == ANGLE_ISOTROPIC:
            text += f"  - Scattering cosine: Isotropic\n"
        elif self.angle_type == ANGLE_ENERGY_CORRELATED:
            text += f"  - Scattering cosine: Energy-correlated\n"
        else:
            text += f"  - Scattering cosine: {distribution.decode_type(self.mu.type)} [ID: {self.mu.ID}]\n"
        text += f"  - Energy spectra\n"
        text += f"      - Probability energy grid {print_1d_array(self.spectrum_probability_grid)}\n"
        for i in range(len(self.energy_spectra)):
            text += f"      - Spectrum {i+1}: {distribution.decode_type(self.energy_spectra[i])} [{print_1d_array(self.spectrum_probability[:,i])}] [ID: {self.energy_spectra[i].ID}]\n"
        return text



class ProtonReactionCapture(ProtonReactionBase):
    # Annotations for Numba mode
    label: str = "proton_capture_reaction"

    def __init__(self, MT, xs, xs_offset, reference_frame, q_value):
        type_ = PROTON_REACTION_CAPTURE
        super().__init__(type_, MT, xs, xs_offset, reference_frame, q_value)

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame, q_value = set_basic_properties(h5_group)
        return cls(MT, xs, xs_offset, reference_frame, q_value)



# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    MT = int(h5_group.attrs["MT"][()])
    xs = h5_group["xs"][()]
    xs_offset = h5_group["xs"].attrs["offset"]
    reference_frame = h5_group["reference_frame"][()].decode("utf-8")
    if reference_frame == "LAB":
        reference_frame = REFERENCE_FRAME_LAB
    elif reference_frame == "COM":
        reference_frame = REFERENCE_FRAME_COM
    q_value = h5_group["Q-value"][()]
    return MT, xs, xs_offset, reference_frame, q_value


def set_angular_distribution(h5_group):
    # Handle missing type attribute
    if "type" not in h5_group.attrs:
        mu_type = "isotropic"
    else:
        mu_type = h5_group.attrs["type"]

    if mu_type == "isotropic":
        angle_type = ANGLE_ISOTROPIC
        mu = simulation.distributions[0]
    elif mu_type == "energy-correlated":
        angle_type = ANGLE_ENERGY_CORRELATED
        mu = simulation.distributions[0]
    elif mu_type == "given_in_energy_distribution":
        # Angular information comes from the Kalbach-Mann energy distribution.
        angle_type = ANGLE_ENERGY_CORRELATED
        mu = simulation.distributions[0]
    elif mu_type == "tabulated":
        angle_type = ANGLE_DISTRIBUTED

        # Check if data is in flattened format or subgroup format
        if "energy" in h5_group:
            # Flattened format
            grid = h5_group[f"energy"][()] * 1e6  # MeV to eV
            offset = h5_group[f"offset"][()]
            value = h5_group[f"value"][()]
            pdf = h5_group[f"pdf"][()]
        else:
            # Subgroup format: E_in_1, E_in_2, etc.
            incident_energies = h5_group["incident_energies"][()] * 1e6  # MeV to eV

            # Collect all cosines and pdfs into flattened arrays
            cosines_list = []
            pdf_list = []
            offset = np.zeros(len(incident_energies), dtype=np.int32)

            for i, energy in enumerate(incident_energies):
                subgroup_name = f"E_in_{i + 1}"
                if subgroup_name in h5_group:
                    subgroup = h5_group[subgroup_name]
                    if subgroup.attrs.get("type", "tabulated") == "tabulated":
                        cosines_list.extend(subgroup["cosines"][()])
                        pdf_list.extend(subgroup["pdf"][()])
                    else:
                        # Isotropic - use dummy values
                        cosines_list.extend([0.0])  # isotropic cosine
                        pdf_list.extend([1.0])  # uniform pdf
                else:
                    # Missing subgroup - assume isotropic
                    cosines_list.extend([0.0])
                    pdf_list.extend([1.0])

                if i < len(incident_energies) - 1:
                    offset[i + 1] = len(cosines_list)

            grid = incident_energies
            value = np.array(cosines_list)
            pdf = np.array(pdf_list)

        mu = DistributionMultiTable(grid, offset, value, pdf)

    return angle_type, mu


def set_energy_distribution(h5_group):
    spectrum_type = h5_group.attrs["type"]

    if spectrum_type == "tabulated":
        grid = h5_group[f"energy"][()] * 1e6  # MeV to eV
        offset = h5_group[f"offset"][()]
        value = h5_group[f"value"][()] * 1e6  # MeV to eV
        pdf = h5_group[f"pdf"][()] / 1e6  # /MeV to /eV
        energy_spectrum = DistributionMultiTable(grid, offset, value, pdf)

    elif spectrum_type == "level-scattering":
        C1 = h5_group["C1"][()] * 1e6  # MeV to eV
        C2 = h5_group["C2"][()]

        energy_spectrum = DistributionLevelScattering(C1, C2)

    elif spectrum_type == "evaporation":
        energy = h5_group[f"temperature_energy_grid"][()] * 1e6  # MeV to eV
        temperature = h5_group[f"temperature"][()] * 1e6  # MeV to eV
        restriction_energy = h5_group[f"restriction_energy"][()] * 1e6  # MeV to eV

        energy_spectrum = DistributionEvaporation(
            energy, temperature, restriction_energy
        )

    elif spectrum_type == "maxwellian":
        energy = h5_group[f"temperature_energy_grid"][()] * 1e6  # MeV to eV
        temperature = h5_group[f"temperature"][()] * 1e6  # MeV to eV
        restriction_energy = h5_group[f"restriction_energy"][()] * 1e6  # MeV to eV
        interpolation = h5_group[f"temperature_interpolation"][()].decode("utf-8")
        if interpolation == "linear":
            interpolation = INTERPOLATION_LINEAR
        elif interpolation == "log":
            interpolation = INTERPOLATION_LOG

        energy_spectrum = DistributionMaxwellian(
            energy, temperature, restriction_energy, interpolation
        )

    elif spectrum_type == "kalbach-mann":
        energy = h5_group[f"energy"][()] * 1e6  # MeV to eV
        offset = h5_group[f"offset"][()]

        energy_out = h5_group[f"energy_out"][()] * 1e6  # MeV to eV
        pdf = h5_group[f"pdf"][()] / 1e6  # /MeV to /eV

        precompound_factor = h5_group[f"precompound_factor"][()]
        angular_slope = h5_group[f"angular_slope"][()]

        energy_spectrum = DistributionKalbachMann(
            energy, offset, energy_out, pdf, precompound_factor, angular_slope
        )

    elif spectrum_type == "energy-angle-tabulated":
        energy = h5_group[f"energy"][()] * 1e6  # MeV to eV
        offset = h5_group[f"offset"][()]

        energy_out = h5_group[f"energy_out"][()] * 1e6  # MeV to eV
        pdf = h5_group[f"pdf"][()] / 1e6  # /MeV to /eV
        cosine_offset = h5_group[f"cosine_offset"][()]

        cosine = h5_group[f"cosine"][()]
        cosine_pdf = h5_group[f"cosine_pdf"][()]

        energy_spectrum = DistributionTabulatedEnergyAngle(
            energy, offset, energy_out, pdf, cosine_offset, cosine, cosine_pdf
        )

    elif spectrum_type == "N-body":
        value = h5_group["value"][()] * 1e6  # MeV to eV
        pdf = h5_group["pdf"][()] / 1e6  # /MeV to /eV

        energy_spectrum = DistributionNBody(value, pdf)

    else:
        print_error(f"Unsupported energy spectrum of type {spectrum_type}")

    return energy_spectrum
