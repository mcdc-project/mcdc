from typing import Annotated
from numpy import float64
from numpy.typing import NDArray

####

from mcdc.object_.data import DataBase, DataTable
import mcdc.object_.distribution as distribution

from mcdc.constant import (
    ANGLE_ISOTROPIC,
    ANGLE_ENERGY_CORRELATED,
    ANGLE_DISTRIBUTED,
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOG,
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    REACTION_NEUTRON_INELASTIC_SCATTERING,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_EXCITATION,
    REACTION_ELECTRON_ELASTIC_SCATTERING,
    REACTION_ELECTRON_IONIZATION,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
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
# Reaction base class
# ======================================================================================


class ReactionBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "reaction"
    #
    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" ir reserved for "xs"
    reference_frame: int

    def __init__(self, type_, MT, xs, xs_offset, reference_frame):
        super().__init__(type_)
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.reference_frame = reference_frame

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - MT: {self.MT}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
        text += f"  - Reference frame: {decode_reference_frame(self.reference_frame)}\n"
        return text


def decode_type(type_):
    if type_ == REACTION_NEUTRON_ELASTIC_SCATTERING:
        return "Neutron elastic scattering"
    elif type_ == REACTION_NEUTRON_CAPTURE:
        return "Neutron capture"
    elif type_ == REACTION_NEUTRON_INELASTIC_SCATTERING:
        return "Neutron inelastic scattering"
    elif type_ == REACTION_NEUTRON_FISSION:
        return "Neutron fission"
    elif type_ == REACTION_ELECTRON_IONIZATION:
        return "Electron ionization"
    elif type_ == REACTION_ELECTRON_ELASTIC_SCATTERING:
        return "Electron elastic scattering"
    elif type_ == REACTION_ELECTRON_BREMSSTRAHLUNG:
        return "Electron bremsstrahlung"
    elif type_ == REACTION_ELECTRON_EXCITATION:
        return "Electron excitation"


def decode_reference_frame(type_):
    if type_ == REFERENCE_FRAME_LAB:
        return "Laboratory"
    elif type_ == REFERENCE_FRAME_COM:
        return "Center of mass"


# ======================================================================================
# Neutron elastic scattering
# ======================================================================================


class ReactionNeutronElasticScattering(ReactionBase):
    # Annotations for Numba mode
    label: str = "neutron_elastic_scattering_reaction"
    #
    mu_table: DistributionMultiTable

    def __init__(self, MT, xs, xs_offset, reference_frame, mu):
        type_ = REACTION_NEUTRON_ELASTIC_SCATTERING
        super().__init__(type_, MT, xs, xs_offset, reference_frame)
        self.mu_table = mu

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)
        _, mu = set_angular_distribution(h5_group["angular_cosine_distribution"])
        return cls(MT, xs, xs_offset, reference_frame, mu)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Scattering cosine: {distribution.decode_type(self.mu_table.type)} [ID: {self.mu_table.ID}]\n"
        return text


# ======================================================================================
# Neutron capture
# ======================================================================================


class ReactionNeutronCapture(ReactionBase):
    # Annotations for Numba mode
    label: str = "neutron_capture_reaction"

    def __init__(self, MT, xs, xs_offset, reference_frame):
        type_ = REACTION_NEUTRON_CAPTURE
        super().__init__(type_, MT, xs, xs_offset, reference_frame)

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)
        return cls(MT, xs, xs_offset, reference_frame)


# ======================================================================================
# Neutron inelastic scattering
# ======================================================================================


class ReactionNeutronInelasticScattering(ReactionBase):
    # Annotations for Numba mode
    label: str = "neutron_inelastic_scattering_reaction"
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
        multiplicity,
        angle_type,
        mu,
        spectrum_probability_grid,
        spectrum_probability,
        energy_spectra,
    ):
        type_ = REACTION_NEUTRON_INELASTIC_SCATTERING
        super().__init__(type_, MT, xs, xs_offset, reference_frame)

        self.reference_frame = reference_frame
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
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)
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


# ======================================================================================
# Neutron fission
# ======================================================================================


class ReactionNeutronFission(ReactionBase):
    # Annotations for Numba mode
    label: str = "neutron_fission_reaction"
    #
    angle_type: int
    mu: DistributionBase
    spectrum: DistributionBase

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        angle_type,
        mu,
        spectrum,
    ):
        type_ = REACTION_NEUTRON_FISSION
        super().__init__(type_, MT, xs, xs_offset, reference_frame)
        self.angle_type = angle_type
        self.mu = mu
        self.spectrum = spectrum

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        # Prompt angular distribution
        angle_type, mu = set_angular_distribution(
            h5_group["angular_cosine_distribution"]
        )

        # Prompt spectrum
        spectrum_names = [x for x in h5_group if x.startswith("energy_spectrum-")]
        if len(spectrum_names) > 1:
            print_error("Unsupported multi-distribution prompt fission spectrum")
        spectrum = set_energy_distribution(h5_group[f"energy_spectrum-1"])

        return cls(MT, xs, xs_offset, reference_frame, angle_type, mu, spectrum)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Prompt neutron\n"
        if self.angle_type == ANGLE_ISOTROPIC:
            text += f"  - Emission cosine: Isotropic\n"
        elif self.angle_type == ANGLE_ENERGY_CORRELATED:
            text += f"  - Emission cosine: Energy-correlated\n"
        else:
            text += f"  - Emission cosine: {distribution.decode_type(self.mu.type)} [ID: {self.mu.ID}]\n"
        text += f"      - Energy spectrum: {distribution.decode_type(self.spectrum)} [ID: {self.spectrum.ID}]\n"

        return text


# ======================================================================================
# Electron ionization
# ======================================================================================


class ReactionElectronIonization(ReactionBase):
    label: str = "electron_ionization_reaction"
    #
    N_subshell: int
    subshell_xs: list[DataBase]
    subshell_product: list[DistributionBase]

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        subshell_xs,
        subshell_product,
    ):
        type_ = REACTION_ELECTRON_IONIZATION
        super().__init__(type_, MT, xs, xs_offset, reference_frame)

        self.N_subshell = len(subshell_xs)
        self.subshell_xs = subshell_xs
        self.subshell_product = subshell_product

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        subshells = h5_group["subshells"]
        subshell_names = list(subshells.keys())

        subshell_xs = []
        subshell_product = []

        for name in subshell_names:
            subshell = subshells[name]

            # Subshell cross section table (each has its own energy grid)
            xs_sub = DataTable(
                subshell["energy_grid"][()],
                subshell["xs"][()]
            )
            subshell_xs.append(xs_sub)

            # Secondary electron energy distribution
            product = subshell["product"]
            prod_dist = DistributionMultiTable(
                product["energy_grid"][()],
                product["energy_offset"][()],
                product["value"][()],
                product["PDF"][()]
            )
            subshell_product.append(prod_dist)

        return cls(
            MT,
            xs,
            xs_offset,
            reference_frame,
            subshell_xs,
            subshell_product,
        )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Number of subshells: {self.N_subshell}\n"
        for i in range(self.N_subshell):
            text += f"    - Subshell {i+1}\n"
            text += f"      - XS: DataTable [ID: {self.subshell_xs[i].ID}]\n"
            prod = self.subshell_product[i]
            text += (
                f"      - Secondary electron spectrum: "
                f"{distribution.decode_type(prod.type)} [ID: {prod.ID}]\n"
            )
        return text


# ======================================================================================
# Electron elastic scattering
# ======================================================================================


class ReactionElectronElasticScattering(ReactionBase):
    label: str = "electron_elastic_scattering_reaction"
    #
    mu_cut: float
    xs_large: DataBase
    mu_table: DistributionMultiTable

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        mu_cut,
        xs_large,
        mu_table,
    ):
        type_ = REACTION_ELECTRON_ELASTIC_SCATTERING
        super().__init__(type_, MT, xs, xs_offset, reference_frame)
        self.mu_cut = mu_cut
        self.xs_large = xs_large
        self.mu_table = mu_table

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        # Large angle data
        large_angle = h5_group["large_angle"]
        mu_cut = float(large_angle.attrs.get("mu_cut", 0.999999))

        xs_large = DataTable(
            large_angle["xs_energy"][()],
            large_angle["xs"][()]
        )

        mu_g = large_angle["scattering_cosine"]
        mu_table = DistributionMultiTable(
            mu_g["energy_grid"][()],
            mu_g["energy_offset"][()],
            mu_g["value"][()],
            mu_g["PDF"][()]
        )

        return cls(
            MT,
            xs,
            xs_offset,
            reference_frame,
            mu_cut,
            xs_large,
            mu_table,
        )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Mu cut: {self.mu_cut}\n"
        text += f"  - Large angle XS: DataTable [ID: {self.xs_large.ID}]\n"
        text += (
            f"  - Scattering cosine: {distribution.decode_type(self.mu_table.type)} "
            f"[ID: {self.mu_table.ID}]\n"
        )
        return text


# ======================================================================================
# Electron bremsstrahlung
# ======================================================================================


class ReactionElectronBremsstrahlung(ReactionBase):
    label: str = "electron_bremsstrahlung_reaction"
    #
    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        type_ = REACTION_ELECTRON_BREMSSTRAHLUNG
        super().__init__(type_, MT, xs, xs_offset, reference_frame)
        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        base = h5_group["energy_loss"]
        eloss = DataTable(base["energy"][()], base["value"][()])

        return cls(MT, xs, xs_offset, reference_frame, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: DataTable [ID: {self.eloss.ID}]\n"
        return text


# ======================================================================================
# Electron excitation
# ======================================================================================


class ReactionElectronExcitation(ReactionBase):
    label: str = "electron_excitation_reaction"
    #
    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        type_ = REACTION_ELECTRON_EXCITATION
        super().__init__(type_, MT, xs, xs_offset, reference_frame)
        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        base = h5_group["energy_loss"]
        eloss = DataTable(base["energy"][()], base["value"][()])

        return cls(MT, xs, xs_offset, reference_frame, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: DataTable [ID: {self.eloss.ID}]\n"
        return text


# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    MT = h5_group.attrs["MT"][()]
    xs = h5_group["xs"][()]
    xs_offset = h5_group["xs"].attrs["offset"]
    reference_frame = h5_group["reference_frame"][()].decode("utf-8")
    if reference_frame == "LAB":
        reference_frame = REFERENCE_FRAME_LAB
    elif reference_frame == "COM":
        reference_frame = REFERENCE_FRAME_COM
    return MT, xs, xs_offset, reference_frame


def set_angular_distribution(h5_group):
    mu_type = h5_group.attrs["type"]
    if mu_type == "isotropic":
        angle_type = ANGLE_ISOTROPIC
        mu = simulation.distributions[0]
    elif mu_type == "energy-correlated":
        angle_type = ANGLE_ENERGY_CORRELATED
        mu = simulation.distributions[0]
    else:
        angle_type = ANGLE_DISTRIBUTED
        grid = h5_group[f"energy"][()] * 1e6  # MeV to eV
        offset = h5_group[f"offset"][()]
        value = h5_group[f"value"][()]
        pdf = h5_group[f"pdf"][()]
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