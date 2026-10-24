from typing import Annotated
import numpy as np
from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import (
    INF,
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOG,
    PROTON_REACTION_ELASTIC_SCATTERING,
    PROTON_REACTION_CAPTURE,
    PROTON_REACTION_INELASTIC_SCATTERING,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
    PARTICLE_NEUTRON,
    PARTICLE_PROTON,
    PARTICLE_ELECTRON,
    PARTICLE_ANY,
)
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.object_.data import DataPolynomial, DataTable
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.distribution import (
    DistributionBase,
    DistributionNone,
    DistributionIsotropic,
    DistributionMultiTable,
    DistributionLevelScattering,
    DistributionEvaporation,
    DistributionMaxwellian,
    DistributionKalbachMann,
    DistributionTableEnergyAngle,
    DistributionNBody,
    validate_angle_energy,
)
from mcdc.object_.util import decode_reference_frame
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Proton reaction base class
# ======================================================================================


class ProtonReactionBase(MCDCPolymorphic):
    # Annotations for Numba mode
    label: str = "proton_reaction"
    sub_type = -1  # Polymorphic base
    #
    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" ir reserved for "xs"
    reference_frame: int
    q_value: float64

    secondary_products: list[SecondaryProduct]

    def __init__(self, MT, xs, xs_offset, reference_frame, q_value):
        super().__init__()
        self.secondary_products = []
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.reference_frame = reference_frame
        self.q_value = q_value

    def __repr__(self):
        text = super().__repr__()
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


# ======================================================================================
# Proton elastic scattering
# ======================================================================================


class ProtonReactionElasticScattering(ProtonReactionBase):
    # Annotations for Numba mode
    label: str = "proton_elastic_scattering_reaction"
    sub_type = PROTON_REACTION_ELASTIC_SCATTERING
    #
    mu_table: DistributionBase

    def __init__(self, MT, xs, xs_offset, reference_frame, mu):
        super().__init__(MT, xs, xs_offset, reference_frame, 0.0)
        if isinstance(mu, DistributionNone):
            raise ValueError(
                "Elastic scattering requires an explicit angular distribution"
            )
        self.mu_table = mu

    @classmethod
    def from_h5_group(cls, h5_group, simulation):
        MT, xs, xs_offset, reference_frame, _ = set_basic_properties(h5_group)
        mu = set_angular_distribution(h5_group["angular_cosine_distribution"])
        return cls(MT, xs, xs_offset, reference_frame, mu)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - mu: {self.mu_table}\n"
        text += f"  - sub_type: {self.sub_type}\n"
        text += f"  - sub_ID: {self.sub_ID}\n"
        # text += f"  - Scattering cosine: {distribution.decode_type(self.mu_table.type)} [ID: {self.mu_table.ID}]\n"
        return text


# ======================================================================================
# Proton inelastic scattering
# ======================================================================================


class ProtonReactionInelasticScattering(ProtonReactionBase):
    # Annotations for Numba mode
    label: str = "proton_inelastic_scattering_reaction"
    sub_type = PROTON_REACTION_INELASTIC_SCATTERING
    #
    multiplicity: int
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
        mu,
        spectrum_probability_grid,
        spectrum_probability,
        energy_spectra,
    ):
        super().__init__(MT, xs, xs_offset, reference_frame, q_value)

        self.multiplicity = multiplicity
        self.mu = mu
        validate_angle_energy(mu, energy_spectra, self.label, reference_frame)
        self.N_spectrum_probability_bin = len(spectrum_probability_grid) - 1
        self.N_spectrum = len(energy_spectra)
        self.spectrum_probability_grid = spectrum_probability_grid
        self.spectrum_probability = spectrum_probability
        self.energy_spectra = energy_spectra

    @classmethod
    def from_h5_group(cls, h5_group, simulation):
        MT, xs, xs_offset, reference_frame, q_value = set_basic_properties(h5_group)
        multiplicity = int(h5_group["multiplicity"][()])

        mu = set_angular_distribution(h5_group["angular_cosine_distribution"])

        spectrum_probability_grid = h5_group["spectrum_probability_grid"][()] * 1e6
        spectrum_probability = h5_group["spectrum_probability"][()]
        energy_spectra = [
            set_energy_distribution(h5_group[name])
            for name in sorted(x for x in h5_group if x.startswith("energy_spectrum-"))
        ]
        reaction = cls(
            MT,
            xs,
            xs_offset,
            reference_frame,
            q_value,
            multiplicity,
            mu,
            spectrum_probability_grid,
            spectrum_probability,
            energy_spectra,
        )

        # Load other species without changing the outgoing-proton multiplicity.
        if "secondary_products" in h5_group:
            for name in sorted(h5_group["secondary_products"]):
                product_group = h5_group["secondary_products"][name]
                # Outgoing protons are handled by the reaction itself.
                if int(product_group.attrs["ZAP"]) == 1001:
                    continue
                reaction.secondary_products.append(
                    set_secondary_product(product_group, simulation)
                )
        return reaction

    def __repr__(self):
        text = super().__repr__()
        if isinstance(self.mu, DistributionNone):
            text += f"  - Scattering cosine: Energy-correlated\n"
        else:
            text += f"  - Scattering cosine: {self.mu.label} [ID: {self.mu.ID}]\n"
        text += f"  - Energy spectra\n"
        text += f"      - Probability energy grid {print_1d_array(self.spectrum_probability_grid)}\n"
        for i in range(len(self.energy_spectra)):
            text += f"      - Spectrum {i+1}: {self.energy_spectra[i].label} [{print_1d_array(self.spectrum_probability[:,i])}] [ID: {self.energy_spectra[i].ID}]\n"
        return text


class ProtonReactionCapture(ProtonReactionBase):
    # Annotations for Numba mode
    label: str = "proton_capture_reaction"
    sub_type = PROTON_REACTION_CAPTURE

    def __init__(self, MT, xs, xs_offset, reference_frame, q_value):
        super().__init__(MT, xs, xs_offset, reference_frame, q_value)

    @classmethod
    def from_h5_group(cls, h5_group, simulation):
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


def set_secondary_product(h5_group, simulation):
    """Load a product from the existing proton-library format."""
    # Read product identity, count, and emission frame.
    zap = int(h5_group.attrs["ZAP"])
    multiplicity = int(h5_group.attrs.get("multiplicity", 1))
    reference_frame = h5_group.attrs.get("reference_frame", "LAB")
    if isinstance(reference_frame, bytes):
        reference_frame = reference_frame.decode("utf-8")
    reference_frame = (
        REFERENCE_FRAME_LAB if reference_frame == "LAB" else REFERENCE_FRAME_COM
    )

    # Load the single outgoing-energy spectrum.
    if "kalbach_mann" in h5_group:
        energy_group = h5_group["kalbach_mann"]
    else:
        energy_group = h5_group["energy_spectrum"]
    energy = set_energy_distribution(energy_group)

    # Load separate angles or recover energy-angle correlation.
    if "angular_cosine_distribution" in h5_group:
        angular_group = h5_group["angular_cosine_distribution"]
        mu = set_angular_distribution(angular_group)
    elif isinstance(
        energy,
        (DistributionKalbachMann, DistributionTableEnergyAngle, DistributionNBody),
    ):
        mu = DistributionNone()
    else:
        mu = DistributionIsotropic()

    # Decode the library's ZAP identifier into explicit species fields.
    particle_type = {1: PARTICLE_NEUTRON, 1001: PARTICLE_PROTON}.get(zap, PARTICLE_ANY)
    atomic_number, mass_number = divmod(zap, 1000)

    # Use a constant yield and select the sole spectrum at every incident energy.
    # Existing library products use the shared object's prompt-emission defaults.
    return SecondaryProduct(
        particle_type,
        DataPolynomial(np.array([float(multiplicity)])),
        reference_frame,
        mu,
        np.array([0.0, INF]),
        np.ones((1, 1)),
        [energy],
        atomic_number=atomic_number,
        mass_number=mass_number,
    )


def set_angular_distribution(h5_group):
    """Translate a proton-library angular law into its reusable representation.

    Explicit isotropy produces DistributionIsotropic; cosine PDFs produce a
    DistributionMultiTable indexed by incident energy in eV. The markers
    ``energy-correlated`` and ``given_in_energy_distribution`` produce
    DistributionNone because the energy-spectrum object supplies both energy
    and angle. In particular, the Kalbach-Mann angular parameters belong to
    the joint distribution loaded by :func:`set_energy_distribution`.
    """
    # Handle missing type attribute
    if "type" not in h5_group.attrs:
        mu_type = "isotropic"
    else:
        mu_type = h5_group.attrs["type"]

    if isinstance(mu_type, bytes):
        mu_type = mu_type.decode("utf-8")
    if mu_type == "isotropic":
        return DistributionIsotropic()
    elif mu_type in ("energy-correlated", "given_in_energy_distribution"):
        return DistributionNone()
    elif mu_type == "tabulated":
        tables = []

        # Collect numerical PDF arrays for each incident energy.
        if "energy" in h5_group:
            grid = h5_group[f"energy"][()] * 1e6  # MeV to eV
            offset = h5_group[f"offset"][()]
            value = h5_group[f"value"][()]
            pdf = h5_group[f"pdf"][()]
            for i, start in enumerate(offset):
                stop = offset[i + 1] if i + 1 < len(offset) else len(value)
                tables.append((value[start:stop], pdf[start:stop]))
        else:
            grid = h5_group["incident_energies"][()] * 1e6  # MeV to eV
            for i in range(len(grid)):
                subgroup = h5_group.get(f"E_in_{i + 1}")
                if (
                    subgroup is not None
                    and subgroup.attrs.get("type", "tabulated") == "tabulated"
                ):
                    table = (subgroup["cosines"][()], subgroup["pdf"][()])
                else:
                    # Isotropic or absent child tables use a uniform cosine density.
                    table = ([-1.0, 1.0], [0.5, 0.5])
                tables.append(table)

        mu = DistributionMultiTable(
            grid, value=[t[0] for t in tables], pdf=[t[1] for t in tables]
        )

        return mu
    raise ValueError(f"Unsupported angular distribution: {mu_type}")


def set_energy_distribution(h5_group):
    """Load an energy-only spectrum or a joint energy-angle distribution.

    Energy coordinates and parameters are converted from MeV to eV, and
    densities from inverse MeV to inverse eV. Dimensionless angular quantities
    retain their source values.

    Kalbach-Mann stores outgoing-energy PDFs with the precompound factor r
    and angular slope a in one DistributionKalbachMann object. Transport first
    samples outgoing energy, evaluates r and a for that energy, and then
    samples the conditional cosine analytically. The slope values are model
    parameters, not cosine coordinates or a separate angular PDF.
    """
    spectrum_type = h5_group.attrs["type"]

    if spectrum_type == "tabulated":
        grid = h5_group[f"energy"][()] * 1e6  # MeV to eV
        offset = h5_group[f"offset"][()]
        value = h5_group[f"value"][()] * 1e6  # MeV to eV
        pdf = h5_group[f"pdf"][()] / 1e6  # /MeV to /eV
        tables = []
        for i, start in enumerate(offset):
            stop = offset[i + 1] if i + 1 < len(offset) else len(value)
            tables.append((value[start:stop], pdf[start:stop]))
        energy_spectrum = DistributionMultiTable(
            grid, value=[t[0] for t in tables], pdf=[t[1] for t in tables]
        )

    elif spectrum_type == "level-scattering":
        C1 = h5_group["C1"][()] * 1e6  # MeV to eV
        C2 = h5_group["C2"][()]

        energy_spectrum = DistributionLevelScattering(C1, C2)

    elif spectrum_type == "evaporation":
        energy = h5_group[f"temperature_energy_grid"][()] * 1e6  # MeV to eV
        temperature = h5_group[f"temperature"][()] * 1e6  # MeV to eV
        restriction_energy = h5_group[f"restriction_energy"][()] * 1e6  # MeV to eV

        nuclear_temperature = DataTable(energy, temperature, INTERPOLATION_LINEAR)
        energy_spectrum = DistributionEvaporation(
            nuclear_temperature, restriction_energy
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

        nuclear_temperature = DataTable(energy, temperature, interpolation)
        energy_spectrum = DistributionMaxwellian(
            nuclear_temperature, restriction_energy
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

        energy_spectrum = DistributionTableEnergyAngle(
            energy, offset, energy_out, pdf, cosine_offset, cosine, cosine_pdf
        )

    elif spectrum_type == "N-body":
        value = h5_group["value"][()]
        pdf = h5_group["pdf"][()]

        energy_spectrum = DistributionNBody(
            h5_group["number_particles"][()],
            float(h5_group["total_mass_ratio"][()]),
            value,
            pdf,
        )

    else:
        print_error(f"Unsupported energy spectrum of type {spectrum_type}")

    return energy_spectrum
