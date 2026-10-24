from typing import Annotated
from numpy import float64
from numpy import int64
from numpy.typing import NDArray
import numpy as np

####

from mcdc.constant import (
    NEUTRON_REACTION_CAPTURE,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    NEUTRON_REACTION_INELASTIC_SCATTERING,
    REFERENCE_FRAME_COM,
    INTERPOLATION_LINEAR,
)
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataBase, DataPolynomial
from mcdc.object_.hdf5_reader import (
    read_data,
    read_reaction_properties,
    read_distribution,
    set_xs_interpolation,
    numbered_groups,
)
from mcdc.object_.distribution import (
    DistributionBase,
    DistributionNone,
    validate_angle_energy,
)
from mcdc.object_.util import decode_reference_frame
from mcdc.print_ import print_1d_array

# ======================================================================================
# Neutron reaction base class
# ======================================================================================


class NeutronReactionBase(MCDCPolymorphic):
    """Base neutron reaction data loaded from the HDF5 physics library.

    Stores the ENDF reaction identifier, cross-section segment and offset,
    reaction reference frame, and Q value.
    """

    # MC/DC framework metadata
    label = "neutron_reaction"
    sub_type = -1  # Polymorphic base

    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" is reserved for the packed xs array.
    xs_interpolations: NDArray[int64]
    xs_interpolation_boundaries: NDArray[int64]
    reference_frame: int
    q_value: float64

    secondary_products: list[SecondaryProduct]

    def __init__(self, MT, xs, xs_offset, reference_frame, q_value):
        super().__init__()
        self.secondary_products = []
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.xs_interpolations = np.array([INTERPOLATION_LINEAR], dtype=int64)
        self.xs_interpolation_boundaries = np.array([len(xs)], dtype=int64)
        self.reference_frame = reference_frame
        self.q_value = q_value

    def __repr__(self):
        text = super().__repr__()

        text += f"  - ID: {self.ID}\n"
        text += f"  - MT: {self.MT}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
        text += f"  - Reference frame: {decode_reference_frame(self.reference_frame)}\n"
        text += f"  - Q-value: {self.q_value}\n"
        return text


# ======================================================================================
# Neutron elastic scattering
# ======================================================================================


class NeutronReactionElasticScattering(NeutronReactionBase):
    """Elastic neutron scattering with incident-energy-dependent cosine data."""

    # MC/DC framework metadata
    label = "neutron_elastic_scattering_reaction"
    sub_type = NEUTRON_REACTION_ELASTIC_SCATTERING

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
        """Build an elastic-scattering reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame, _ = set_basic_properties(h5_group)
        if reference_frame != REFERENCE_FRAME_COM:
            raise ValueError(
                f"{h5_group.name}: elastic scattering requires center-of-mass angular data"
            )
        mu = set_angular_distribution(h5_group["angular-cosine-distribution"])
        return set_xs_interpolation(
            cls(MT, xs, xs_offset, reference_frame, mu), h5_group
        )

    def __repr__(self):
        text = super().__repr__()
        text += (
            f"  - Scattering cosine: {self.mu_table.label} [ID: {self.mu_table.ID}]\n"
        )
        return text


# ======================================================================================
# Neutron capture
# ======================================================================================


class NeutronReactionCapture(NeutronReactionBase):
    """Neutron capture reaction."""

    # MC/DC framework metadata
    label = "neutron_capture_reaction"
    sub_type = NEUTRON_REACTION_CAPTURE

    def __init__(self, MT, xs, xs_offset, reference_frame, q_value):
        super().__init__(MT, xs, xs_offset, reference_frame, q_value)

    @classmethod
    def from_h5_group(cls, h5_group, simulation):
        """Build a capture reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame, q_value = set_basic_properties(h5_group)
        return set_xs_interpolation(
            cls(MT, xs, xs_offset, reference_frame, q_value), h5_group
        )


# ======================================================================================
# Neutron inelastic scattering
# ======================================================================================


class NeutronReactionInelasticScattering(NeutronReactionBase):
    """Inelastic scattering with angular data and one or more energy spectra."""

    # MC/DC framework metadata
    label = "neutron_inelastic_scattering_reaction"
    sub_type = NEUTRON_REACTION_INELASTIC_SCATTERING

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
        """Build an inelastic-scattering reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame, q_value = set_basic_properties(h5_group)
        multiplicity = int(h5_group["multiplicity"][()])

        mu = set_angular_distribution(h5_group.get("angular-cosine-distribution"))

        # Energy spectra
        spectrum_probability_grid = h5_group["spectrum-probability-grid"][()]
        spectrum_probability = h5_group["spectrum-probability"][()]
        energy_spectra = [
            read_distribution(child)
            for child in numbered_groups(h5_group, "energy-spectrum-")
        ]
        validate_angle_energy(mu, energy_spectra, h5_group.name, reference_frame)
        if (
            not energy_spectra
            or np.any(spectrum_probability < 0)
            or not np.allclose(np.sum(spectrum_probability, axis=1), 1.0)
        ):
            raise ValueError(
                f"{h5_group.name}: spectrum probabilities must sum to one in every bin"
            )

        return set_xs_interpolation(
            cls(
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
            ),
            h5_group,
        )

    def __repr__(self):
        text = super().__repr__()
        if isinstance(self.mu, DistributionNone):
            text += f"  - Scattering cosine: Energy-correlated\n"
        else:
            text += f"  - Scattering cosine: {self.mu.label} [ID: {self.mu.ID}]\n"
        text += f"  - Energy spectra\n"
        text += f"      - Probability energy grid {print_1d_array(self.spectrum_probability_grid)}\n"
        for i in range(len(self.energy_spectra)):
            text += f"      - Spectrum {i}: {self.energy_spectra[i].label} [{print_1d_array(self.spectrum_probability[:,i])}] [ID: {self.energy_spectra[i].ID}]\n"
        return text


# ======================================================================================
# Neutron fission
# ======================================================================================


class NeutronReactionFission(NeutronReactionBase):
    """Fission channel owning prompt yields, spectra, and delayed precursor groups."""

    # MC/DC framework metadata
    label = "neutron_fission_reaction"
    sub_type = NEUTRON_REACTION_FISSION

    mu: DistributionBase
    spectrum: DistributionBase
    prompt_multiplicity: DataBase
    delayed_multiplicity: DataBase
    N_delayed_precursor: int
    delayed_fractions: NDArray[float64]
    delayed_decay_rates: NDArray[float64]
    delayed_spectra: list[DistributionBase]

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        q_value,
        mu,
        spectrum,
    ):
        super().__init__(MT, xs, xs_offset, reference_frame, q_value)
        self.mu = mu
        self.spectrum = spectrum
        validate_angle_energy(mu, [spectrum], self.label, reference_frame)
        self.prompt_multiplicity = DataPolynomial(np.array([0.0]))
        self.delayed_multiplicity = DataPolynomial(np.array([0.0]))
        self.N_delayed_precursor = 0
        self.delayed_fractions = np.zeros(0)
        self.delayed_decay_rates = np.zeros(0)
        self.delayed_spectra = []

    @classmethod
    def from_h5_group(cls, h5_group, simulation):
        """Build a fission reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame, q_value = set_basic_properties(h5_group)

        # Prompt angular distribution
        mu = set_angular_distribution(
            h5_group.get("prompt-angular-cosine-distribution")
        )

        # Prompt spectrum
        spectrum = read_distribution(h5_group["prompt-energy-distribution"])
        validate_angle_energy(mu, [spectrum], h5_group.name, reference_frame)

        result = cls(MT, xs, xs_offset, reference_frame, q_value, mu, spectrum)
        result.prompt_multiplicity = read_data(h5_group["prompt-multiplicity"])
        result.delayed_multiplicity = read_data(h5_group["delayed-multiplicity"])
        delayed = h5_group["delayed-neutrons"]
        result.delayed_fractions = delayed["fractions"][()]
        result.delayed_decay_rates = delayed["decay-rates"][()]
        result.delayed_spectra = [
            read_distribution(child)
            for child in numbered_groups(delayed, "energy-distribution-")
        ]
        result.N_delayed_precursor = len(result.delayed_fractions)
        if (
            len(result.delayed_decay_rates) != result.N_delayed_precursor
            or len(result.delayed_spectra) != result.N_delayed_precursor
        ):
            raise ValueError(
                f"{delayed.name}: delayed fractions, decay rates, and spectra must have equal lengths"
            )
        if result.N_delayed_precursor and (
            np.any(result.delayed_fractions < 0)
            or not np.isclose(sum(result.delayed_fractions), 1.0)
            or np.any(result.delayed_decay_rates <= 0)
        ):
            raise ValueError(
                f"{delayed.name}: invalid delayed-neutron probabilities or decay rates"
            )
        return set_xs_interpolation(result, h5_group)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Prompt neutron\n"
        if isinstance(self.mu, DistributionNone):
            text += f"  - Emission cosine: Energy-correlated\n"
        else:
            text += f"  - Emission cosine: {self.mu.label} [ID: {self.mu.ID}]\n"
        text += (
            f"      - Energy spectrum: {self.spectrum.label} [ID: {self.spectrum.ID}]\n"
        )

        return text


# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    """Read shared reaction fields and the neutron reaction's eV Q value."""
    return (*read_reaction_properties(h5_group), float(h5_group["Q-value"][()]))


def set_angular_distribution(h5_group):
    """Read an explicit angle law, or identify an angle supplied by a joint law."""
    if h5_group is None:
        return DistributionNone()
    return read_distribution(h5_group)
