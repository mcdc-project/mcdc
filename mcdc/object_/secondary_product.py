from typing import Annotated

from numpy import float64
from numpy.typing import NDArray

from mcdc.constant import INF
from mcdc.object_.base import MCDCObject
from mcdc.object_.data import DataBase
from mcdc.object_.distribution import DistributionBase, validate_angle_energy


class SecondaryProduct(MCDCObject):
    """Product of a different species from the incident particle.

    Production yield is the nonnegative mean number emitted per reaction,
    evaluated at the incident energy.
    Spectra are alternatives selected by their probabilities.
    DistributionNone in mu identifies spectra that supply energy and angle
    together; otherwise, mu supplies an explicit independent angular law.

    Nuclear identity is optional (-1 means unspecified).
    excitation_state identifies the nuclear state, with zero denoting the ground state.
    Delayed emission uses a decay constant in inverse seconds; prompt emission has
    delayed=False and the INF decay-constant sentinel.
    """

    # MC/DC framework metadata
    label = "secondary_product"

    # Product identity
    particle_type: int
    atomic_number: int
    mass_number: int
    excitation_state: int

    # Mean production per reaction, as a function of incident energy
    production_yield: DataBase

    # Emission timing
    delayed: bool
    decay_constant: float64

    # Emission distributions in the specified reference frame
    reference_frame: int
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
        particle_type,
        production_yield,
        reference_frame,
        mu,
        spectrum_probability_grid,
        spectrum_probability,
        energy_spectra,
        *,
        atomic_number=-1,
        mass_number=-1,
        excitation_state=-1,
        delayed=False,
        decay_constant=INF,
    ):
        super().__init__()

        # Product identity and mean production
        self.particle_type = particle_type
        self.atomic_number = atomic_number
        self.mass_number = mass_number
        self.excitation_state = excitation_state
        self.production_yield = production_yield

        # Prompt emission uses the INF decay-constant sentinel.
        self.delayed = delayed
        self.decay_constant = float64(decay_constant) if delayed else float64(INF)

        # Angular sampling and energy-spectrum selection
        self.reference_frame = reference_frame
        self.mu = mu
        validate_angle_energy(mu, energy_spectra, self.label, reference_frame)
        self.N_spectrum_probability_bin = len(spectrum_probability_grid) - 1
        self.N_spectrum = len(energy_spectra)
        self.spectrum_probability_grid = spectrum_probability_grid
        self.spectrum_probability = spectrum_probability
        self.energy_spectra = energy_spectra
