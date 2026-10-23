import numpy as np

from collections.abc import Sequence
from numpy import float64, int64
from numpy.typing import ArrayLike, NDArray

from mcdc.constant import (
    DISTRIBUTION_NONE,
    DISTRIBUTION_PMF,
    DISTRIBUTION_TABULATED,
    DISTRIBUTION_MULTITABLE,
    DISTRIBUTION_LEVEL_SCATTERING,
    DISTRIBUTION_EVAPORATION,
    DISTRIBUTION_MAXWELLIAN,
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    DISTRIBUTION_N_BODY,
    DISTRIBUTION_ACE_LAW4,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
)
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataTable
from mcdc.object_.util import (
    cdf_from_pdf,
    pdf_from_cdf,
    multi_cdf_from_pdf,
    cmf_from_pmf,
)
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Distribution base class
# ======================================================================================


class DistributionBase(MCDCPolymorphic):
    """Base class for probability distributions sampled during transport."""

    # MC/DC framework metadata
    label = "distribution"
    sub_type = -1  # Polymorphic base


# ======================================================================================
# None
# ======================================================================================
# Placeholder for a distribution that does not need to store data:
#   - Isotropic distributions
#   - Energy-correlated angles stored in the energy distribution


class DistributionNone(DistributionBase):
    """Placeholder for an implicit or externally stored distribution."""

    # MC/DC framework metadata
    label = "none_distribution"
    sub_type = DISTRIBUTION_NONE


# ======================================================================================
# Probability mass function
# ======================================================================================


class DistributionPMF(DistributionBase):
    """Discrete probability mass function.

    Parameters
    ----------
    value : array_like
        Values that may be sampled.
    pmf : array_like
        Nonnegative relative masses, normalized internally.
    """

    # MC/DC framework metadata
    label = "pmf_distribution"
    sub_type = DISTRIBUTION_PMF

    value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]

    def __init__(self, value: ArrayLike, pmf: ArrayLike) -> None:
        super().__init__()

        self.value = np.asarray(value, dtype=float64)
        pmf_array = np.asarray(pmf, dtype=float64)

        if self.value.ndim != 1:
            print_error("value must be one-dimensional.")
        if pmf_array.ndim != 1:
            print_error("pmf must be one-dimensional.")
        if len(self.value) == 0:
            print_error("value and pmf must contain at least one entry.")
        if len(self.value) != len(pmf_array):
            print_error("value and pmf must have the same length.")

        self.pmf, self.cmf = cmf_from_pmf(pmf_array)

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pmf {print_1d_array(self.pmf)}\n"
        return text


# ======================================================================================
# Tabulated
# ======================================================================================


class DistributionTabulated(DistributionBase):
    """One-dimensional continuous tabulated distribution.

    Exactly one of ``pdf`` or ``cdf`` must be supplied alongside ``value``.
    PDF input is treated as piecewise linear; CDF input produces a histogram
    density. The distribution is normalized internally.
    """

    # MC/DC framework metadata
    label = "tabulated_distribution"
    sub_type = DISTRIBUTION_TABULATED

    pdf: DataTable

    def __init__(
        self,
        value: ArrayLike,
        pdf: ArrayLike | None = None,
        cdf: ArrayLike | None = None,
    ) -> None:
        super().__init__()

        if (pdf is None) == (cdf is None):
            print_error("Exactly one of pdf or cdf must be provided.")

        value_array = np.asarray(value, dtype=float64)

        if value_array.ndim != 1:
            print_error("value must be one-dimensional.")
        if len(value_array) == 0:
            print_error("value must contain at least one entry.")

        if pdf is not None:
            pdf_array = np.asarray(pdf, dtype=float64)

            if pdf_array.ndim != 1:
                print_error("pdf must be one-dimensional.")
            if len(pdf_array) != len(value_array):
                print_error("value and pdf must have the same length.")

            interpolation = INTERPOLATION_LINEAR
            pdf_normalized, cdf_normalized = cdf_from_pdf(
                value_array,
                pdf_array,
            )
        else:
            cdf_array = np.asarray(cdf, dtype=float64)

            if cdf_array.ndim != 1:
                print_error("cdf must be one-dimensional.")
            if len(cdf_array) != len(value_array):
                print_error("value and cdf must have the same length.")

            interpolation = INTERPOLATION_HISTOGRAM
            pdf_normalized, cdf_normalized = pdf_from_cdf(
                value_array,
                cdf_array,
            )

        self.pdf = DataTable(
            value_array,
            pdf_normalized,
            interpolation,
            aux=cdf_normalized,
        )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - probability density {print_1d_array(self.pdf.y)}\n"
        text += "  - cumulative distribution " f"{print_1d_array(self.pdf.aux[0])}\n"
        return text


# ======================================================================================
# Multi-table
# ======================================================================================


class DistributionMultiTable(DistributionBase):
    """Family of tabulated distributions indexed by another grid.

    ``offset`` marks the first entry of each table in the flattened ``value``
    and probability arrays. Exactly one of ``pdf`` or ``cdf`` is required.
    """

    # MC/DC framework metadata
    label = "multi_table_distribution"
    sub_type = DISTRIBUTION_MULTITABLE

    grid: NDArray[float64]
    tables: list[DistributionTabulated]

    def __init__(
        self,
        grid: ArrayLike,
        offset: ArrayLike,
        value: ArrayLike,
        pdf: ArrayLike | None = None,
        cdf: ArrayLike | None = None,
    ) -> None:
        super().__init__()

        if (pdf is None) == (cdf is None):
            print_error("Exactly one of pdf or cdf must be provided.")

        self.grid = np.asarray(grid, dtype=float64)
        offset_array = np.asarray(offset, dtype=int64)
        value_array = np.asarray(value, dtype=float64)

        if self.grid.ndim != 1:
            print_error("grid must be one-dimensional.")
        if offset_array.ndim != 1:
            print_error("offset must be one-dimensional.")
        if value_array.ndim != 1:
            print_error("value must be one-dimensional.")

        if len(self.grid) == 0:
            print_error("grid must contain at least one value.")
        if len(self.grid) != len(offset_array):
            print_error("grid and offset must have the same length.")
        if len(value_array) == 0:
            print_error("value must contain at least one value.")
        if offset_array[0] != 0:
            print_error("offset[0] must be zero.")
        if np.any(offset_array[1:] <= offset_array[:-1]):
            print_error("offset must be strictly increasing.")
        if offset_array[-1] >= len(value_array):
            print_error("Every offset must refer to an element in value.")

        pdf_array = None
        cdf_array = None

        if pdf is not None:
            pdf_array = np.asarray(pdf, dtype=float64)

            if pdf_array.ndim != 1:
                print_error("pdf must be one-dimensional.")
            if len(pdf_array) != len(value_array):
                print_error("pdf and value must have the same length.")

        if cdf is not None:
            cdf_array = np.asarray(cdf, dtype=float64)

            if cdf_array.ndim != 1:
                print_error("cdf must be one-dimensional.")
            if len(cdf_array) != len(value_array):
                print_error("cdf and value must have the same length.")

        stop = np.empty(len(offset_array), dtype=int64)
        stop[:-1] = offset_array[1:]
        stop[-1] = len(value_array)

        self.tables = []

        for start_i, stop_i in zip(offset_array, stop):
            if stop_i <= start_i:
                print_error("Each table must contain at least one value.")

            if pdf_array is not None:
                table = DistributionTabulated(
                    value_array[start_i:stop_i],
                    pdf=pdf_array[start_i:stop_i],
                )
            elif cdf_array is not None:
                table = DistributionTabulated(
                    value_array[start_i:stop_i],
                    cdf=cdf_array[start_i:stop_i],
                )
            else:
                table = DistributionTabulated([])  # Unachievable

            self.tables.append(table)

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - grid: {print_1d_array(self.grid)}\n"
        text += f"  - tables: {len(self.tables)}\n"

        for i, table in enumerate(self.tables):
            text += f"    - table[{i}] " f"(grid={self.grid[i]:.6g}, N={table.pdf.N})\n"

        return text


class DistributionACELaw4(DistributionBase):
    """ACE LAW 4 outgoing-energy tables with discrete lines and continuum."""

    label = "ace_law4_distribution"
    sub_type = DISTRIBUTION_ACE_LAW4

    energy: NDArray[float64]
    offset: NDArray[int64]
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    n_discrete: NDArray[int64]
    interpolation: NDArray[int64]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        value: ArrayLike,
        pdf: ArrayLike,
        cdf: ArrayLike,
        n_discrete: ArrayLike,
        interpolation: ArrayLike,
    ) -> None:
        super().__init__()

        self.energy = np.asarray(energy, dtype=float64)
        self.offset = np.asarray(offset, dtype=int64)
        self.value = np.asarray(value, dtype=float64)
        self.pdf = np.asarray(pdf, dtype=float64)
        self.cdf = np.asarray(cdf, dtype=float64)
        self.n_discrete = np.asarray(n_discrete, dtype=int64)
        self.interpolation = np.asarray(interpolation, dtype=int64)

        n_tables = len(self.energy)
        if self.energy.ndim != 1 or n_tables == 0:
            print_error("ACE LAW 4 incident-energy grid must be nonempty and 1D.")
        if self.offset.ndim != 1 or len(self.offset) != n_tables:
            print_error("ACE LAW 4 offsets must match the incident-energy grid.")
        if self.n_discrete.ndim != 1 or len(self.n_discrete) != n_tables:
            print_error("ACE LAW 4 discrete-line counts must match the incident grid.")
        if self.interpolation.ndim != 1 or len(self.interpolation) != n_tables:
            print_error("ACE LAW 4 interpolation flags must match the incident grid.")
        if self.value.ndim != 1 or self.pdf.ndim != 1 or self.cdf.ndim != 1:
            print_error("ACE LAW 4 values, PDFs, and CDFs must be one-dimensional.")
        if len(self.value) == 0 or len(self.pdf) != len(self.value):
            print_error("ACE LAW 4 values and PDFs must have the same nonzero length.")
        if len(self.cdf) != len(self.value):
            print_error("ACE LAW 4 values and CDFs must have the same length.")
        if self.offset[0] != 0 or np.any(self.offset[1:] <= self.offset[:-1]):
            print_error("ACE LAW 4 offsets must start at zero and increase.")
        if np.any(self.offset < 0) or self.offset[-1] >= len(self.value):
            print_error("ACE LAW 4 offsets must refer to an existing table.")
        if np.any(np.diff(self.energy) <= 0.0):
            print_error("ACE LAW 4 incident energies must be strictly increasing.")
        if np.any((self.interpolation != 1) & (self.interpolation != 2)):
            print_error("ACE LAW 4 outgoing interpolation must be histogram or linear.")

        ends = np.append(self.offset[1:], len(self.value))
        for start, end, discrete in zip(self.offset, ends, self.n_discrete):
            if end <= start or discrete < 0 or discrete > end - start:
                print_error("ACE LAW 4 table has an invalid discrete-line count.")
            normalization = self.cdf[end - 1]
            if normalization <= 0.0 or np.any(np.diff(self.cdf[start:end]) < 0.0):
                print_error("ACE LAW 4 table must have a positive, nondecreasing CDF.")
            self.cdf[start:end] /= normalization
            self.pdf[start:end] /= normalization

            continuum_start = start + discrete
            if end - continuum_start > 1 and np.any(
                np.diff(self.value[continuum_start:end]) <= 0.0
            ):
                print_error("ACE LAW 4 continuum energies must be strictly increasing.")


# ======================================================================================
# Level scattering
# ======================================================================================


class DistributionLevelScattering(DistributionBase):
    """Discrete-level inelastic-scattering energy distribution.

    Parameters
    ----------
    C1, C2 : float
        Level-scattering law coefficients.
    """

    # MC/DC framework metadata
    label = "level_scattering_distribution"
    sub_type = DISTRIBUTION_LEVEL_SCATTERING

    C1: float
    C2: float

    def __init__(self, C1: float, C2: float) -> None:
        super().__init__()

        self.C1 = C1
        self.C2 = C2

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - C1: {self.C1} [/eV^l]\n"
        text += f"  - C2: {self.C2}\n"
        return text


# ======================================================================================
# Evaporation
# ======================================================================================


class DistributionEvaporation(DistributionBase):
    """Evaporation spectrum with incident-energy-dependent temperature."""

    # MC/DC framework metadata
    label = "evaporation_distribution"
    sub_type = DISTRIBUTION_EVAPORATION

    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid: NDArray[float64],
        nuclear_temperature_value: NDArray[float64],
        restriction_energy: float,
        temperature_interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None,
    ) -> None:
        super().__init__()

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolations,
            interpolation_boundaries,
        )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += (
            "  - Nuclear temperature "
            f"{print_1d_array(self.nuclear_temperature.y)} [eV]\n"
        )
        text += (
            "  - Nuclear temperature energy grid "
            f"{print_1d_array(self.nuclear_temperature.x)} [eV]\n"
        )
        return text


# ======================================================================================
# Maxwellian
# ======================================================================================


class DistributionMaxwellian(DistributionBase):
    """Maxwellian spectrum with incident-energy-dependent temperature."""

    # MC/DC framework metadata
    label = "maxwellian_distribution"
    sub_type = DISTRIBUTION_MAXWELLIAN

    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid: NDArray[float64],
        nuclear_temperature_value: NDArray[float64],
        restriction_energy: float,
        temperature_interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None,
    ) -> None:
        super().__init__()

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolations,
            interpolation_boundaries,
        )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += (
            "  - Nuclear temperature "
            f"{print_1d_array(self.nuclear_temperature.y)} [eV]\n"
        )
        text += (
            "  - Nuclear temperature energy grid "
            f"{print_1d_array(self.nuclear_temperature.x)} [eV]\n"
        )
        return text


# ======================================================================================
# Kalbach-Mann
# ======================================================================================


class DistributionKalbachMann(DistributionBase):
    """Correlated Kalbach-Mann outgoing energy-angle distribution.

    Incident-energy tables are stored in flattened arrays delimited by
    ``offset``. Probability densities are normalized per table.
    """

    # MC/DC framework metadata
    label = "kalbach_mann_distribution"
    sub_type = DISTRIBUTION_KALBACH_MANN

    energy: NDArray[float64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    precompound_factor: NDArray[float64]
    angular_slope: NDArray[float64]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        energy_out: ArrayLike,
        pdf: ArrayLike,
        precompound_factor: ArrayLike,
        angular_slope: ArrayLike,
    ) -> None:
        super().__init__()

        self.energy = np.asarray(energy, dtype=float64)
        self.offset = np.asarray(offset, dtype=int64)
        self.energy_out = np.asarray(energy_out, dtype=float64)
        pdf_array = np.asarray(pdf, dtype=float64)
        self.precompound_factor = np.asarray(
            precompound_factor,
            dtype=float64,
        )
        self.angular_slope = np.asarray(angular_slope, dtype=float64)

        self.pdf, self.cdf = multi_cdf_from_pdf(
            self.offset,
            self.energy_out,
            pdf_array,
        )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - grid {print_1d_array(self.energy)} [eV]\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - energy {print_1d_array(self.energy_out)} [eV]\n"
        text += f"  - energy-pdf {print_1d_array(self.pdf)} [/eV]\n"
        text += "  - precompound factor " f"{print_1d_array(self.precompound_factor)}\n"
        text += f"  - angular slope {print_1d_array(self.angular_slope)}\n"
        return text


# ======================================================================================
# Tabulated energy-angle
# ======================================================================================


class DistributionTabulatedEnergyAngle(DistributionBase):
    """Correlated tabulated outgoing energy and scattering-angle distribution.

    ``offset`` delimits outgoing-energy tables and ``cosine_offset`` delimits
    conditional cosine tables in the flattened arrays.
    """

    # MC/DC framework metadata
    label = "tabulated_energy_angle_distribution"
    sub_type = DISTRIBUTION_TABULATED_ENERGY_ANGLE

    energy: NDArray[float64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    cosine_offset_: NDArray[int64]
    cosine: NDArray[float64]
    cosine_pdf: NDArray[float64]
    cosine_cdf: NDArray[float64]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        energy_out: ArrayLike,
        pdf: ArrayLike,
        cosine_offset: ArrayLike,
        cosine: ArrayLike,
        cosine_pdf: ArrayLike,
    ) -> None:
        super().__init__()

        self.energy = np.asarray(energy, dtype=float64)
        self.offset = np.asarray(offset, dtype=int64)
        self.energy_out = np.asarray(energy_out, dtype=float64)
        pdf_array = np.asarray(pdf, dtype=float64)
        self.cosine_offset_ = np.asarray(cosine_offset, dtype=int64)
        self.cosine = np.asarray(cosine, dtype=float64)
        cosine_pdf_array = np.asarray(cosine_pdf, dtype=float64)

        self.pdf, self.cdf = multi_cdf_from_pdf(
            self.offset,
            self.energy_out,
            pdf_array,
        )

        self.cosine_pdf = cosine_pdf_array.copy()
        self.cosine_cdf = np.zeros_like(self.cosine_pdf)

        for i in range(len(self.offset)):
            energy_start = self.offset[i]

            if i + 1 < len(self.offset):
                energy_stop = self.offset[i + 1]
            else:
                energy_stop = len(self.energy_out)

            inner_offset = self.cosine_offset_[energy_start:energy_stop]

            if len(inner_offset) == 0:
                print_error(
                    "Each incident-energy table must reference at least one "
                    "cosine distribution."
                )

            cosine_start = inner_offset[0]

            if i + 1 < len(self.offset):
                cosine_stop = self.cosine_offset_[energy_stop]
            else:
                cosine_stop = len(self.cosine)

            inner_offset_local = inner_offset - cosine_start

            (
                self.cosine_pdf[cosine_start:cosine_stop],
                self.cosine_cdf[cosine_start:cosine_stop],
            ) = multi_cdf_from_pdf(
                inner_offset_local,
                self.cosine[cosine_start:cosine_stop],
                cosine_pdf_array[cosine_start:cosine_stop],
            )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - grid {print_1d_array(self.energy)} [eV]\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - energy {print_1d_array(self.energy_out)} [eV]\n"
        text += f"  - energy-pdf {print_1d_array(self.pdf)} [/eV]\n"
        text += "  - cosine-offset " f"{print_1d_array(self.cosine_offset_)}\n"
        text += f"  - cosine {print_1d_array(self.cosine)}\n"
        text += f"  - cosine-pdf {print_1d_array(self.cosine_pdf)}\n"
        return text


# ======================================================================================
# N-body
# ======================================================================================


class DistributionNBody(DistributionBase):
    """N-body phase-space outgoing-energy distribution.

    Parameters
    ----------
    values, probabilities : array_like
        Outgoing values and their piecewise-linear relative density.
    """

    # MC/DC framework metadata
    label = "nbody_distribution"
    sub_type = DISTRIBUTION_N_BODY

    pdf: DataTable

    def __init__(
        self,
        values: ArrayLike,
        probabilities: ArrayLike,
    ) -> None:
        super().__init__()

        value_array = np.asarray(values, dtype=float64)
        probability_array = np.asarray(probabilities, dtype=float64)

        if value_array.ndim != 1:
            print_error("values must be one-dimensional.")
        if probability_array.ndim != 1:
            print_error("probabilities must be one-dimensional.")
        if len(value_array) != len(probability_array):
            print_error("values and probabilities must have the same length.")

        pdf_normalized, cdf_normalized = cdf_from_pdf(
            value_array,
            probability_array,
        )

        self.pdf = DataTable(
            value_array,
            pdf_normalized,
            INTERPOLATION_LINEAR,
            aux=cdf_normalized,
        )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - pdf {print_1d_array(self.pdf.y)}\n"
        return text
