import numpy as np

from numbers import Integral
from numpy import float64, int64
from numpy.typing import ArrayLike, NDArray

from mcdc.constant import (
    DISTRIBUTION_NONE,
    DISTRIBUTION_ISOTROPIC,
    DISTRIBUTION_TABLE,
    DISTRIBUTION_MULTITABLE,
    DISTRIBUTION_LEVEL_SCATTERING,
    DISTRIBUTION_EVAPORATION,
    DISTRIBUTION_MAXWELLIAN,
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_TABLE_ENERGY_ANGLE,
    DISTRIBUTION_N_BODY,
    DISTRIBUTION_MIXTURE,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
    REFERENCE_FRAME_COM,
    PROBABILITY_TOLERANCE,
)
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import (
    DataBase,
    DataNone,
    DataPolynomial,
    DataTable,
    decode_interpolation,
)
from mcdc.object_.util import (
    cdf_from_pdf,
    pdf_from_cdf,
    multi_cdf_from_pdf,
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
# Placeholder for an absent distribution, including a separate angular law
# whose angle is supplied by a joint energy-angle distribution.


class DistributionNone(DistributionBase):
    """An absent distribution; this placeholder must never be sampled."""

    # MC/DC framework metadata
    label = "none_distribution"
    sub_type = DISTRIBUTION_NONE


# ======================================================================================
# Isotropic
# ======================================================================================


class DistributionIsotropic(DistributionBase):
    """Uniform angular cosine on [-1, 1], shared once per simulation.

    No probability arrays are required. Direction-handling code supplies the
    uniform azimuth and applies the reaction's reference-frame transformation.
    """

    label = "isotropic_distribution"
    sub_type = DISTRIBUTION_ISOTROPIC


# ======================================================================================
# Table
# ======================================================================================


class DistributionTable(DistributionBase):
    """One-dimensional continuous, discrete, or mixed distribution.

    ``value`` with ``pdf`` (or a continuous ``cdf``) describes a density;
    ``discrete_value`` with ``pmf`` describes point masses. Either component
    may be omitted. The PDF integral and PMF sum are normalized together,
    preserving their relative probabilities. Interpolation applies only to
    the continuous density. CDF input describes a piecewise-linear CDF.
    """

    label = "table_distribution"
    sub_type = DISTRIBUTION_TABLE

    pdf: DataBase
    discrete_value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]
    continuous_probability: float

    def __init__(
        self,
        value=None,
        pdf=None,
        cdf=None,
        *,
        discrete_value=None,
        pmf=None,
        interpolations=INTERPOLATION_LINEAR,
        interpolation_boundaries=None,
    ):
        super().__init__()
        (
            x,
            density,
            cumulative,
            self.discrete_value,
            self.pmf,
            self.continuous_probability,
            interpolations,
            interpolation_boundaries,
        ) = _probability_components(
            value,
            pdf,
            cdf,
            discrete_value,
            pmf,
            interpolations,
            interpolation_boundaries,
        )
        self.cmf = np.r_[0.0, np.cumsum(self.pmf)]
        self.pdf = DataNone()
        if self.continuous_probability > 0:
            self.pdf = DataTable(
                x, density, interpolations, interpolation_boundaries, aux=cumulative
            )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - continuous probability: {self.continuous_probability}\n"
        text += f"  - discrete values: {print_1d_array(self.discrete_value)}\n"
        text += f"  - probability masses: {print_1d_array(self.pmf)}\n"
        if self.continuous_probability > 0:
            text += f"  - continuous values: {print_1d_array(self.pdf.x)}\n"
            text += f"  - probability density: {print_1d_array(self.pdf.y)}\n"
        return text


# ======================================================================================
# Shared probability components
# ======================================================================================


def _probability_components(
    value, pdf, cdf, discrete_value, pmf, interpolations, interpolation_boundaries
):
    """Validate and jointly normalize a table's continuous and discrete parts.

    Return raw arrays rather than Distribution objects so standalone and packed
    tables share validation without constructing per-grid child objects.
    """
    continuous = value is not None
    discrete = discrete_value is not None
    if continuous != (pdf is not None or cdf is not None):
        raise ValueError("Continuous values require exactly one PDF or CDF")
    if pdf is not None and cdf is not None:
        raise ValueError("Supply either PDF or CDF, not both")
    if discrete != (pmf is not None) or not (continuous or discrete):
        raise ValueError(
            "Supply a continuous density, discrete values with PMF, or both"
        )
    discrete_values = np.asarray(
        [] if discrete_value is None else discrete_value, dtype=float64
    )
    masses = np.asarray([] if pmf is None else pmf, dtype=float64)
    if (
        discrete_values.ndim != 1
        or masses.ndim != 1
        or discrete_values.shape != masses.shape
        or (discrete and len(masses) == 0)
        or not np.all(np.isfinite(discrete_values))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
    ):
        raise ValueError(
            "Discrete values and nonnegative finite masses must be matching vectors"
        )
    x, density, cumulative = np.array([]), np.array([]), np.array([])
    continuous_mass = 0.0
    if continuous:
        x = np.asarray(value, dtype=float64)
        if cdf is not None:
            raw_cdf = np.asarray(cdf, dtype=float64)
            density, cumulative = pdf_from_cdf(x, raw_cdf)
            continuous_mass = float(raw_cdf[-1] - raw_cdf[0])
            density, cumulative = (
                density * continuous_mass,
                cumulative * continuous_mass,
            )
            interpolations, interpolation_boundaries = INTERPOLATION_HISTOGRAM, None
        else:
            density, cumulative = cdf_from_pdf(
                x, pdf, interpolations, interpolation_boundaries, normalize=False
            )
            continuous_mass = float(cumulative[-1])
    total = float(np.sum(masses)) + continuous_mass
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Combined probability must have positive finite mass")
    return (
        x,
        density / total,
        cumulative / total,
        discrete_values,
        masses / total,
        continuous_mass / total,
        interpolations,
        interpolation_boundaries,
    )


# ======================================================================================
# Multi-table
# ======================================================================================


class DistributionMultiTable(DistributionBase):
    """Conditional continuous, discrete, or mixed tables in packed arrays.

    Each grid point owns one jointly normalized PDF/PMF pair. Continuous arrays
    and discrete arrays have independent N + 1 offsets; absent components occupy
    empty slices. No child DistributionTable objects are constructed.
    Supply per-grid vectors (using None for absent components), or concatenate
    them and supply offsets/discrete_offsets. PDF or CDF input describes the
    continuous component; CDF increments retain that component's probability.
    Local PDF rules use exclusive boundaries and their own per-grid offsets.
    Corresponding discrete outcomes retain the same index across each linear
    grid interval. Their probabilities and coordinates interpolate separately.
    """

    label = "multi_table_distribution"
    sub_type = DISTRIBUTION_MULTITABLE

    grid: NDArray[float64]
    offsets: NDArray[int64]
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    discrete_offsets: NDArray[int64]
    discrete_value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]
    continuous_probability: NDArray[float64]
    interpolations: NDArray[int64]
    interpolation_boundaries: NDArray[int64]
    pdf_interpolations: NDArray[int64]
    pdf_interpolation_boundaries: NDArray[int64]
    pdf_interpolation_offsets: NDArray[int64]

    def __init__(
        self,
        grid,
        value=None,
        pdf=None,
        *,
        cdf=None,
        offsets=None,
        discrete_value=None,
        pmf=None,
        discrete_offsets=None,
        interpolations=INTERPOLATION_LINEAR,
        interpolation_boundaries=None,
        pdf_interpolations=INTERPOLATION_LINEAR,
        pdf_interpolation_boundaries=None,
        pdf_interpolation_offsets=None,
    ):
        super().__init__()
        self.grid, self.interpolations, self.interpolation_boundaries = (
            _incident_regions(grid, interpolations, interpolation_boundaries)
        )
        count = len(self.grid)
        if (value is None) != (pdf is None and cdf is None):
            raise ValueError("Continuous values require a PDF or CDF")
        if pdf is not None and cdf is not None:
            raise ValueError("Supply either PDF or CDF, not both")
        if (discrete_value is None) != (pmf is None):
            raise ValueError("Discrete values require probability masses")
        values = _packed_components(value, offsets, count, "PDF", minimum=2)
        pdfs = _packed_components(pdf, offsets, count, "PDF", minimum=2)
        cdfs = _packed_components(cdf, offsets, count, "CDF", minimum=2)
        discrete_values = _packed_components(
            discrete_value, discrete_offsets, count, "discrete", minimum=1
        )
        masses = _packed_components(pmf, discrete_offsets, count, "PMF", minimum=1)
        if not isinstance(pdf_interpolations, Integral):
            raw_rules = np.asarray(pdf_interpolation_offsets)
            if (
                raw_rules.ndim != 1
                or not np.issubdtype(raw_rules.dtype, np.integer)
                or len(raw_rules) != count + 1
                or raw_rules[0] != 0
                or np.any(np.diff(raw_rules) < 0)
                or raw_rules[-1] != len(pdf_interpolations)
                or pdf_interpolation_boundaries is None
                or len(pdf_interpolation_boundaries) != len(pdf_interpolations)
            ):
                raise ValueError("Invalid packed PDF interpolation offsets")
        xs, ys, cs, rules, ends, ds, ps, ms, continuous = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        point_offsets, rule_offsets, mass_offsets = [0], [0], [0]
        for i, (x, y, c, d, m) in enumerate(
            zip(values, pdfs, cdfs, discrete_values, masses)
        ):
            if isinstance(pdf_interpolations, Integral):
                law, boundary = pdf_interpolations, None
            else:
                a, b = raw_rules[i : i + 2]
                law = np.asarray(pdf_interpolations)[a:b]
                boundary = np.asarray(pdf_interpolation_boundaries)[a:b]
                if (x is None) != (a == b):
                    raise ValueError(
                        "PDF interpolation regions must match continuous tables"
                    )
            x, y, c, d, m, probability, law, boundary = _probability_components(
                x, y, c, d, m, law, boundary
            )
            if len(x):
                law, boundary = _distribution_regions(len(x), law, boundary)
            else:
                law, boundary = np.array([], dtype=int64), np.array([], dtype=int64)
            xs.append(x)
            ys.append(y)
            cs.append(c)
            ds.append(d)
            ps.append(m)
            ms.append(np.cumsum(m))
            continuous.append(probability)
            rules.append(law)
            ends.append(boundary)
            point_offsets.append(point_offsets[-1] + len(x))
            rule_offsets.append(rule_offsets[-1] + len(law))
            mass_offsets.append(mass_offsets[-1] + len(d))
        self.offsets = np.asarray(point_offsets, dtype=int64)
        self.discrete_offsets = np.asarray(mass_offsets, dtype=int64)
        self.value, self.pdf, self.cdf = map(np.concatenate, (xs, ys, cs))
        self.discrete_value, self.pmf, self.cmf = map(np.concatenate, (ds, ps, ms))
        self.continuous_probability = np.asarray(continuous, dtype=float64)
        self.pdf_interpolations = np.concatenate(rules)
        self.pdf_interpolation_boundaries = np.concatenate(ends)
        self.pdf_interpolation_offsets = np.asarray(rule_offsets, dtype=int64)
        # Line indices identify corresponding outcomes across linear intervals.
        region = 0
        for i in range(count - 1):
            while i + 1 >= self.interpolation_boundaries[region]:
                region += 1
            if (
                self.grid[i + 1] > self.grid[i]
                and self.interpolations[region] == INTERPOLATION_LINEAR
                and mass_offsets[i + 1] - mass_offsets[i]
                != mass_offsets[i + 2] - mass_offsets[i + 1]
            ):
                raise ValueError(
                    "Linear multi-table intervals require corresponding discrete lines"
                )

    def __repr__(self):
        return (
            super().__repr__()
            + f"  - grid: {print_1d_array(self.grid)}\n  - Probability tables: {len(self.grid)}\n"
        )


def _packed_components(values, offsets, count, name, minimum):
    """Split optional packed vectors or validate a per-grid vector sequence."""
    if values is None:
        return [None] * count
    if offsets is None:
        result = list(values)
        if len(result) != count:
            raise ValueError(f"Grid and {name} table counts must match")
        return [None if x is None or np.asarray(x).size == 0 else x for x in result]
    raw = np.asarray(offsets)
    array = np.asarray(values)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or len(raw) != count + 1
        or raw[0] != 0
        or array.ndim != 1
        or raw[-1] != len(array)
        or np.any(np.diff(raw) < 0)
        or np.any((np.diff(raw) > 0) & (np.diff(raw) < minimum))
    ):
        raise ValueError(f"Invalid packed {name} offsets")
    return [None if a == b else array[a:b] for a, b in zip(raw[:-1], raw[1:])]


def _distribution_regions(count, laws, boundaries, outer=False):
    """Return validated exclusive interpolation regions for one local grid."""
    if isinstance(laws, Integral):
        laws, boundaries = np.array([laws], dtype=int64), np.array([count], dtype=int64)
    else:
        laws = np.asarray(laws, dtype=int64)
        if boundaries is None:
            raise ValueError("Interpolation boundaries are required for multiple laws")
        boundaries = np.asarray(boundaries, dtype=int64)
    if (
        laws.ndim != 1
        or boundaries.ndim != 1
        or not len(laws)
        or len(laws) != len(boundaries)
        or boundaries[0] < min(2, count)
        or boundaries[-1] != count
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("Invalid interpolation regions")
    for law in laws:
        decode_interpolation(law)
    if outer and np.any(
        ~np.isin(laws, [INTERPOLATION_HISTOGRAM, INTERPOLATION_LINEAR])
    ):
        raise ValueError("Distribution grid interpolation must be histogram or linear")
    return laws, boundaries


def _incident_regions(grid, laws, boundaries):
    """Validate a finite conditional grid and histogram/linear table selection.

    Repeated coordinates encode a discontinuity; sampling uses the distribution
    to the right of that coordinate. Region boundaries are exclusive point indices.
    """
    grid = np.asarray(grid, dtype=float64)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) < 0)
    ):
        raise ValueError(
            "Distribution grid must be finite, nonempty, and nondecreasing"
        )
    laws, boundaries = _distribution_regions(len(grid), laws, boundaries, outer=True)
    return grid, laws, boundaries


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
    """Evaporation spectrum with temperature supplied as an evaluable Data object."""

    # MC/DC framework metadata
    label = "evaporation_distribution"
    sub_type = DISTRIBUTION_EVAPORATION

    nuclear_temperature: DataBase
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature: DataBase,
        restriction_energy: float = 0.0,
    ) -> None:
        super().__init__()

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = nuclear_temperature

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += f"  - Nuclear temperature data: {self.nuclear_temperature}\n"
        return text


# ======================================================================================
# Maxwellian
# ======================================================================================


class DistributionMaxwellian(DistributionBase):
    """Maxwellian spectrum with temperature supplied as an evaluable Data object."""

    # MC/DC framework metadata
    label = "maxwellian_distribution"
    sub_type = DISTRIBUTION_MAXWELLIAN

    nuclear_temperature: DataBase
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature: DataBase,
        restriction_energy: float = 0.0,
    ) -> None:
        super().__init__()

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = nuclear_temperature

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += f"  - Nuclear temperature data: {self.nuclear_temperature}\n"
        return text


# ======================================================================================
# Kalbach-Mann
# ======================================================================================


class DistributionKalbachMann(DistributionBase):
    """Correlated Kalbach-Mann outgoing energy-angle distribution.

    Incident-energy tables are stored in flattened arrays delimited by
    ``offset``. Probability densities are normalized per table.
    ``incident_interpolations`` selects histogram or linear dependence between
    these tables; ``interpolations`` describes the outgoing-energy PDF segments.
    """

    # MC/DC framework metadata
    label = "kalbach_mann_distribution"
    sub_type = DISTRIBUTION_KALBACH_MANN

    energy: NDArray[float64]
    incident_interpolations: NDArray[int64]
    incident_interpolation_boundaries: NDArray[int64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    precompound_factor: NDArray[float64]
    angular_slope: NDArray[float64]
    interpolations: NDArray[int64]
    precompound_tables: list[DataBase]
    angular_slope_tables: list[DataBase]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        energy_out: ArrayLike,
        pdf: ArrayLike,
        precompound_factor: ArrayLike,
        angular_slope: ArrayLike,
        *,
        incident_interpolations=INTERPOLATION_LINEAR,
        incident_interpolation_boundaries=None,
    ) -> None:
        super().__init__()

        (
            self.energy,
            self.incident_interpolations,
            self.incident_interpolation_boundaries,
        ) = _incident_regions(
            energy, incident_interpolations, incident_interpolation_boundaries
        )
        self.offset = np.asarray(offset, dtype=int64)
        self.energy_out = np.asarray(energy_out, dtype=float64)
        pdf_array = np.array(pdf, dtype=float64, copy=True)
        self.precompound_factor = np.asarray(
            precompound_factor,
            dtype=float64,
        )
        self.angular_slope = np.asarray(angular_slope, dtype=float64)
        self.interpolations = np.full(
            len(self.energy_out), INTERPOLATION_LINEAR, dtype=int64
        )
        self.precompound_tables = []
        self.angular_slope_tables = []

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
# Table energy-angle
# ======================================================================================


class DistributionTableEnergyAngle(DistributionBase):
    """Correlated tabulated outgoing energy and scattering-angle distribution.

    ``offset`` delimits outgoing-energy tables and ``cosine_offset`` delimits
    conditional cosine tables in the flattened arrays. Incident interpolation
    selects between whole energy-angle tables, separately from the local energy
    and cosine PDF interpolation.
    """

    # MC/DC framework metadata
    label = "table_energy_angle_distribution"
    sub_type = DISTRIBUTION_TABLE_ENERGY_ANGLE

    energy: NDArray[float64]
    incident_interpolations: NDArray[int64]
    incident_interpolation_boundaries: NDArray[int64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    cosine_offset_: NDArray[int64]
    cosine: NDArray[float64]
    cosine_pdf: NDArray[float64]
    cosine_cdf: NDArray[float64]
    interpolations: NDArray[int64]
    cosine_interpolations: NDArray[int64]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        energy_out: ArrayLike,
        pdf: ArrayLike,
        cosine_offset: ArrayLike,
        cosine: ArrayLike,
        cosine_pdf: ArrayLike,
        *,
        incident_interpolations=INTERPOLATION_LINEAR,
        incident_interpolation_boundaries=None,
    ) -> None:
        super().__init__()

        (
            self.energy,
            self.incident_interpolations,
            self.incident_interpolation_boundaries,
        ) = _incident_regions(
            energy, incident_interpolations, incident_interpolation_boundaries
        )
        self.offset = np.asarray(offset, dtype=int64)
        self.energy_out = np.asarray(energy_out, dtype=float64)
        pdf_array = np.array(pdf, dtype=float64, copy=True)
        self.cosine_offset_ = np.asarray(cosine_offset, dtype=int64)
        self.cosine = np.asarray(cosine, dtype=float64)
        cosine_pdf_array = np.asarray(cosine_pdf, dtype=float64)
        self.interpolations = np.full(
            len(self.energy_out), INTERPOLATION_LINEAR, dtype=int64
        )
        self.cosine_interpolations = np.full(
            len(self.cosine), INTERPOLATION_LINEAR, dtype=int64
        )

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
    number_particles : int
        Number of bodies in the final-state phase space, at least three.
    total_mass_ratio : float
        Total final-state mass divided by the emitted particle mass, greater
        than one. Together with the reaction Q-value and target mass this
        sets the maximum outgoing center-of-mass energy.
    values, probabilities : array_like
        Dimensionless outgoing-energy fractions in [0, 1] and their relative
        probability density. Transport scales a sampled fraction by the
        incident-energy-dependent maximum energy.
    interpolations : int or sequence of int, optional
        Density interpolation laws, defaulting to linear.
    interpolation_boundaries : sequence of int, optional
        Exclusive upper-point indices for the interpolation regions.
    """

    # MC/DC framework metadata
    label = "nbody_distribution"
    sub_type = DISTRIBUTION_N_BODY

    pdf: DataTable
    number_particles: int
    total_mass_ratio: float64

    def __init__(
        self,
        number_particles: int,
        total_mass_ratio: float,
        values: ArrayLike,
        probabilities: ArrayLike,
        *,
        interpolations=INTERPOLATION_LINEAR,
        interpolation_boundaries=None,
    ) -> None:
        super().__init__()

        if not isinstance(number_particles, Integral) or number_particles < 3:
            raise ValueError("N-body phase space requires at least three particles")
        if not np.isfinite(total_mass_ratio) or total_mass_ratio <= 1:
            raise ValueError(
                "N-body total mass ratio must be finite and greater than one"
            )
        self.number_particles = int(number_particles)
        self.total_mass_ratio = float64(total_mass_ratio)
        value_array = np.asarray(values, dtype=float64)
        probability_array = np.asarray(probabilities, dtype=float64)

        if value_array.ndim != 1:
            print_error("values must be one-dimensional.")
        if probability_array.ndim != 1:
            print_error("probabilities must be one-dimensional.")
        if len(value_array) != len(probability_array):
            print_error("values and probabilities must have the same length.")
        if np.any(value_array < 0) or np.any(value_array > 1):
            raise ValueError("N-body energy fractions must lie in [0, 1]")

        pdf_normalized, cdf_normalized = cdf_from_pdf(
            value_array,
            probability_array,
            interpolations,
            interpolation_boundaries,
        )

        self.pdf = DataTable(
            value_array,
            pdf_normalized,
            interpolations,
            interpolation_boundaries,
            aux=cdf_normalized,
        )

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Number of particles: {self.number_particles}\n"
        text += f"  - Total mass ratio: {self.total_mass_ratio}\n"
        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - pdf {print_1d_array(self.pdf.y)}\n"
        return text


# ======================================================================================
# Mixture
# ======================================================================================


class DistributionMixture(DistributionBase):
    """Select one component using incident-energy-dependent probability Data.

    Each probability is a Data polynomial or table evaluated at incident energy
    in eV. Tables retain independent grids, interpolation regions, and constant
    endpoint extrapolation. Evaluated probabilities must be finite, nonnegative,
    and sum to one within PROBABILITY_TOLERANCE. Every selected component uses
    the containing field's sampled quantities and reference frame.

    Components may themselves be mixtures. Separate angular distributions and
    joint energy-angle distributions must remain consistent throughout a reaction.
    """

    label = "mixture_distribution"
    sub_type = DISTRIBUTION_MIXTURE

    probabilities: list[DataBase]
    components: list[DistributionBase]

    def __init__(self, probabilities, components):
        super().__init__()
        self.probabilities = list(probabilities)
        self.components = list(components)
        if not self.components or len(self.probabilities) != len(self.components):
            raise ValueError("A mixture requires one probability per component")
        if any(
            not isinstance(item, DistributionBase) or isinstance(item, DistributionNone)
            for item in self.components
        ):
            raise ValueError("Mixture components must be sampleable Distributions")
        for probability in self.probabilities:
            _validate_mixture_probability(probability)
        if all(
            isinstance(item, DataPolynomial) and len(item.coefficients) == 1
            for item in self.probabilities
        ):
            total = sum(item.coefficients[0] for item in self.probabilities)
            if not 1.0 - PROBABILITY_TOLERANCE <= total <= 1.0 + PROBABILITY_TOLERANCE:
                raise ValueError("Mixture probabilities must sum to one")

    def __repr__(self):
        text = super().__repr__()
        for i, (probability, component) in enumerate(
            zip(self.probabilities, self.components)
        ):
            text += f"  - Component {i}: {component.label} [ID: {component.ID}]\n"
            text += f"    Probability: {probability.label} [ID: {probability.ID}]\n"
        return text


def _validate_mixture_probability(probability):
    """Check probability Data and interpolation domains before runtime packing.

    Polynomial values and the sum of independently interpolated functions are
    checked by the sampler at the actual incident energy. Constant polynomials
    and tabulated ordinates can also be bounded here.
    """
    if isinstance(probability, DataPolynomial):
        coefficients = probability.coefficients
        if not len(coefficients) or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "Mixture polynomial coefficients must be finite and nonempty"
            )
        if (
            len(coefficients) == 1
            and not 0.0 <= coefficients[0] <= 1.0 + PROBABILITY_TOLERANCE
        ):
            raise ValueError("Mixture probabilities must lie in [0, 1]")
    elif isinstance(probability, DataTable):
        x, y = probability.x, probability.y
        if (
            len(x) < 2
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) < 0)
            or x[-1] <= x[0]
            or not np.all(np.isfinite(y))
            or np.any(y < 0)
            or np.any(y > 1.0 + PROBABILITY_TOLERANCE)
        ):
            raise ValueError(
                "Mixture probability tables require finite ordered coordinates and probabilities in [0, 1]"
            )
        laws, boundaries = _distribution_regions(
            len(x), probability.interpolations, probability.interpolation_boundaries
        )
        start = 0
        for law, end in zip(laws, boundaries):
            active = x[start + 1 : end] > x[start : end - 1]
            name = decode_interpolation(law)
            if name in ("semilog-x", "log") and np.any(x[start : end - 1][active] <= 0):
                raise ValueError("Logarithmic probability coordinates must be positive")
            if name in ("semilog-y", "log") and (
                np.any(y[start : end - 1][active] <= 0)
                or np.any(y[start + 1 : end][active] <= 0)
            ):
                raise ValueError("Logarithmic probability ordinates must be positive")
            start = end - 1
    else:
        raise ValueError("Mixture probabilities must be Data polynomials or tables")


# ======================================================================================
# Independent and joint angle-energy validation
# ======================================================================================


def validate_angle_energy(angular, spectra, path, reference_frame=None):
    """Require either a separate cosine law or a joint law for every spectrum.

    DistributionNone means that each energy spectrum must supply both energy
    and angle. An explicit angular distribution, including isotropy, pairs
    with energy-only spectra. Mixed independent/joint alternatives would leave
    the reaction's angular source ambiguous and are rejected.
    """
    joint = (DistributionKalbachMann, DistributionTableEnergyAngle, DistributionNBody)
    for spectrum in spectra:
        if isinstance(spectrum, DistributionMixture):
            validate_angle_energy(angular, spectrum.components, path, reference_frame)
            continue
        if (
            isinstance(spectrum, DistributionNBody)
            and reference_frame is not None
            and reference_frame != REFERENCE_FRAME_COM
        ):
            raise ValueError(
                f"{path}: N-body phase space requires the COM reference frame"
            )
        if isinstance(angular, DistributionNone) != isinstance(spectrum, joint):
            raise ValueError(
                f"{path}: angular fields must agree with the energy distribution's correlation"
            )
