import numpy as np

from numpy import float64, int64
from numpy.typing import NDArray

####

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
)
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.data import DataTable
from mcdc.object_.util import cdf_from_pdf, multi_cdf_from_pdf, cmf_from_pmf
from mcdc.print_ import print_1d_array

# ======================================================================================
# Distribution base class
# ======================================================================================


class DistributionBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "distribution"

    def __init__(self, type_, register=True):
        super().__init__(type_, register)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


def decode_type(type_):
    if type_ == DISTRIBUTION_NONE:
        return "Distribution (None)"
    elif type_ == DISTRIBUTION_PMF:
        return "Distribution (PMF)"
    elif type_ == DISTRIBUTION_TABULATED:
        return "Distribution (Tabulated)"
    elif type_ == DISTRIBUTION_MULTITABLE:
        return "Distribution (Multi Table)"
    elif type_ == DISTRIBUTION_LEVEL_SCATTERING:
        return "Distribution (Level scattering)"
    elif type_ == DISTRIBUTION_EVAPORATION:
        return "Distribution (Evaporation)"
    elif type_ == DISTRIBUTION_MAXWELLIAN:
        return "Distribution (Maxwellian spectrum)"
    elif type_ == DISTRIBUTION_KALBACH_MANN:
        return "Distribution (Kalbach-Mann)"
    elif type_ == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
        return "Distribution (Tabulated energy-angle)"
    elif type_ == DISTRIBUTION_N_BODY:
        return "Distribution (N-body)"


# ======================================================================================
# None
# ======================================================================================
# Placeholder for distribution that does not need to store data:
#   - Isotropic
#   - Energy-correlated angle (stored in the energy distribution)


class DistributionNone(DistributionBase):
    # Annotations for Numba mode
    label: str = "none_distribution"

    def __init__(self):
        type_ = DISTRIBUTION_NONE
        super().__init__(type_, False)
        self.ID = 0


# ======================================================================================
# Probability Mass Function (PMF)
# ======================================================================================


class DistributionPMF(DistributionBase):
    # Annotations for Numba mode
    label: str = "pmf_distribution"
    #
    value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]

    def __init__(self, value, pmf):
        type_ = DISTRIBUTION_PMF
        super().__init__(type_)

        self.value = value
        self.pmf = pmf

        self.pmf, self.cmf = cmf_from_pmf(pmf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pmf {print_1d_array(self.pmf)}\n"
        return text


# ======================================================================================
# Tabulated
# ======================================================================================


class DistributionTabulated(DistributionBase):
    # Annotations for Numba mode
    label: str = "tabulated_distribution"
    #
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]

    def __init__(self, value, pdf):
        type_ = DISTRIBUTION_TABULATED
        super().__init__(type_)

        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = cdf_from_pdf(value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pdf {print_1d_array(self.pdf)}\n"
        return text


# ======================================================================================
# Multi-table
# ======================================================================================


class DistributionMultiTable(DistributionBase):
    # Annotations for Numba mode
    label: str = "multi_table_distribution"
    #
    grid: NDArray[float64]
    offset: NDArray[int64]
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]

    def __init__(self, grid, offset, value, pdf):
        type_ = DISTRIBUTION_MULTITABLE
        super().__init__(type_)

        self.grid = grid
        self.offset = offset
        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = multi_cdf_from_pdf(offset, value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.grid)}\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pdf {print_1d_array(self.pdf)}\n"
        return text


# ======================================================================================
# Level scattering
# ======================================================================================


class DistributionLevelScattering(DistributionBase):
    # Annotations for Numba mode
    label: str = "level_scattering_distribution"
    #
    C1: float
    C2: float

    def __init__(self, C1, C2):
        type_ = DISTRIBUTION_LEVEL_SCATTERING
        super().__init__(type_)

        self.C1 = C1
        self.C2 = C2

    def __repr__(self):
        text = super().__repr__()
        text += f"  - C1 {print_1d_array(self.C1)} [/eV^l]\n"
        text += f"  - C2: {self.C2}\n"
        return text


# ======================================================================================
# Evaporation
# ======================================================================================


class DistributionEvaporation(DistributionBase):
    # Annotations for Numba mode
    label: str = "evaporation_distribution"
    #
    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid,
        nuclear_temperature_value,
        restriction_energy,
        temperature_interpolation,
    ):
        type_ = DISTRIBUTION_EVAPORATION
        super().__init__(type_)

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolation,
        )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += f"  - Nuclear temperature {print_1d_array(self.nuclear_temperature.y)} [eV]\n"
        text += f"  - Nuclear temperature energy grid {print_1d_array(self.nuclear_temperature.x)} [eV]\n"
        return text


# ======================================================================================
# Maxwellian distribution
# ======================================================================================


class DistributionMaxwellian(DistributionBase):
    # Annotations for Numba mode
    label: str = "maxwellian_distribution"
    #
    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid,
        nuclear_temperature_value,
        restriction_energy,
        temperature_interpolation,
    ):
        type_ = DISTRIBUTION_MAXWELLIAN
        super().__init__(type_)

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolation,
        )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += f"  - Nuclear temperature {print_1d_array(self.nuclear_temperature.y)} [eV]\n"
        text += f"  - Nuclear temperature energy grid {print_1d_array(self.nuclear_temperature.x)} [eV]\n"
        return text


# ======================================================================================
# Kalbach-Mann
# ======================================================================================


class DistributionKalbachMann(DistributionBase):
    # Annotations for Numba mode
    label: str = "kalbach_mann_distribution"
    #
    energy: NDArray[float64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    precompound_factor: NDArray[float64]
    angular_slope: NDArray[float64]

    def __init__(
        self, energy, offset, energy_out, pdf, precompound_factor, angular_slope
    ):
        type_ = DISTRIBUTION_KALBACH_MANN
        super().__init__(type_)

        self.energy = energy
        self.offset = offset

        self.energy_out = energy_out
        self.pdf = pdf

        self.precompound_factor = precompound_factor
        self.angular_slope = angular_slope

        self.pdf, self.cdf = multi_cdf_from_pdf(offset, energy_out, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.energy)} [eV]\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - energy {print_1d_array(self.energy_out)} [eV]\n"
        text += f"  - energy-pdf {print_1d_array(self.pdf)} [/eV]\n"
        text += f"  - precompound factor {print_1d_array(self.precompound_factor)}\n"
        text += f"  - angular slope {print_1d_array(self.angular_slope)}\n"
        return text


# ======================================================================================
# Tabulated energy-angle
# ======================================================================================


class DistributionTabulatedEnergyAngle(DistributionBase):
    # Annotations for Numba mode
    label: str = "tabulated_energy_angle_distribution"
    #
    energy: NDArray[float64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    cosine_offset_: NDArray[int64]  # "cosine_offset" is reserved to describe "cosine"
    cosine: NDArray[float64]
    cosine_pdf: NDArray[float64]
    cosine_cdf: NDArray[float64]

    def __init__(
        self, energy, offset, energy_out, pdf, cosine_offset, cosine, cosine_pdf
    ):
        type_ = DISTRIBUTION_TABULATED_ENERGY_ANGLE
        super().__init__(type_)

        self.energy = energy
        self.offset = offset

        self.energy_out = energy_out
        self.pdf = pdf
        self.cosine_offset_ = cosine_offset

        self.cosine = cosine
        self.cosine_pdf = cosine_pdf

        self.pdf, self.cdf = multi_cdf_from_pdf(offset, energy_out, pdf)

        self.cosine_cdf = np.zeros_like(self.cosine_pdf)
        for i in range(len(offset)):
            start = offset[i]
            if i + 1 < len(offset):
                end = offset[i + 1]
            else:
                end = len(cosine)
            inner_offset = cosine_offset[start:end]

            start = inner_offset[0]
            if i + 1 < len(offset):
                end = cosine_offset[end]
            else:
                end = len(cosine)

            inner_offset_local = inner_offset - inner_offset[0]
            self.cosine_pdf[start:end], self.cosine_cdf[start:end] = multi_cdf_from_pdf(
                inner_offset_local, cosine[start:end], cosine_pdf[start:end]
            )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.energy)} [eV]\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - energy {print_1d_array(self.energy_out)} [eV]\n"
        text += f"  - energy-pdf {print_1d_array(self.pdf)} [/eV]\n"
        text += f"  - cosine-offset {print_1d_array(self.cosine_offset_)}\n"
        text += f"  - cosine {print_1d_array(self.cosine)}\n"
        text += f"  - cosine-pdf {print_1d_array(self.cosine_pdf)}\n"
        return text


# ======================================================================================
# N-Body
# ======================================================================================


class DistributionNBody(DistributionBase):
    # Annotations for Numba mode
    label: str = "nbody_distribution"
    #
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]

    def __init__(self, value, pdf):
        type_ = DISTRIBUTION_N_BODY
        super().__init__(type_)

        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = cdf_from_pdf(value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pdf {print_1d_array(self.pdf)}\n"
        return text
