from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import (
    DATA_NONE,
    DATA_TABLE,
    DATA_POLYNOMIAL,
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOG,
)
from mcdc.object_.base import ObjectPolymorphic
from mcdc.print_ import print_1d_array

# ======================================================================================
# Data base class
# ======================================================================================


class DataBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "data"

    def __init__(self, type_, register=True):
        super().__init__(type_, register)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


def decode_type(type_):
    if type_ == DATA_NONE:
        return "Data (None)"
    elif type_ == DATA_TABLE:
        return "Data (Table)"
    elif type_ == DATA_POLYNOMIAL:
        return "Data (Polynomial function)"


# ======================================================================================
# None
# ======================================================================================
# Placeholder for data that does not need to store anything:
#   - Fission multiplicity and delayed precursor data for non-fissionable nuclide


class DataNone(DataBase):
    # Annotations for Numba mode
    label: str = "none_data"

    def __init__(self):
        type_ = DATA_NONE
        super().__init__(type_, False)
        self.ID = 0


# ======================================================================================
# Table data
# ======================================================================================


class DataTable(DataBase):
    # Annotations for Numba mode
    label: str = "table_data"
    #
    x: NDArray[float64]
    y: NDArray[float64]
    interpolation: int

    def __init__(self, x, y, interpolation=INTERPOLATION_LINEAR):
        type_ = DATA_TABLE
        super().__init__(type_)

        self.x = x
        self.y = y
        self.interpolation = interpolation

    def __repr__(self):
        text = super().__repr__()
        text += f"  - x {print_1d_array(self.x)}\n"
        text += f"  - y {print_1d_array(self.y)}\n"
        if self.interpolation == INTERPOLATION_LINEAR:
            text += f"  - Interpolation: linear\n"
        elif self.interpolation == INTERPOLATION_LOG:
            text += f"  - Interpolation: log\n"
        return text


# ======================================================================================
# Polynomial data
# ======================================================================================


class DataPolynomial(DataBase):
    # Annotations for Numba mode
    label: str = "polynomial_data"
    #
    coefficients: NDArray[float64]

    def __init__(self, coeffs):
        type_ = DATA_POLYNOMIAL
        super().__init__(type_)

        self.coefficients = coeffs

    def __repr__(self):
        text = super().__repr__()
        text += f"  - coefficients {print_1d_array(self.coefficients)}\n"
        return text
