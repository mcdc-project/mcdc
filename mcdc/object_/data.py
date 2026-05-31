import numpy as np

from collections.abc import Sequence
from numpy import float64, int64
from numpy.typing import NDArray
from typing import Annotated

####

from mcdc.constant import (
    DATA_NONE,
    DATA_TABLE,
    DATA_POLYNOMIAL,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
    INTERPOLATION_SEMILOGX,
    INTERPOLATION_SEMILOGY,
    INTERPOLATION_LOG,
)
from mcdc.object_.base import ObjectPolymorphic
from mcdc.print_ import print_1d_array, print_error

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
    """
    Tabulated one-dimensional data with ENDF/ACE-style interpolation regions.

    The interpolation laws apply only to `y`. The optional `aux` array stores
    additional data aligned with `x`, such as a CDF associated with a PDF table.
    Auxiliary data are stored for lookup only and are not interpolated.
    """

    # Annotations for Numba mode
    label: str = "table_data"
    #
    N: int
    x: NDArray[float64]
    y: NDArray[float64]
    #
    interpolations: NDArray[int64]
    interpolation_boundaries: NDArray[int64]
    #
    N_aux: int
    aux: Annotated[NDArray[np.float64], ("N_aux", "N")]

    def __init__(
        self,
        x: NDArray[float64],
        y: NDArray[float64],
        interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None = None,
        aux: NDArray[float64] | None = None,
    ) -> None:
        """
        Create a tabulated data object.

        Parameters
        ----------
        x : ndarray of float64
            Independent variable values.
        y : ndarray of float64
            Dependent variable values.
        interpolations : int or sequence of int
            Interpolation law or list of interpolation laws.
        interpolation_boundaries : sequence of int, optional
            Region boundaries. Required when `interpolations` is a sequence.
        aux : ndarray of float64, optional
            Auxiliary data aligned with `x`. A one-dimensional array is stored
            internally with shape `(1, N)`. A two-dimensional array must have
            shape `(N_aux, N)`. Auxiliary data are not interpolated.

        Notes
        -----
        Boundaries are stored as Python-style exclusive upper indices.
        Therefore the final boundary should be `len(x)`.
        """

        # Set type
        type_ = DATA_TABLE
        super().__init__(type_)

        # Set primary data
        self.x = x
        self.y = y
        self.N = len(x)

        # Basic size checks
        if self.N == 0:
            print_error("x and y must contain at least one value.")
        if len(self.y) != self.N:
            print_error("x and y must have the same length.")

        # Set auxiliary data
        if aux is None:
            self.N_aux = 0
            self.aux = np.zeros((0, self.N), dtype=float)
        elif aux.ndim == 1:
            self.N_aux = 1
            if len(aux) != self.N:
                print_error("1D aux must have the same length as x.")
            self.aux = np.zeros((1, self.N), dtype=float)
            self.aux[0, :] = aux
        elif aux.ndim == 2:
            self.N_aux = aux.shape[0]
            if aux.shape[1] != self.N:
                print_error("2D aux must have shape (N_aux, len(x)).")
            self.aux = aux
        else:
            print_error("aux must be None, 1D, or 2D.")

        # Set interpolations and boundaries
        if isinstance(interpolations, int):
            self.interpolations = np.array([interpolations], dtype=int)
            self.interpolation_boundaries = np.array([self.N], dtype=int)
        else:
            self.interpolations = np.array(interpolations, dtype=int)

            if interpolation_boundaries is None:
                print_error("Missing interpolation boundaries in tabulated data.")

            self.interpolation_boundaries = np.array(
                interpolation_boundaries,
                dtype=int,
            )

        # Interpolation-region checks
        if len(self.interpolations) == 0:
            print_error("At least one interpolation law is required.")

        if len(self.interpolations) != len(self.interpolation_boundaries):
            print_error(
                "interpolations and interpolation_boundaries must have the same length."
            )

        if self.interpolation_boundaries[-1] != self.N:
            print_error("Last interpolation boundary must equal len(x).")

        previous = 0
        for boundary in self.interpolation_boundaries:
            if boundary <= previous:
                print_error("interpolation_boundaries must be strictly increasing.")
            if boundary > self.N:
                print_error("interpolation_boundaries cannot exceed len(x).")
            previous = boundary

    def __repr__(self) -> str:
        """Return a human-readable summary of the tabulated data."""
        text = super().__repr__()
        text += f"  - x {print_1d_array(self.x)}\n"
        text += f"  - y {print_1d_array(self.y)}\n"

        if self.N_aux > 0:
            text += f"  - aux shape: {self.aux.shape}\n"
            for i in range(len(self.aux)):
                text += f"    - aux[{i}]: {print_1d_array(self.aux[i])}\n"

        if len(self.interpolations) == 1:
            text += (
                f"  - Interpolation: "
                f"{decode_interpolation(self.interpolations[0])}\n"
            )
        else:
            text += "  - Interpolation regions:\n"

            start = 0
            for interp, end in zip(
                self.interpolations,
                self.interpolation_boundaries,
            ):
                text += f"    - [{start}, {end}): " f"{decode_interpolation(interp)}\n"
                start = end

        return text


def decode_interpolation(type_) -> str:
    """Convert an interpolation integer code to its string name."""
    if type_ == INTERPOLATION_HISTOGRAM:
        return "histogram"
    elif type_ == INTERPOLATION_LINEAR:
        return "linear"
    elif type_ == INTERPOLATION_SEMILOGX:
        return "semilog-x"
    elif type_ == INTERPOLATION_SEMILOGY:
        return "semilog-y"
    elif type_ == INTERPOLATION_LOG:
        return "log"
    else:
        raise ValueError(f"Unknown interpolation type: {type_}")


def encode_interpolation(type_) -> int:
    """Convert an interpolation string name to its integer code."""
    if type_ == "histogram":
        return INTERPOLATION_HISTOGRAM
    elif type_ == "linear":
        return INTERPOLATION_LINEAR
    elif type_ == "semilog-x":
        return INTERPOLATION_SEMILOGX
    elif type_ == "semilog-y":
        return INTERPOLATION_SEMILOGY
    elif type_ == "log":
        return INTERPOLATION_LOG
    else:
        raise ValueError(f"Unknown interpolation name: {type_}")


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
