from mcdc.constant import DATA_TABLE, DATA_POLYNOMIAL, DATA_MULTIPDF, DATA_MAXWELLIAN
from mcdc.objects import ObjectPolymorphic
from mcdc.prints import print_1d_array
from mcdc.util import cdf_from_pdf


class DataContainer(ObjectPolymorphic):
    def __init__(self, label, type_):
        super().__init__(label, type_)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


class DataTable(DataContainer):
    def __init__(self, x, y):
        label = "data_table"
        type_ = DATA_TABLE
        super().__init__(label, type_)

        self.x = x
        self.y = y

    def __repr__(self):
        text = super().__repr__()
        text += f"  - x {print_1d_array(self.x)}\n"
        text += f"  - y {print_1d_array(self.y)}\n"
        return text


class DataPolynomial(DataContainer):
    def __init__(self, coeffs):
        label = "data_polynomial"
        type_ = DATA_POLYNOMIAL
        super().__init__(label, type_)

        self.coefficients = coeffs

    def __repr__(self):
        text = super().__repr__()
        text += f"  - coefficients {print_1d_array(self.coefficients)}\n"
        return text


class DataMultiPDF(DataContainer):
    def __init__(self, grid, offset, value, pdf):
        label = "data_multipdf"
        type_ = DATA_MULTIPDF
        super().__init__(label, type_)

        self.grid = grid
        self.offset = offset
        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = cdf_from_pdf(offset, value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.grid)}\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pdf {print_1d_array(self.pdf)}\n"
        return text


class DataMaxwellian(DataContainer):
    def __init__(
        self,
        restriction_energy,
        nuclear_temperature_energy_grid,
        nuclear_temperature_value,
    ):
        label = "data_maxwellian"
        type_ = DATA_MAXWELLIAN
        super().__init__(label, type_)

        self.U = restriction_energy
        self.T = DataTable(nuclear_temperature_energy_grid, nuclear_temperature_value)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - U {print_1d_array(self.U)}\n"
        text += f"  - T energy_grid {print_1d_array(self.T.x)}\n"
        text += f"  - T {print_1d_array(self.T.y)}\n"
        return text


def decode_type(type_):
    if type_ == DATA_TABLE:
        return "Data (Table)"
    elif type_ == DATA_POLYNOMIAL:
        return "Data (Polynomial function)"
    elif type_ == DATA_MULTIPDF:
        return "Data (Multi PDF)"
    elif type_ == DATA_MAXWELLIAN:
        return "Data (Maxwellian spectrum)"
