from numpy import float64
from numpy.typing import NDArray

####

import mcdc.object_.distribution as distribution

from mcdc.constant import (
    ELECTRON_REACTION_BREMSSTRAHLUNG,
    ELECTRON_REACTION_EXCITATION,
    ELECTRON_REACTION_ELASTIC_SCATTERING,
    ELECTRON_REACTION_IONIZATION,
    MU_CUTOFF,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
)
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.data import DataBase, DataTable
from mcdc.object_.distribution import DistributionBase, DistributionMultiTable
from mcdc.print_ import print_1d_array

# ======================================================================================
# Electron reaction base class
# ======================================================================================


class ElectronReactionBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "electron_reaction"
    #
    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" is reserved for "xs"
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
    if type_ == ELECTRON_REACTION_IONIZATION:
        return "Electron ionization"
    elif type_ == ELECTRON_REACTION_ELASTIC_SCATTERING:
        return "Electron elastic scattering"
    elif type_ == ELECTRON_REACTION_BREMSSTRAHLUNG:
        return "Electron bremsstrahlung"
    elif type_ == ELECTRON_REACTION_EXCITATION:
        return "Electron excitation"


def decode_reference_frame(type_):
    if type_ == REFERENCE_FRAME_LAB:
        return "Laboratory"
    elif type_ == REFERENCE_FRAME_COM:
        return "Center of mass"


# ======================================================================================
# Electron ionization
# ======================================================================================


class ElectronReactionIonization(ElectronReactionBase):
    # Annotations for Numba mode
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
        type_ = ELECTRON_REACTION_IONIZATION
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
            subshell_xs.append(
                DataTable(subshell["energy_grid"][()], subshell["xs"][()])
            )

            # Secondary electron energy distribution
            product = subshell["product"]
            if "CDF" in product:
                subshell_product.append(
                    DistributionMultiTable(
                        product["energy_grid"][()],
                        product["energy_offset"][()],
                        product["value"][()],
                        cdf=product["CDF"][()],
                    )
                )
            else:
                subshell_product.append(
                    DistributionMultiTable(
                        product["energy_grid"][()],
                        product["energy_offset"][()],
                        product["value"][()],
                        product["PDF"][()],
                    )
                )


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
            product = self.subshell_product[i]
            text += f"      - Secondary electron spectrum: {distribution.decode_type(product.type)} [ID: {product.ID}]\n"
        return text


# ======================================================================================
# Electron elastic scattering
# ======================================================================================


class ElectronReactionElasticScattering(ElectronReactionBase):
    # Annotations for Numba mode
    label: str = "electron_elastic_scattering_reaction"
    #
    mu_cut: float
    xs_large: DataBase
    mu: DistributionMultiTable

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        xs_large,
        mu,
    ):
        type_ = ELECTRON_REACTION_ELASTIC_SCATTERING
        super().__init__(type_, MT, xs, xs_offset, reference_frame)
        self.mu_cut = MU_CUTOFF
        self.xs_large = xs_large
        self.mu = mu

    @classmethod
    def from_h5_group(cls, h5_group):
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        large_angle = h5_group["large_angle"]
        xs_large = DataTable(large_angle["xs_energy"][()], large_angle["xs"][()])

        mu_group = large_angle["scattering_cosine"]
        if "CDF" in mu_group:
            mu = DistributionMultiTable(
                mu_group["energy_grid"][()],
                mu_group["energy_offset"][()],
                mu_group["value"][()],
                cdf=mu_group["CDF"][()],
            )
        else:
            mu = DistributionMultiTable(
                mu_group["energy_grid"][()],
                mu_group["energy_offset"][()],
                mu_group["value"][()],
                mu_group["PDF"][()],
            )


        return cls(MT, xs, xs_offset, reference_frame, xs_large, mu)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Mu cut: {self.mu_cut}\n"
        text += f"  - Large angle XS: DataTable [ID: {self.xs_large.ID}]\n"
        text += f"  - Scattering cosine: {distribution.decode_type(self.mu.type)} [ID: {self.mu.ID}]\n"
        return text


# ======================================================================================
# Electron bremsstrahlung
# ======================================================================================


class ElectronReactionBremsstrahlung(ElectronReactionBase):
    # Annotations for Numba mode
    label: str = "electron_bremsstrahlung_reaction"
    #
    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        type_ = ELECTRON_REACTION_BREMSSTRAHLUNG
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


class ElectronReactionExcitation(ElectronReactionBase):
    # Annotations for Numba mode
    label: str = "electron_excitation_reaction"
    #
    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        type_ = ELECTRON_REACTION_EXCITATION
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
