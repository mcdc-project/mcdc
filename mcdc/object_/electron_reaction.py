import numpy as np
from numpy import float64
from numpy.typing import NDArray

####

import mcdc.object_.distribution as distribution

from mcdc.constant import (
    ELECTRON_REACTION_BREMSSTRAHLUNG,
    ELECTRON_REACTION_EXCITATION,
    ELECTRON_REACTION_ELASTIC_SCATTERING,
    ELECTRON_REACTION_IONIZATION,
    INTERPOLATION_LINEAR,
    MU_CUTOFF,
)
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataBase, DataNone, DataTable
from mcdc.object_.hdf5_reader import (
    ordered_subshells,
    read_data,
    read_distribution,
    read_interpolation,
    read_reaction_properties,
)
from mcdc.object_.distribution import (
    DistributionBase,
    DistributionMultiTable,
    DistributionNone,
)
from mcdc.object_.util import decode_reference_frame
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Electron reaction base class
# ======================================================================================


class ElectronReactionBase(MCDCPolymorphic):
    """Base electron reaction data loaded from the HDF5 physics library.

    Stores the ENDF reaction identifier, cross-section segment and its offset
    into the element energy grid, and the reaction reference frame.
    """

    # MC/DC framework metadata
    label = "electron_reaction"
    sub_type = -1  # Polymorphic base

    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" is reserved for "xs"
    reference_frame: int

    secondary_products: list[SecondaryProduct]

    def __init__(self, MT, xs, xs_offset, reference_frame):
        super().__init__()
        self.secondary_products = []
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.reference_frame = reference_frame

    def __repr__(self):
        text = super().__repr__()

        text += f"  - ID: {self.ID}\n"
        text += f"  - MT: {self.MT}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
        text += f"  - Reference frame: {decode_reference_frame(self.reference_frame)}\n"
        return text


# ======================================================================================
# Electron ionization
# ======================================================================================


class ElectronReactionIonization(ElectronReactionBase):
    """Electron ionization reaction with subshell cross sections and products."""

    # MC/DC framework metadata
    label = "electron_ionization_reaction"
    sub_type = ELECTRON_REACTION_IONIZATION

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
        super().__init__(MT, xs, xs_offset, reference_frame)

        self.N_subshell = len(subshell_xs)
        self.subshell_xs = subshell_xs
        self.subshell_product = subshell_product

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an ionization reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        subshells = h5_group["subshells"]

        subshell_xs = []
        subshell_product = []

        for subshell in ordered_subshells(subshells):
            if len(subshell.get("cross-products", ())):
                raise ValueError(
                    f"{subshell.name}: cross-product transport is unsupported"
                )

            # Subshell cross section table (each has its own energy grid)
            subshell_xs.append(read_data(subshell["xs"]))

            # Secondary electron energy distribution. The sampler skips the
            # tables at or below the subshell binding energy.
            product = read_distribution(subshell["secondary-energy-distribution"])
            binding_energy = float(subshell["binding-energy"][()])
            if not isinstance(product, DistributionMultiTable):
                raise ValueError(
                    f"{subshell.name}: ionization requires multi-table spectra"
                )
            if product.grid[-1] <= binding_energy:
                print_error(
                    f"Ionization subshell {subshell.name} has no product spectrum above "
                    "its binding energy."
                )
            subshell_product.append(product)

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
    """Elastic electron scattering with large-angle cross section and cosine law.

    ``mean_deflection`` is the mean of ``1 - mu``, the transport cross section
    over the elastic one, on the cross-section grid. With it, a cosine sampled
    from the table below the incident energy is rescaled to that energy's mean
    deflection; libraries without a transport cross section carry a
    ``DataNone`` and choose between the bounding tables instead.
    """

    # MC/DC framework metadata
    label = "electron_elastic_scattering_reaction"
    sub_type = ELECTRON_REACTION_ELASTIC_SCATTERING

    mu_cut: float
    xs_large: DataBase
    mu: DistributionMultiTable
    mean_deflection: DataBase

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        xs_large,
        mu,
        mean_deflection,
    ):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.mu_cut = MU_CUTOFF
        self.xs_large = xs_large
        self.mu = mu
        self.mean_deflection = mean_deflection

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an elastic-scattering reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        large_angle = h5_group["large-angle"]
        xs_large = read_data(large_angle["xs"])
        mu = read_distribution(large_angle["angular-cosine-distribution"])
        if not isinstance(mu, DistributionMultiTable):
            raise ValueError(f"{large_angle.name}: elastic angles require multi-table")

        mean_deflection = DataNone()
        if "transport-xs" in h5_group:
            grid = h5_group.file["xs-energy-grid"][()]
            transport_data = read_data(h5_group["transport-xs"])
            energy = grid[xs_offset : xs_offset + len(xs)]
            if not isinstance(transport_data, DataTable) or not np.array_equal(
                transport_data.x, energy
            ):
                raise ValueError(
                    f"{h5_group.name}: transport-xs must use the elastic grid"
                )
            transport = transport_data.y
            ratio = np.divide(
                transport, xs, out=np.zeros_like(transport), where=xs > 0.0
            )
            mean_deflection = DataTable(energy, ratio, INTERPOLATION_LINEAR)

        return cls(MT, xs, xs_offset, reference_frame, xs_large, mu, mean_deflection)

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
    """Electron bremsstrahlung reaction with tabulated energy loss.

    ``photon_energy`` is the photon energy spectrum, sampled when photons are
    transported.
    """

    # MC/DC framework metadata
    label = "electron_bremsstrahlung_reaction"
    sub_type = ELECTRON_REACTION_BREMSSTRAHLUNG

    eloss: DataBase
    photon_energy: DistributionBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss, photon_energy):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.eloss = eloss
        self.photon_energy = photon_energy

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a bremsstrahlung reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        eloss = read_data(h5_group["energy-loss"])

        # Older libraries have no photon spectrum; the photon then takes the
        # mean energy loss
        photon_energy = DistributionNone()
        if "photon-energy-distribution" in h5_group:
            photon_energy = read_distribution(h5_group["photon-energy-distribution"])
            if not isinstance(photon_energy, DistributionMultiTable):
                raise ValueError(
                    f"{h5_group.name}: bremsstrahlung requires multi-table spectra"
                )

        return cls(MT, xs, xs_offset, reference_frame, eloss, photon_energy)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: DataTable [ID: {self.eloss.ID}]\n"
        text += f"  - Photon energy: {distribution.decode_type(self.photon_energy.type)} [ID: {self.photon_energy.ID}]\n"
        return text


# ======================================================================================
# Electron excitation
# ======================================================================================


class ElectronReactionExcitation(ElectronReactionBase):
    """Electron excitation reaction with tabulated energy loss."""

    # MC/DC framework metadata
    label = "electron_excitation_reaction"
    sub_type = ELECTRON_REACTION_EXCITATION

    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an excitation reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        eloss = read_data(h5_group["energy-loss"])

        return cls(MT, xs, xs_offset, reference_frame, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: DataTable [ID: {self.eloss.ID}]\n"
        return text


# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    """Read properties shared by all electron reactions from an HDF5 group."""

    properties = read_reaction_properties(h5_group)
    laws, _ = read_interpolation(h5_group["xs"])
    if np.any(laws != INTERPOLATION_LINEAR):
        raise ValueError(
            f"{h5_group.name}: electron cross sections require linear interpolation"
        )
    return properties
