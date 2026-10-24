import numpy as np

from numpy import float64, int64
from numpy.typing import NDArray

####

from mcdc.constant import (
    INTERPOLATION_LINEAR,
    PHOTON_REACTION_COHERENT_SCATTERING,
    PHOTON_REACTION_INCOHERENT_SCATTERING,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    REFERENCE_FRAME_LAB,
)
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataBase
from mcdc.object_.hdf5_reader import (
    read_data,
    read_reaction_properties,
    set_xs_interpolation,
    text,
)
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.object_.util import decode_reference_frame
from mcdc.print_ import print_1d_array

# ======================================================================================
# Photon reaction base class
# ======================================================================================


class PhotonReactionBase(MCDCPolymorphic):
    """Base photon reaction data loaded from the HDF5 physics library.

    Stores the ENDF MF=23 reaction identifier, the cross-section segment and its
    offset into the element energy grid, and the reaction reference frame. The
    physics itself lives in :mod:`mcdc.transport.physics.photon`; these objects
    carry only data.
    """

    # MC/DC framework metadata
    label = "photon_reaction"
    sub_type = -1  # Polymorphic base

    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" is reserved for "xs"
    xs_interpolations: NDArray[int64]
    xs_interpolation_boundaries: NDArray[int64]
    reference_frame: int

    secondary_products: list[SecondaryProduct]

    def __init__(self, MT, xs, xs_offset, reference_frame):
        super().__init__()
        self.secondary_products = []
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.xs_interpolations = np.array([INTERPOLATION_LINEAR], dtype=int64)
        self.xs_interpolation_boundaries = np.array([len(xs)], dtype=int64)
        self.reference_frame = reference_frame

    def __repr__(self):
        text = super().__repr__()
        text += f"  - ID: {self.ID}\n"
        text += f"  - MT: {self.MT}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
        text += f"  - Reference frame: {decode_reference_frame(self.reference_frame)}\n"
        return text


def decode_type(type_):
    """Return the display name for a packed photon reaction code."""

    if type_ == PHOTON_REACTION_COHERENT_SCATTERING:
        return "Photon coherent (Rayleigh) scattering"
    elif type_ == PHOTON_REACTION_INCOHERENT_SCATTERING:
        return "Photon incoherent (Compton) scattering"
    elif type_ == PHOTON_REACTION_PHOTOELECTRIC:
        return "Photon photoelectric absorption"
    elif type_ == PHOTON_REACTION_PAIR_PRODUCTION:
        return "Photon pair production"


# ======================================================================================
# Coherent (Rayleigh) scattering
# ======================================================================================


class PhotonReactionCoherentScattering(PhotonReactionBase):
    """Coherent scattering with the tabulated atomic form factor F(q, Z).

    Elastic: the photon energy is unchanged and only the direction is deflected.
    The form factor carries its own momentum-transfer grid, so it is stored as a
    :class:`DataTable` rather than packed against the element energy grid.
    """

    # MC/DC framework metadata
    label = "photon_coherent_scattering_reaction"
    sub_type = PHOTON_REACTION_COHERENT_SCATTERING

    form_factor: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, form_factor):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.form_factor = form_factor

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a coherent-scattering reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        form_factor = read_data(h5_group["form-factor"])
        return set_xs_interpolation(
            cls(MT, xs, xs_offset, reference_frame, form_factor), h5_group
        )

    def __repr__(self):
        text = super().__repr__()
        text += (
            f"  - Form factor: {self.form_factor.label} [ID: {self.form_factor.ID}]\n"
        )
        return text


# ======================================================================================
# Incoherent (Compton) scattering
# ======================================================================================


class PhotonReactionIncoherentScattering(PhotonReactionBase):
    """Incoherent scattering, sampled with Klein-Nishina kinematics.

    The tabulated cross section sets the interaction probability; the outgoing
    energy and angle come from the Klein-Nishina distribution sampled by Kahn's
    composition-rejection method in
    :mod:`mcdc.transport.physics.photon.native`.
    """

    # MC/DC framework metadata
    label = "photon_incoherent_scattering_reaction"
    sub_type = PHOTON_REACTION_INCOHERENT_SCATTERING

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an incoherent-scattering reaction from a library HDF5 group."""
        # TODO: Load and store scattering-function when transport applies it
        # to the Klein-Nishina angular distribution.
        return set_xs_interpolation(cls(*set_basic_properties(h5_group)), h5_group)


# ======================================================================================
# Photoelectric absorption
# ======================================================================================


class PhotonReactionPhotoelectric(PhotonReactionBase):
    """Photoelectric absorption with shell-resolved subshell cross sections.

    Each subshell carries its own energy grid, so the subshell cross sections are
    :class:`DataTable` objects, mirroring
    :class:`~mcdc.object_.electron_reaction.ElectronReactionIonization`.
    """

    # MC/DC framework metadata
    label = "photon_photoelectric_reaction"
    sub_type = PHOTON_REACTION_PHOTOELECTRIC

    N_subshell: int
    subshell_xs: list[DataBase]
    subshell_designators: NDArray[int64]

    def __init__(
        self, MT, xs, xs_offset, reference_frame, subshell_xs, subshell_designators
    ):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.N_subshell = len(subshell_xs)
        self.subshell_xs = subshell_xs
        self.subshell_designators = np.asarray(subshell_designators, dtype=int64)

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a photoelectric reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        subshell_xs, designators = [], []
        for subshell in ordered_subshells(h5_group.get("subshells", {})):
            if len(subshell.get("cross-products", ())):
                raise ValueError(
                    f"{subshell.name}: cross-product transport is unsupported"
                )
            subshell_xs.append(read_data(subshell["xs"]))
            designators.append(int(subshell["designator"][()]))

        return set_xs_interpolation(
            cls(MT, xs, xs_offset, reference_frame, subshell_xs, designators), h5_group
        )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Number of subshells: {self.N_subshell}\n"
        for i in range(self.N_subshell):
            text += f"    - Subshell {i+1} XS: {self.subshell_xs[i].label} [ID: {self.subshell_xs[i].ID}]\n"
        return text


# ======================================================================================
# Pair production
# ======================================================================================


class PhotonReactionPairProduction(PhotonReactionBase):
    """Pair production in the nuclear (MT-517) or electron (MT-515) field.

    The channel cross section supplies its threshold. The photon converts to
    an electron-positron pair whose kinetic energy deposits locally; the positron annihilates,
    emitting two 511 keV photons; both are handled in
    :mod:`mcdc.transport.physics.photon.native`.
    """

    # MC/DC framework metadata
    label = "photon_pair_production_reaction"
    sub_type = PHOTON_REACTION_PAIR_PRODUCTION

    electron_field: bool

    def __init__(self, MT, xs, xs_offset, reference_frame, electron_field):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.electron_field = electron_field

    @classmethod
    def from_h5_group(cls, h5_group):
        """Read pair-production cross sections and the producing field."""
        field = text(h5_group["field"][()])
        if field not in ("electron", "nuclear"):
            raise ValueError(
                f"{h5_group.name}: unknown pair-production field {field!r}"
            )
        return set_xs_interpolation(
            cls(*set_basic_properties(h5_group), field == "electron"), h5_group
        )


# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    """Read shared reaction fields and require the photon model's LAB frame."""
    MT, xs, xs_offset, reference_frame = read_reaction_properties(h5_group)
    if reference_frame != REFERENCE_FRAME_LAB:
        raise ValueError(f"{h5_group.name}: photon reactions require LAB data")
    return MT, xs, xs_offset, reference_frame


def ordered_subshells(group):
    """Order shell records by explicit designator, independently of group names."""
    children = sorted(group.values(), key=lambda child: int(child["designator"][()]))
    designators = [int(child["designator"][()]) for child in children]
    if len(set(designators)) != len(designators) or any(i <= 0 for i in designators):
        raise ValueError("Subshell designators must be distinct positive integers")
    return children
