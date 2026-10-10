from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import (
    INTERPOLATION_LINEAR,
    PHOTON_REACTION_COHERENT,
    PHOTON_REACTION_INCOHERENT,
    PHOTON_REACTION_PAIR_PRODUCTION,
    PHOTON_REACTION_PHOTOELECTRIC,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
)
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataBase, DataTable
from mcdc.object_.electron_reaction import read_energy
from mcdc.object_.secondary_product import SecondaryProduct
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


def decode_type(type_):
    """Return the display name for a packed photon reaction code."""

    if type_ == PHOTON_REACTION_COHERENT:
        return "Photon coherent (Rayleigh) scattering"
    elif type_ == PHOTON_REACTION_INCOHERENT:
        return "Photon incoherent (Compton) scattering"
    elif type_ == PHOTON_REACTION_PHOTOELECTRIC:
        return "Photon photoelectric absorption"
    elif type_ == PHOTON_REACTION_PAIR_PRODUCTION:
        return "Photon pair production"


def decode_reference_frame(type_):
    """Return the display name for a packed reference-frame code."""

    if type_ == REFERENCE_FRAME_LAB:
        return "Laboratory"
    elif type_ == REFERENCE_FRAME_COM:
        return "Center of mass"


# ======================================================================================
# Coherent (Rayleigh) scattering
# ======================================================================================


class PhotonReactionCoherent(PhotonReactionBase):
    """Coherent scattering with the tabulated atomic form factor F(q, Z).

    Elastic: the photon energy is unchanged and only the direction is deflected.
    The form factor carries its own momentum-transfer grid, so it is stored as a
    :class:`DataTable` rather than packed against the element energy grid.
    """

    # MC/DC framework metadata
    label = "photon_coherent_reaction"
    sub_type = PHOTON_REACTION_COHERENT

    form_factor: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, form_factor):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.form_factor = form_factor

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a coherent-scattering reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        # Linear, not log-log: the library's momentum-transfer grid starts at
        # q = 0, where log(q) is undefined.
        form_factor_group = h5_group["form_factor"]
        form_factor = DataTable(
            form_factor_group["momentum_transfer"][()],
            form_factor_group["value"][()],
            INTERPOLATION_LINEAR,
        )

        return cls(MT, xs, xs_offset, reference_frame, form_factor)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Form factor: DataTable [ID: {self.form_factor.ID}]\n"
        return text


# ======================================================================================
# Incoherent (Compton) scattering
# ======================================================================================


class PhotonReactionIncoherent(PhotonReactionBase):
    """Incoherent scattering, sampled with Klein-Nishina kinematics.

    The tabulated cross section sets the interaction probability; the outgoing
    energy and angle come from the Klein-Nishina distribution sampled by Kahn's
    composition-rejection method in
    :mod:`mcdc.transport.physics.photon.native`.
    """

    # MC/DC framework metadata
    label = "photon_incoherent_reaction"
    sub_type = PHOTON_REACTION_INCOHERENT

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an incoherent-scattering reaction from a library HDF5 group."""
        return cls(*set_basic_properties(h5_group))


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

    def __init__(self, MT, xs, xs_offset, reference_frame, subshell_xs):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.N_subshell = len(subshell_xs)
        self.subshell_xs = subshell_xs

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a photoelectric reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        subshell_xs = []
        if "subshells" in h5_group:
            subshells = h5_group["subshells"]
            for name in subshells:
                subshell = subshells[name]
                # Linear, matching ElectronReactionIonization: subshell cross
                # sections are identically zero below their absorption edge, and
                # log-log interpolation is undefined across a zero ordinate.
                subshell_xs.append(
                    DataTable(
                        read_energy(subshell["energy_grid"]),
                        subshell["xs"][()],
                        INTERPOLATION_LINEAR,
                    )
                )

        return cls(MT, xs, xs_offset, reference_frame, subshell_xs)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Number of subshells: {self.N_subshell}\n"
        for i in range(self.N_subshell):
            text += (
                f"    - Subshell {i+1} XS: DataTable [ID: {self.subshell_xs[i].ID}]\n"
            )
        return text


# ======================================================================================
# Pair production
# ======================================================================================


class PhotonReactionPairProduction(PhotonReactionBase):
    """Pair production in the nuclear (MT-517) or electron (MT-515) field.

    Above the ``2 m_e c^2`` threshold the photon converts to an electron-positron
    pair. The pair kinetic energy deposits locally and the positron annihilates,
    emitting two 511 keV photons; both are handled in
    :mod:`mcdc.transport.physics.photon.native`.
    """

    # MC/DC framework metadata
    label = "photon_pair_production_reaction"
    sub_type = PHOTON_REACTION_PAIR_PRODUCTION

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a pair-production reaction from a library HDF5 group."""
        return cls(*set_basic_properties(h5_group))


# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    """Read properties shared by all photon reactions from an HDF5 group."""

    MT = h5_group.attrs["MT"][()]
    xs = h5_group["xs"][()]
    xs_offset = h5_group["xs"].attrs["offset"]
    reference_frame = REFERENCE_FRAME_LAB
    if "reference_frame" in h5_group:
        frame_name = h5_group["reference_frame"][()]
        if isinstance(frame_name, bytes):
            frame_name = frame_name.decode("utf-8")
        if frame_name == "COM":
            reference_frame = REFERENCE_FRAME_COM
    return MT, xs, xs_offset, reference_frame
