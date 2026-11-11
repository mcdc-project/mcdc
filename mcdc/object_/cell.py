from __future__ import annotations
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from mcdc.object_.surface import Surface

####

import numpy as np
import sympy

from numpy import float64
from numpy.typing import NDArray
from operator import attrgetter
from types import NoneType
from typing import Annotated, Sequence
from sympy.logic.boolalg import Boolean

####

from mcdc.constant import (
    BOOL_AND,
    BOOL_NOT,
    BOOL_OR,
    FILL_LATTICE,
    FILL_MATERIAL,
    FILL_NONE,
    FILL_UNIVERSE,
    PI,
)
from mcdc.object_.base import MCDCObject
from mcdc.object_.material import Material
from mcdc.object_.tally import TallyInteraction, TallyTracklength
from mcdc.object_.universe import Universe, Lattice
from mcdc.print_ import print_error

# ======================================================================================
# Region
# ======================================================================================

RegionType: TypeAlias = Literal[
    "all",
    "halfspace",
    "intersection",
    "union",
    "complement",
]


class Region:
    """Boolean combination of oriented surface half-spaces.

    Regions are normally built with unary ``+`` and ``-`` on
    :class:`~mcdc.object_.surface.Surface` objects, followed by ``&``
    (intersection), ``|`` (union), and ``~`` (complement). During compilation,
    the expression is converted to reverse Polish notation for evaluation by
    the geometry kernels.

    The supported region types and components are:

    - ``"all"`` uses ``A=None`` and ``B=None``.
    - ``"halfspace"`` uses a :class:`~mcdc.object_.surface.Surface` for ``A``
      and ``-1`` or ``1`` for ``B``.
    - ``"intersection"`` and ``"union"`` use a :class:`Region` for both
      ``A`` and ``B``.
    - ``"complement"`` uses a :class:`Region` for ``A`` and ``B=None``.
    """

    type: RegionType
    A: Surface | Region | NoneType
    B: Region | int | NoneType

    def __init__(
        self,
        type_: RegionType,
        A: Surface | Region | NoneType,
        B: Region | int | NoneType,
    ) -> None:
        self.type = type_
        self.A = A
        self.B = B

    @classmethod
    def make_halfspace(cls, surface: Surface, sense: Literal[-1, 1]) -> Region:
        """Create the positive or negative half-space of a surface.

        Parameters
        ----------
        surface : Surface
            Bounding surface.
        sense : int
            Positive for the positive half-space and negative for the negative
            half-space.

        Returns
        -------
        Region
            Half-space region used to build a cell expression.
        """
        region = Region("halfspace", surface, sense)
        return region

    def __and__(self, other: Region) -> Region:
        return Region("intersection", self, other)

    def __or__(self, other: Region) -> Region:
        return Region("union", self, other)

    def __invert__(self) -> Region:
        return Region("complement", self, None)

    def __repr__(self) -> str:
        return f"{str.capitalize(self.type)} Region"


# ======================================================================================
# Cell
# ======================================================================================


class Cell(MCDCObject):
    """Material- or universe-filled regions of the simulation geometry.

    Parameters
    ----------
    region : Region, optional
        Boolean region expression. If omitted, the cell covers all space.
    fill : Material, Universe, Lattice, or None, optional
        Material or nested geometry placed in the cell. ``None`` creates a void
        cell.
    name : str, optional
        User-facing name. An automatic name is assigned during compilation when
        omitted.
    translation : sequence of 3 float, optional
        Translation, in cm, applied when entering a universe or lattice fill.
    rotation : sequence of 3 float, optional
        Rotation angles about the x, y, and z axes, in degrees, applied when
        entering a universe or lattice fill.

    Notes
    -----
    A cell region is commonly written as ``+left & -right``. Surface signs select
    half-spaces; intersections, unions, and complements may be combined freely.

    Examples
    --------
    Fill a slab between two z planes with a one-group material:

    >>> import numpy as np
    >>> import mcdc
    >>> material = mcdc.Material.multigroup(capture=np.array([1.0]))
    >>> lower = mcdc.Surface.PlaneZ(z=0.0)
    >>> upper = mcdc.Surface.PlaneZ(z=2.0)
    >>> cell = mcdc.Cell(region=+lower & -upper, fill=material)

    Create a void cell outside the slab:

    >>> void = mcdc.Cell(name="Upper void", region=+upper)

    Combine regions with a union:

    >>> left_sphere = mcdc.Surface.Sphere(center=[-1.0, 0.0, 0.0], radius=0.5)
    >>> right_sphere = mcdc.Surface.Sphere(center=[1.0, 0.0, 0.0], radius=0.5)
    >>> two_spheres = mcdc.Cell(
    ...     region=-left_sphere | -right_sphere,
    ...     fill=material,
    ... )

    Fill the complement of that union:

    >>> outside_spheres = mcdc.Cell(
    ...     region=~(-left_sphere | -right_sphere),
    ...     fill=material,
    ... )

    Place a reusable universe with a translation and rotation:

    >>> assembly = mcdc.Universe(name="Assembly", cells=[cell])
    >>> placed_assembly = mcdc.Cell(
    ...     fill=assembly,
    ...     translation=[5.0, 0.0, 0.0],
    ...     rotation=[0.0, 0.0, 90.0],
    ... )
    """

    # MC/DC framework metadata
    label = "cell"
    non_numba = ["region", "region_RPN", "fill"]

    name: str

    # Region definition
    region: Region  # Non-numba
    region_RPN_tokens: list[int]
    region_RPN: Boolean  # Non-numba
    surfaces: list[Surface]

    # Fill definition
    fill: Material | Universe | Lattice | NoneType  # Non-numba
    fill_type: int
    fill_ID: int
    fill_translated: bool
    fill_rotated: bool
    translation: Annotated[NDArray[float64], (3,)]
    rotation: Annotated[NDArray[float64], (3,)]

    # Attached tallies
    interaction_tallies: list[TallyInteraction]
    tracklength_tallies: list[TallyTracklength]

    def __init__(
        self,
        region: Region | NoneType = None,
        fill: Material | Universe | Lattice | NoneType = None,
        name: str = "",
        translation: Sequence[float] = [0.0, 0.0, 0.0],
        rotation: Sequence[float] = [0.0, 0.0, 0.0],
    ) -> None:
        super().__init__()

        self.name = name or "(Unnamed cell)"
        self.region = region or Region("all", None, None)
        self.fill = fill

        # Local coordinate modifier
        self.translation = np.array(translation, dtype=float)
        self.rotation = np.array(rotation, dtype=float)
        self.fill_translated = False
        self.fill_rotated = False
        if (self.translation != 0.0).any():
            self.fill_translated = True
        if (self.rotation != 0.0).any():
            self.fill_rotated = True
            # Convert ritation
            self.rotation *= PI / 180.0

        # Cell tallies
        self.interaction_tallies = []
        self.tracklength_tallies = []

    @property
    def surfaces(self) -> list[Surface]:
        """Return unique region surfaces without compiling the cell.

        Discover surfaces on each access so changes to the region are reflected.
        Compiled surfaces retain their simulation-ID ordering.
        """
        from mcdc.object_.surface import Surface

        surfaces = []
        seen = set()
        stack = [self.region]
        while stack:
            region = stack.pop()
            if region.type == "halfspace":
                surface = region.A
                if isinstance(surface, Surface) and id(surface) not in seen:
                    seen.add(id(surface))
                    surfaces.append(surface)
            elif region.type in ("intersection", "union"):
                stack.extend([region.B, region.A])
            elif region.type == "complement":
                stack.append(region.A)

        return sorted(surfaces, key=attrgetter("ID"))

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        # Compile fill if needed
        fill = self.fill
        if fill:
            fill._compile_into_simulation(simulation)

        # Numba representation of the cell fill
        if isinstance(fill, Material):
            self.fill_type = FILL_MATERIAL
            self.fill_ID = fill.ID
        elif isinstance(fill, Universe):
            self.fill_type = FILL_UNIVERSE
            self.fill_ID = fill.ID
        elif isinstance(fill, Lattice):
            self.fill_type = FILL_LATTICE
            self.fill_ID = fill.ID
        elif fill == None:
            self.fill_type = FILL_NONE
            self.fill_ID = -1

        # Set region Reversed Polished Notation (RPN)
        if self.region.type != "all":
            self.region_RPN_tokens = generate_RPN_tokens(self.region, simulation)
            self.region_RPN = generate_RPN(self.region_RPN_tokens)
        else:
            self.region_RPN_tokens = []
            self.region_RPN = Boolean(True)

        return True

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Name: {self.name}\n"
        if self.compile_ID > 0:
            text += f"  - Region RPN: {self.region_RPN}\n"
        else:
            text += f"  - {self.region}\n"
        if self.fill:
            text += f"  - Fill [{self.fill.label.title().replace('_', ' ')}]: {self.fill.name}\n"
        else:
            text += f"  - Fill [None]"
        if self.fill_translated:
            text += f"  - Translation: {self.translation}\n"
        if self.fill_rotated:
            text += f"  - Rotation: {self.rotation * 180 / PI}\n"
        text += f"  - Bounding surfaces: {[x.name for x in self.surfaces]}\n"
        if len(self.interaction_tallies) > 0:
            text += f"  - Interaction tallies: {[x.name for x in self.interaction_tallies]}\n"
        if len(self.tracklength_tallies) > 0:
            text += f"  - Tracklength tallies: {[x.name for x in self.tracklength_tallies]}\n"
        return text


def generate_RPN_tokens(region, simulation):
    """Compile a region expression into geometry-kernel RPN tokens.

    Surface objects encountered in the expression are registered with
    ``simulation`` as part of this operation.
    """
    from mcdc.object_.surface import Surface

    # The RPN tokens
    rpn_tokens = []

    # Build RPN based on recursive evaluation of the region
    stack = [region]
    while len(stack) > 0:
        token = stack.pop()

        # Resolve region token
        if isinstance(token, Region):
            A = token.A
            B = token.B

            if token.type == "halfspace" and (
                isinstance(A, Surface) and isinstance(B, int)
            ):
                surface = A
                sense = B

                # Compile and register surface
                surface._compile_into_simulation(simulation)
                rpn_tokens.append(surface.ID)

                if sense < 0:
                    rpn_tokens.append(BOOL_NOT)

            elif token.type == "intersection" and (
                isinstance(A, Region) and isinstance(B, Region)
            ):
                stack += ["&", token.A, token.B]

            elif token.type == "union" and (
                isinstance(A, Region) and isinstance(B, Region)
            ):
                stack += ["|", token.A, token.B]

            elif token.type == "complement" and (isinstance(A, Region)):
                stack += ["~", token.A]

            else:
                print_error(
                    f"Invalid RPN tokens for Region of type {token.type}: {A}, {B}"
                )

        # Register RPN token
        else:
            if token == "&":
                rpn_tokens.append(BOOL_AND)
            elif token == "|":
                rpn_tokens.append(BOOL_OR)
            elif token == "~":
                rpn_tokens.append(BOOL_NOT)
            else:
                print_error(f"Unrecognized RPN token: {token}")

    return rpn_tokens


def generate_RPN(rpn_tokens):
    """Convert region RPN tokens to a simplified SymPy Boolean expression."""
    stack = []

    for token in rpn_tokens:
        if token >= 0:
            stack.append(sympy.symbols(f"s{token}"))
        else:
            if token == BOOL_AND or token == BOOL_OR:
                item_1 = stack.pop()
                item_2 = stack.pop()
                if token == BOOL_AND:
                    stack.append(item_1 & item_2)
                else:
                    stack.append(item_1 | item_2)

            elif token == BOOL_NOT:
                item = stack.pop()
                if isinstance(item, Region):
                    item = sympy.symbols(str(item)[8:])

                stack.append(~item)

    return sympy.logic.boolalg.simplify_logic(stack[0])
