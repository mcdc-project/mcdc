from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcdc.object_.surface import Surface

####

import numpy as np
import sympy

from numpy import float64
from numpy.typing import NDArray
from operator import attrgetter
from types import NoneType
from typing import Annotated, Iterable
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
from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.material import MaterialBase
from mcdc.object_.simulation import simulation
from mcdc.object_.tally import TallyCell
from mcdc.object_.universe import Universe, Lattice
from mcdc.print_ import print_error

# ======================================================================================
# Region
# ======================================================================================


# Region-making helper that checks if an identical region is already created
def make_region(type_, A, B):
    for existing_region in simulation.regions:
        if (
            type_ == existing_region.type
            and A == existing_region.A
            and B == existing_region.B
        ):
            return existing_region
    return Region(type_, A, B)


class Region(ObjectNonSingleton):
    type: str
    A: Surface | Region | NoneType
    B: Region | int | NoneType

    def __init__(self, type_, A, B):
        super().__init__()

        self.type = type_
        self.A = A
        self.B = B

    @classmethod
    def make_halfspace(cls, surface, sense):
        region = make_region("halfspace", surface, sense)
        return region

    def __and__(self, other):
        return make_region("intersection", self, other)

    def __or__(self, other):
        return make_region("union", self, other)

    def __invert__(self):
        return make_region("complement", self, None)

    def __repr__(self):
        text = "Region: "
        if self.type == "halfspace":
            if self.B > 0:
                text += "+s%i" % self.A.ID
            else:
                text += "-s%i" % self.A.ID
        elif self.type == "intersection":
            text += "r%i & r%i" % (self.A.ID, self.B.ID)
        elif self.type == "union":
            text += "r%i | r%i" % (self.A.ID, self.B.ID)
        elif self.type == "complement":
            text += "~r%i" % (self.A.ID)
        elif self.type == "all":
            text += "all"

        return text


# ======================================================================================
# Cell
# ======================================================================================


class Cell(ObjectNonSingleton):
    # Annotations for Numba mode
    label: str = "cell"
    non_numba: list[str] = ["region", "fill", "region_RPN"]
    #
    name: str
    region: Region
    fill: MaterialBase | Universe | Lattice | NoneType
    fill_translated: bool
    fill_rotated: bool
    translation: Annotated[NDArray[float64], (3,)]
    rotation: Annotated[NDArray[float64], (3,)]
    region_RPN_tokens: list[int]
    region_RPN: Boolean
    surfaces: list[Surface]
    tallies: list[TallyCell]
    #
    fill_type: int
    fill_ID: int

    def __init__(
        self,
        region: Region | NoneType = None,
        fill: MaterialBase | Universe | Lattice | NoneType = None,
        name: str = "",
        translation: Iterable[float] = [0.0, 0.0, 0.0],
        rotation: Iterable[float] = [0.0, 0.0, 0.0],
    ):
        super().__init__()

        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.ID}"

        # Set region
        if region is None:
            self.region = make_region("all", None, None)
        else:
            self.region = region

        # Set fill
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

        # Set region Reversed Polished Notation (RPN)
        if self.region.type != "all":
            self.region_RPN_tokens = generate_RPN_tokens(self.region)
            self.region_RPN = generate_RPN(self.region_RPN_tokens)
        else:
            self.region_RPN_tokens = []
            self.region_RPN = Boolean(True)

        # List surfaces
        self.surfaces = list_surfaces(self.region_RPN_tokens)

        # Cell tallies
        self.tallies = []

        # ==============================================================================
        # Numba attribute manual set up
        # ==============================================================================

        # Numba representation of the cell fill
        #   (Because polymorphic Ffill object is not supported)
        if isinstance(fill, MaterialBase):
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
        else:
            print_error(f"Unsupported cell fill: {fill}")

    def __repr__(self):
        text = "\n"
        text += f"Cell\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - {self.region}\n"
        if isinstance(self.fill, MaterialBase):
            text += f"  - Fill (material): {self.fill.name}\n"
        elif isinstance(self.fill, Lattice):
            text += f"  - Fill (lattice): {self.fill.name}\n"
        elif isinstance(self.fill, Universe):
            text += f"  - Fill (universe): {self.fill.name}\n"
        if self.fill_translated:
            text += f"  - Translation: {self.translation}\n"
        if self.fill_rotated:
            text += f"  - Rotation: {self.rotation * 180 / PI}\n"
        text += f"  - Bounding surfaces: {[x.ID for x in self.surfaces]}\n"
        if len(self.tallies) > 0:
            text += f"  - Tallies: {[x.ID for x in self.tallies]}\n"
        return text


def generate_RPN_tokens(region):
    # The RPN tokens
    rpn_tokens = []

    # Build RPN based on recursive evaluation of the region
    stack = [region]
    while len(stack) > 0:
        token = stack.pop()
        if isinstance(token, Region):
            if token.type == "halfspace":
                rpn_tokens.append(token.A.ID)
                if token.B < 0:
                    rpn_tokens.append(BOOL_NOT)
            elif token.type == "intersection":
                stack += ["&", token.A, token.B]
            elif token.type == "union":
                stack += ["|", token.A, token.B]
            elif token.type == "complement":
                stack += ["~", token.A]
        else:
            if token == "&":
                rpn_tokens.append(BOOL_AND)
            elif token == "|":
                rpn_tokens.append(BOOL_OR)
            elif token == "~":
                rpn_tokens.append(BOOL_NOT)
            else:
                print_error(f"Unrecognized token in the generating region RPN: {token}")

    return rpn_tokens


def generate_RPN(rpn_tokens):
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


def list_surfaces(rpn_tokens):
    surfaces = []

    for token in rpn_tokens:
        if token >= 0:
            surface = simulation.surfaces[token]
            if surface not in surfaces:
                surfaces.append(surface)

    return sorted(surfaces, key=attrgetter("ID"))
