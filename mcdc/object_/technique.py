from typing import TYPE_CHECKING, Annotated
from numpy.typing import NDArray
import numpy as np
from mcdc.object_.base import ObjectSingleton
if TYPE_CHECKING:
    from mcdc.object_.cell import Cell
from mcdc.constant import FILL_MATERIAL
from mcdc.print_ import print_error

# ======================================================================================
# Implicit capture
# ======================================================================================


class ImplicitCapture(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "implicit_capture"
    active: bool

    def __init__(self):
        self.active = False

    def __call__(self, active: bool = True):
        self.active = active


# ======================================================================================
# ForcedCollisions
# ======================================================================================


class ForcedCollisions(ObjectSingleton):
    #Annotations for Numba mode
    label: str = "forced_collisions"
    active: bool

    cell_IDs: list[np.int64]

    def __init__(self):
        self.active = False
        self.cell_IDs = []
    
    def __call__(self, cells):
        for cell in cells:
            if cell.fill_type != FILL_MATERIAL:
                print_error(
                    f"Invalid cell fill on cell: \n{cell}\nForced collision technique is only valid on cells with material fill"
                )
            self.cell_IDs.append(cell.ID)


# ======================================================================================
# Weighted emission
# ======================================================================================


class WeightedEmission(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "weighted_emission"

    active: bool
    weight_target: float

    def __init__(self):
        self.active = False
        self.weight_target = 0.0

    def __call__(self, active: bool = True, weight_target: float = 1.0):
        self.active = active
        self.weight_target = weight_target


# ======================================================================================
# Weight roulette
# ======================================================================================


class WeightRoulette(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "weight_roulette"

    weight_threshold: float
    weight_target: float

    def __init__(self):
        self.weight_threshold = 0.0
        self.weight_target = 1.0

    def __call__(self, weight_threshold: float = 0.0, weight_target: float = 1.0):
        if weight_threshold > weight_target:
            print_error(
                "For weight roulette, weight threshold has to be smaller than the target"
            )
        self.weight_threshold = weight_threshold
        self.weight_target = weight_target


# ======================================================================================
# Population control
# ======================================================================================


class PopulationControl(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "population_control"
    active: bool

    def __init__(self):
        self.active = False

    def __call__(self, active: bool = True):
        self.active = active
