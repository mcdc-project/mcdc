from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64
from mcdc.object_.base import ObjectBase, ObjectSingleton
from mcdc.object_.mesh import MeshBase
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
# Weight Windows
# ======================================================================================


class WeightWindows(ObjectSingleton):
    label: str = "weight_windows"

    active: bool
    mesh: MeshBase
    Nx: int
    Ny: int
    Nz: int

    # flattened array of ww
    weight_windows: NDArray[float64]

    def __init__(self):
        self.active = False

    def __call__(self, mesh, weight_windows):
        # get mesh size
        match mesh.label:
            case "uniform_mesh":
                nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
            case "structured_mesh":
                nx, ny, nz = mesh.x.shape[0], mesh.y.shape[0], mesh.z.shape[0]
            case _:
                print_error(
                    f"{type(mesh).__name__} is not supported for weight windows"
                )
        
        mesh_shape = (nx, ny, nz)
        ww_shape = weight_windows.shape
        expected_shape = (*mesh_shape, 3)
        if ww_shape != expected_shape:
            print_error(
                f"Weight window array has shape {ww_shape}, but expected {expected_shape}"
            )

        self.active = True
        self.mesh = mesh
        self.Nx = nx
        self.Ny = ny
        self.Nz = nz
        self.weight_windows = weight_windows.reshape(-1)


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
