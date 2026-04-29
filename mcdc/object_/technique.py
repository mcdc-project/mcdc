from numpy.typing import NDArray
import numpy as np
from typing import Annotated
from mcdc.constant import INF
from mcdc.object_.base import ObjectSingleton
from mcdc.object_.mesh import MeshBase, MeshUniform
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


class GlobalWeightRoulette(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "global_weight_roulette"

    active: bool
    weight_threshold: float
    weight_target: float

    def __init__(self):
        self.active = False
        self.weight_threshold = 0.0
        self.weight_target = 1.0

    def __call__(self, weight_threshold: float = 0.0, weight_target: float = 1.0):
        if weight_threshold > weight_target:
            print_error(
                "For weight roulette, weight threshold has to be smaller than the target"
            )
        self.active = True
        self.weight_threshold = weight_threshold
        self.weight_target = weight_target


# ======================================================================================
# Weight Windows
# ======================================================================================


class WeightWindows(ObjectSingleton):
    label: str = "weight_windows"

    active: bool

    # energy
    energy_bounds: NDArray[np.float64]
    Ne: int
    # space
    mesh: MeshBase
    Nx: int
    Ny: int
    Nz: int

    # arrays of ww params
    lower_weights: Annotated[NDArray[np.float64], ("Ne", "Nx", "Ny", "Nz")]
    target_weights: Annotated[NDArray[np.float64], ("Ne", "Nx", "Ny", "Nz")]
    upper_weights: Annotated[NDArray[np.float64], ("Ne", "Nx", "Ny", "Nz")]

    def __init__(self):
        self.active = False
        self.energy_bounds = np.array([0.0, 1.0])
        self.Ne = 1
        self.mesh_ID = -1  # skirt around having to create a MeshBase instance
        self.Nx, self.Ny, self.Nz = 1, 1, 1
        self.Nt = 1
        shape = (self.Ne, self.Nx, self.Ny, self.Nz)
        self.lower_weights = np.array([1.0]).reshape(*shape)
        self.target_weights = np.array([1.0]).reshape(*shape)
        self.upper_weights = np.array([1.0]).reshape(*shape)

    def __call__(self, weight_windows, mesh=None, energy=None):
        # fill in defaults
        if mesh is None:
            mesh = MeshUniform()
        if energy is None:
            # usable for both groups and max energy
            energy = np.array([-0.5, INF])

        # get mesh size
        match mesh.label:
            case "uniform_mesh":
                nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
            case "structured_mesh":
                nx, ny, nz = (
                    mesh.x.shape[0] - 1,
                    mesh.y.shape[0] - 1,
                    mesh.z.shape[0] - 1,
                )
            case _:
                print_error(
                    f"{type(mesh).__name__} is not supported for weight windows"
                )
        # validate energy as strictly increasing
        if not (np.diff(energy) > 0).all():
            print_error("Energy bounds must be strictly increasing")
        # get energy size
        if len(energy.shape) != 1:
            print_error(
                f"Invalid shape for energy; expected 1D got {len(energy.shape)}D"
            )
        ne = energy.shape[0] - 1

        # check correct shape
        mesh_shape = (nx, ny, nz)
        ww_shape = weight_windows.shape
        expected_shape = (ne, *mesh_shape, 3)
        if ww_shape != expected_shape:
            print_error(
                f"Weight window array has shape {ww_shape}, but expected {expected_shape}"
            )

        self.active = True
        self.energy_bounds = energy
        self.Ne = ne
        self.mesh = mesh
        self.Nx, self.Ny, self.Nz = mesh_shape
        shape = (self.Ne, *mesh_shape)
        self.lower_weights = weight_windows[..., 0]
        self.target_weights = weight_windows[..., 1]
        self.upper_weights = weight_windows[..., 2]

        # check weight windows are valid
        if (self.lower_weights <= 0.0).any():
            print_error(
                "Lower bound weights must be strictly positive to avoid invalid roulette behavior"
            )
        if (self.lower_weights > self.target_weights).any():
            print_error(
                "Lower bound weight can not be greater than the target weight for any weight window"
            )
        if (self.target_weights > self.upper_weights).any():
            print_error(
                "Target weight can not be greater than the upper bound weight for any weight window"
            )


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
