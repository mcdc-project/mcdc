from typing import Iterable
import numpy as np

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import INF, MESH_STRUCTURED, MESH_UNIFORM
from mcdc.object_.base import ObjectPolymorphic
from mcdc.print_ import print_1d_array

# ======================================================================================
# Mesh base class
# ======================================================================================


class MeshBase(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "mesh"
    #
    name: str
    N_bin: int
    Nx: int
    Ny: int
    Nz: int

    def __init__(self, type_, name):
        super().__init__(type_)

        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.child_ID}"

        self.N_bin = 0

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - # of bins: {self.N_bin}\n"
        return text


def decode_type(type_):
    if type_ == MESH_UNIFORM:
        return "Uniform mesh"
    elif type_ == MESH_STRUCTURED:
        return "Structured mesh"


# ======================================================================================
# Uniform mesh
# ======================================================================================


class MeshUniform(MeshBase):
    """
    Define a uniform rectilinear mesh.

    Each axis is specified as ``(origin, width, N_bins)``.

    Parameters
    ----------
    name : str, optional
        User label.
    x : tuple of (float, float, int), optional
        ``(x0, dx, Nx)`` — origin, bin width, and number of bins along x.
    y : tuple of (float, float, int), optional
        ``(y0, dy, Ny)`` — origin, bin width, and number of bins along y.
    z : tuple of (float, float, int), optional
        ``(z0, dz, Nz)`` — origin, bin width, and number of bins along z.

    Returns
    -------
    MeshUniform
        The uniform mesh object.

    See Also
    --------
    mcdc.MeshStructured : Creates a mesh with arbitrary bin edges.
    mcdc.TallyMesh : Creates a tally on a mesh.
    """

    # Annotations for Numba mode
    label: str = "uniform_mesh"
    #
    x0: float
    dx: float
    Nx: int
    y0: float
    dy: float
    Ny: int
    z0: float
    dz: float
    Nz: int

    def __init__(
        self,
        name: str = "",
        x: tuple[float, float, int] = (-INF, 2 * INF, 1),
        y: tuple[float, float, int] = (-INF, 2 * INF, 1),
        z: tuple[float, float, int] = (-INF, 2 * INF, 1),
    ):
        type_ = MESH_UNIFORM
        super().__init__(type_, name)

        # Set the grid
        self.x0 = x[0]
        self.dx = x[1]
        self.Nx = x[2]
        self.y0 = y[0]
        self.dy = y[1]
        self.Ny = y[2]
        self.z0 = z[0]
        self.dz = z[1]
        self.Nz = z[2]

        self.N_bin = self.Nx * self.Ny * self.Nz

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Grid specification\n"
        text += f"    - (x0, dx, Nx): ({self.x0}, {self.dx}, {self.Nx}) [cm]\n"
        text += f"    - (y0, dy, Ny): ({self.y0}, {self.dy}, {self.Ny}) [cm]\n"
        text += f"    - (z0, dz, Nz): ({self.z0}, {self.dz}, {self.Nz}) [cm]\n"
        return text


# ======================================================================================
# Structured mesh
# ======================================================================================


class MeshStructured(MeshBase):
    """
    Define a structured rectilinear mesh with arbitrary bin edges.

    Parameters
    ----------
    name : str, optional
        User label.
    x : array_like of float, optional
        Bin edges along x (cm).
    y : array_like of float, optional
        Bin edges along y (cm).
    z : array_like of float, optional
        Bin edges along z (cm).

    Returns
    -------
    MeshStructured
        The structured mesh object.

    See Also
    --------
    mcdc.MeshUniform : Creates a uniform mesh.
    mcdc.TallyMesh : Creates a tally on a mesh.
    """
    # Annotations for Numba mode
    label: str = "structured_mesh"
    #
    x: NDArray[float64]
    y: NDArray[float64]
    z: NDArray[float64]

    def __init__(
        self,
        name: str = "",
        x: Iterable[float] = [-INF, INF],
        y: Iterable[float] = [-INF, INF],
        z: Iterable[float] = [-INF, INF],
    ):
        type_ = MESH_STRUCTURED
        super().__init__(type_, name)

        # Set the grid
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1

        self.N_bin = self.Nx * self.Ny * self.Nz

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Grid specification\n"
        text += f"    - x {print_1d_array(self.x)} cm\n"
        text += f"    - y {print_1d_array(self.y)} cm\n"
        text += f"    - z {print_1d_array(self.z)} cm\n"
        return text
