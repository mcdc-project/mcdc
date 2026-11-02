from typing import Sequence, Annotated
from pathlib import Path
import numpy as np

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import INF, MESH_STRUCTURED, MESH_UNIFORM, MESH_RECTILINEAR_UNSTRUCTURED
from mcdc.object_.base import MCDCPolymorphic
from mcdc.print_ import print_1d_array

# ======================================================================================
# Mesh base class
# ======================================================================================


class MeshBase(MCDCPolymorphic):
    """Base class for spatial meshes used by tallies and techniques."""

    # MC/DC framework metadata
    label = "mesh"
    sub_type = -1  # Polymorphic base

    name: str
    N_bin: int
    Nx: int
    Ny: int
    Nz: int

    def __init__(self, name: str) -> None:
        super().__init__()

        self.name = name or "(Unnamed mesh)"
        self.N_bin = 0

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Name: {self.name}\n"
        text += f"  - # of bins: {self.N_bin}\n"
        return text


# ======================================================================================
# Uniform mesh
# ======================================================================================


class MeshUniform(MeshBase):
    """Uniform spatial bins for simulation tallies and transport techniques.

    Parameters
    ----------
    name : str, optional
        User-facing mesh name.
    x, y, z : tuple of (float, float, int), optional
        ``(origin, spacing, number_of_bins)`` for each axis, in cm. Omitted
        axes default to one effectively unbounded bin.

    Examples
    --------
    Create 100 uniform bins along z from 0 to 10 cm:

    >>> import mcdc
    >>> mesh = mcdc.MeshUniform(z=(0.0, 0.1, 100))

    Create a two-dimensional x-y mesh:

    >>> mesh_xy = mcdc.MeshUniform(
    ...     x=(-5.0, 0.5, 20),
    ...     y=(-5.0, 0.5, 20),
    ... )

    Create a three-dimensional mesh with different axis spacings:

    >>> mesh_xyz = mcdc.MeshUniform(
    ...     x=(0.0, 1.0, 10),
    ...     y=(0.0, 2.0, 5),
    ...     z=(-3.0, 0.25, 24),
    ... )
    """

    # MC/DC framework metadata
    label = "uniform_mesh"
    sub_type = MESH_UNIFORM

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
    ) -> None:
        super().__init__(
            name,
        )

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

    def __repr__(self) -> str:
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
    """Structured spatial bins for simulation tallies and transport techniques.

    Parameters
    ----------
    name : str, optional
        User-facing mesh name.
    x, y, z : sequence of float, optional
        Strictly ordered grid boundaries in cm. Each omitted axis defaults to
        one effectively unbounded bin.

    Examples
    --------
    Create nonuniform bins along z:

    >>> import mcdc
    >>> mesh = mcdc.MeshStructured(z=[0.0, 0.5, 2.0, 10.0])

    Create a two-dimensional mesh from explicit boundaries:

    >>> mesh_xy = mcdc.MeshStructured(
    ...     x=[-2.0, -1.0, 0.0, 2.0],
    ...     y=[-3.0, 0.0, 1.0, 3.0],
    ... )

    Mix uniformly generated and explicitly listed boundaries:

    >>> import numpy as np
    >>> mesh_xyz = mcdc.MeshStructured(
    ...     x=np.linspace(-5.0, 5.0, 21),
    ...     y=[-1.0, 0.0, 1.0],
    ...     z=np.linspace(0.0, 10.0, 101),
    ... )
    """

    # MC/DC framework metadata
    label = "structured_mesh"
    sub_type = MESH_STRUCTURED

    x: NDArray[float64]
    y: NDArray[float64]
    z: NDArray[float64]

    def __init__(
        self,
        name: str = "",
        x: Sequence[float] | NDArray[float64] = np.array([-INF, INF]),
        y: Sequence[float] | NDArray[float64] = np.array([-INF, INF]),
        z: Sequence[float] | NDArray[float64] = np.array([-INF, INF]),
    ) -> None:
        super().__init__(name)

        # Set the grid
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1

        self.N_bin = self.Nx * self.Ny * self.Nz

    def __repr__(self) -> str:
        text = super().__repr__()
        text += f"  - Grid specification\n"
        text += f"    - x {print_1d_array(self.x)} cm\n"
        text += f"    - y {print_1d_array(self.y)} cm\n"
        text += f"    - z {print_1d_array(self.z)} cm\n"
        return text


class MeshUnstructured(MeshBase):

    label = "rectilinear_unstructured_mesh"
    sub_type = MESH_RECTILINEAR_UNSTRUCTURED

    N_cells: int
    N_dim: int
    x_bounds: Annotated[NDArray[float64], ("N_cells", 2)]
    y_bounds: Annotated[NDArray[float64], ("N_cells", 2)]
    z_bounds: Annotated[NDArray[float64], ("N_cells", 2)]
    x_neighbors: Annotated[NDArray[int], ("N_cells", "N_dim", 2)]
    y_neighbors: Annotated[NDArray[int], ("N_cells", "N_dim", 2)]
    z_neighbors: Annotated[NDArray[int], ("N_cells", "N_dim", 2)]

    def __init__(self):
        self.N_dim = 1
        self.N_bin = 1
        self.N_cells = self.N_bin
        self.x_bounds = np.array([[-INF, INF]])
        self.y_bounds = np.array([[-INF, INF]]) 
        self.z_bounds = np.array([[-INF, INF]])
        self.x_neighbors = np.array([[-1, -1]])
        self.y_neighbors = np.array([[-1, -1]])
        self.z_neighbors = np.array([[-1, -1]])

    @classmethod
    def from_gmsh(cls, gmsh_file: str | Path):
        return cls()