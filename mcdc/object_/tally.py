from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcdc.object_.cell import Cell
    from mcdc.object_.surface import Surface

####

import numpy as np
import operator

from functools import reduce
from numpy import float64
from numpy.typing import NDArray
from typing import Annotated, Sequence
from types import NoneType

####

import mcdc.object_.mesh as mesh_module

from mcdc.constant import (
    INF,
    MESH_STRUCTURED,
    MESH_UNIFORM,
    PI,
    SCORE_FLUX,
    SCORE_DENSITY,
    SCORE_COLLISION,
    SCORE_CAPTURE,
    SCORE_FISSION,
    SCORE_NET_CURRENT,
    SCORE_ENERGY_DEPOSITION,
    SCORE_CURRENT_IN,
    SCORE_CURRENT_OUT,
    SPATIAL_FILTER_CELL,
    SPATIAL_FILTER_SURFACE,
    SPATIAL_FILTER_MESH,
    SPATIAL_FILTER_NONE,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    TALLY_SURFACE_CROSSING,
    TALLY_COLLISION,
    TALLY_TRACKLENGTH,
)
from mcdc.object_.mesh import MeshBase, MeshStructured, MeshUniform
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array, print_error

SURFACE_SCORES = {"net-current"}
CELL_CURRENT_SCORES = {"net-current", "current-in", "current-out"}
TRACKLENGTH_SCORES = {"flux", "density", "collision", "capture", "fission"}
COLLISION_SCORES = {"energy_deposition"}


class Tally(ObjectPolymorphic):
    """
    Define a tally.
    """

    # Annotations for Numba mode
    label: str = "tally"
    #
    name: str
    scores: list[int]
    #
    filter_direction: bool
    filter_energy: bool
    filter_time: bool
    mu: NDArray[float64]
    azi: NDArray[float64]
    polar_reference: Annotated[NDArray[float64], (3,)]
    energy: NDArray[float64]
    time: NDArray[float64]
    #
    bin: NDArray[float64]
    bin_sum: NDArray[float64]
    bin_sum_square: NDArray[float64]
    bin_shape: list[int]
    #
    stride_mu: int
    stride_azi: int
    stride_energy: int
    stride_time: int

    def __new__(
        cls,
        name: str = "",
        scores: list[str] = ["flux"],
        surface: Surface | NoneType = None,
        cell: Cell | NoneType = None,
        mesh: MeshBase | NoneType = None,
        mu: Sequence[float] | NoneType = None,
        azi: Sequence[float] | NoneType = None,
        polar_reference: Sequence[float] | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
        x: Sequence[float] | NoneType = None,
        y: Sequence[float] | NoneType = None,
        z: Sequence[float] | NoneType = None,
    ) -> TallySurfaceCrossing | TallyTracklength | TallyCollision:
        # Determine type and create the tally self based on the provided
        # spatial filters and scores

        has_current_score = any(score in CELL_CURRENT_SCORES for score in scores)

        # Surface/cell current tally
        if has_current_score:
            for score in scores:
                if score not in CELL_CURRENT_SCORES:
                    print_error(
                        "Cannot mix current scores with non-current scores "
                        f"in one tally. Current scores: {CELL_CURRENT_SCORES}."
                    )

            if surface is not None and cell is not None:
                print_error(
                    "Current tally must specify exactly one of surface or cell."
                )

            if surface is None and cell is None:
                print_error("Current scores need either a surface or a cell tally.")

            if surface is not None and not set(scores) <= SURFACE_SCORES:
                print_error(
                    "Surface tally currently supports only " "scores=['net-current']."
                )

            # Cell-filtered current tallies share the surface-crossing estimator.
            return super().__new__(TallySurfaceCrossing)

        # Unsupported score for explicit surface selector
        if surface is not None:
            for score in scores:
                print_error(
                    f"Scoring '{score}' with surface tally is not supported. "
                    f"Supported surface tally scores: {SURFACE_SCORES}."
                )

        # Collision tally
        if set(scores) <= COLLISION_SCORES:
            return super().__new__(TallyCollision)

        # Tracklength tally
        if set(scores) <= TRACKLENGTH_SCORES:
            return super().__new__(TallyTracklength)

        # Error: Unsupported score combination
        print_error(
            "Cannot mix tracklength scores with collision ones."
            f"\n  Tracklength scores: {TRACKLENGTH_SCORES}"
            f"\n  Collision scores: {COLLISION_SCORES}"
        )

    def __init__(
        self,
        name: str = "",
        scores: list[str] = ["flux"],
        surface: Surface | NoneType = None,
        cell: Cell | NoneType = None,
        mesh: MeshBase | NoneType = None,
        mu: Sequence[float] | NoneType = None,
        azi: Sequence[float] | NoneType = None,
        polar_reference: Sequence[float] | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
        x: Sequence[float] | NoneType = None,
        y: Sequence[float] | NoneType = None,
        z: Sequence[float] | NoneType = None,
        spatial_shape: tuple[int] | NoneType = None,
    ):
        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.child_ID}"

        # Set scores
        self.scores = []
        for score in scores:
            if score == "flux":
                self.scores.append(SCORE_FLUX)
            elif score == "density":
                self.scores.append(SCORE_DENSITY)
            elif score == "collision":
                self.scores.append(SCORE_COLLISION)
            elif score == "capture":
                self.scores.append(SCORE_CAPTURE)
            elif score == "fission":
                self.scores.append(SCORE_FISSION)
            elif score == "net-current":
                self.scores.append(SCORE_NET_CURRENT)
            elif score == "current-in":
                self.scores.append(SCORE_CURRENT_IN)
            elif score == "current-out":
                self.scores.append(SCORE_CURRENT_OUT)
            elif score == "energy_deposition":
                self.scores.append(SCORE_ENERGY_DEPOSITION)
            else:
                print_error(f"Unknown tally score: {score}")

        # Phase-space filters
        self.mu = np.array([-1.0, 1.0])
        self.azi = np.array([-PI, PI])
        self.polar_reference = np.array([0.0, 0.0, 1.0])
        self.energy = np.array([-1.0, INF])
        self.time = np.array([0.0, INF])
        self.filter_direction = False
        self.filter_energy = False
        self.filter_time = False
        if mu is not None:
            self.mu = np.array(mu)
            self.filter_direction = True
        if azi is not None:
            self.azi = np.array(azi)
            self.filter_direction = True
        if polar_reference is not None:
            polar_reference = np.array(polar_reference)
            self.polar_reference /= polar_reference / np.linalg.norm(polar_reference)
        if energy is not None:
            if type(energy) == str and energy == "all_groups":
                G = simulation.materials[0].G
                self.energy = np.linspace(0, G, G + 1) - 0.5
            else:
                self.energy = np.array(energy)
            self.filter_energy = True
        if time is not None:
            self.time = np.array(time)
            self.filter_time = True

        # Determine bin shape
        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_time = len(self.time) - 1
        N_score = len(self.scores)
        #
        if spatial_shape is None:
            shape = (N_mu, N_azi, N_energy, N_time, N_score)
        else:
            shape = (N_mu, N_azi, N_energy, N_time) + spatial_shape + (N_score,)

        # Set bins and strides
        self._set_bin_shape_and_strides(shape)

    def _set_bin_shape_and_strides(self, shape):
        # Set bins
        self.bin_shape = list(shape)

        # Set strides
        self.stride_time = reduce(operator.mul, shape[4:])
        self.stride_energy = reduce(operator.mul, shape[3:])
        self.stride_azi = reduce(operator.mul, shape[2:])
        self.stride_mu = reduce(operator.mul, shape[1:])

    def _use_census_based_tally(self, frequency):
        first_census = simulation.settings.census_time[0]
        self.time = np.linspace(0.0, first_census, frequency + 1)

        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_score = len(self.scores)

        spatial_shape = None
        if len(self.bin_shape) > 5:
            spatial_shape = tuple(self.bin_shape[4:-1])

        if spatial_shape is None:
            shape = (N_mu, N_azi, N_energy, frequency, N_score)
        else:
            shape = (N_mu, N_azi, N_energy, frequency) + spatial_shape + (N_score,)

        self._set_bin_shape_and_strides(shape)

    def _phasespace_filter_text(self):
        text = ""
        text += f"  - Scores: {[decode_score_type(x) for x in self.scores]}\n"
        if self.filter_time or self.filter_energy or self.filter_direction:
            text += f"  - Phase-space filters\n"
        if self.filter_time:
            text += f"    - Time {print_1d_array(self.time)} s\n"
        if self.filter_energy:
            text += f"    - Energy {print_1d_array(self.energy)} eV\n"
        if self.filter_direction:
            text += f"    - Direction\n"
            text += f"    -   Polar reference: {self.polar_reference}\n"
            text += f"    -   Polar cosine {print_1d_array(self.mu)}\n"
            text += f"    -   Azimuthal angle {print_1d_array(self.azi)}\n"
        return text

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        return text


def decode_type(type_):
    if type_ == TALLY_TRACKLENGTH:
        return "Tracklength tally"
    elif type_ == TALLY_SURFACE_CROSSING:
        return "Surface crossing tally"
    elif type_ == TALLY_COLLISION:
        return "Collision tally"


def decode_score_type(type_, lower_case=False):
    if type_ == SCORE_FLUX:
        return "Flux" if not lower_case else "flux"
    elif type_ == SCORE_DENSITY:
        return "Density" if not lower_case else "density"
    elif type_ == SCORE_COLLISION:
        return "Collision" if not lower_case else "collision"
    elif type_ == SCORE_CAPTURE:
        return "Capture" if not lower_case else "capture"
    elif type_ == SCORE_FISSION:
        return "Fission" if not lower_case else "fission"
    elif type_ == SCORE_NET_CURRENT:
        return "Net current" if not lower_case else "net-current"
    elif type_ == SCORE_ENERGY_DEPOSITION:
        return "Energy deposition" if not lower_case else "energy_deposition"
    elif type_ == SCORE_CURRENT_IN:
        return "Current in" if not lower_case else "current-in"
    elif type_ == SCORE_CURRENT_OUT:
        return "Current out" if not lower_case else "current-out"


# ======================================================================================
# Surface tally
# ======================================================================================


class TallySurfaceCrossing(Tally):
    # Annotations for Numba mode
    label: str = "surface_tally"
    non_numba: list[str] = ["spatial_filter"]
    #
    spatial_filter: Surface | Cell
    spatial_filter_type: int
    spatial_filter_ID: int
    surface: Surface
    filter_surface_bounds: bool
    has_x_bounds: bool
    has_y_bounds: bool
    has_z_bounds: bool
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __init__(
        self,
        surface: Surface | NoneType = None,
        cell: Cell | NoneType = None,
        name: str = "",
        scores: list[str] = ["flux"],
        mu: Sequence[float] | NoneType = None,
        azi: Sequence[float] | NoneType = None,
        polar_reference: Sequence[float] | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
        x: Sequence[float] | NoneType = None,
        y: Sequence[float] | NoneType = None,
        z: Sequence[float] | NoneType = None,
    ):
        type_ = TALLY_SURFACE_CROSSING
        super(Tally, self).__init__(type_)
        super().__init__(
            name,
            scores,
            mu=mu,
            azi=azi,
            polar_reference=polar_reference,
            energy=energy,
            time=time,
        )

        if SCORE_ENERGY_DEPOSITION in self.scores:
            print_error(
                "Score 'energy_deposition' uses the collision estimator and is not supported "
                "for this tally type."
            )

        if surface is not None and cell is not None:
            print_error("Surface tally must specify exactly one of surface or cell.")
        if surface is None and cell is None:
            print_error("Surface tally requires either a surface or a cell.")

        if surface is not None:
            self.spatial_filter = surface
            self.spatial_filter_type = SPATIAL_FILTER_SURFACE
            self.spatial_filter_ID = surface.ID
            self.surface = surface
            surface.tallies.append(self)
        else:
            self.spatial_filter = cell
            self.spatial_filter_type = SPATIAL_FILTER_CELL
            self.spatial_filter_ID = cell.ID
            self.surface = cell.surfaces[0]
            for boundary_surface in cell.surfaces:
                boundary_surface.tallies.append(self)

        # Optional bounds for axis-aligned planar surface tallies.
        self.filter_surface_bounds = False
        self.has_x_bounds = False
        self.has_y_bounds = False
        self.has_z_bounds = False
        self.x_min = -INF
        self.x_max = INF
        self.y_min = -INF
        self.y_max = INF
        self.z_min = -INF
        self.z_max = INF

        if self.spatial_filter_type == SPATIAL_FILTER_CELL and (
            x is not None or y is not None or z is not None
        ):
            print_error(
                "Cell-filtered surface current tally does not support explicit x/y/z bounds."
            )

        if x is not None:
            x = np.array(x, dtype=float)
            if len(x) != 2:
                print_error(
                    "Surface tally bound x must have exactly two values [min, max]."
                )
            self.has_x_bounds = True
            self.x_min = min(x[0], x[1])
            self.x_max = max(x[0], x[1])

        if y is not None:
            y = np.array(y, dtype=float)
            if len(y) != 2:
                print_error(
                    "Surface tally bound y must have exactly two values [min, max]."
                )
            self.has_y_bounds = True
            self.y_min = min(y[0], y[1])
            self.y_max = max(y[0], y[1])

        if z is not None:
            z = np.array(z, dtype=float)
            if len(z) != 2:
                print_error(
                    "Surface tally bound z must have exactly two values [min, max]."
                )
            self.has_z_bounds = True
            self.z_min = min(z[0], z[1])
            self.z_max = max(z[0], z[1])

        self.filter_surface_bounds = (
            self.has_x_bounds or self.has_y_bounds or self.has_z_bounds
        )

        if self.filter_surface_bounds:
            if self.surface.type not in (
                SURFACE_PLANE_X,
                SURFACE_PLANE_Y,
                SURFACE_PLANE_Z,
            ):
                print_error(
                    "Bounded surface tally currently supports only PlaneX, PlaneY, and PlaneZ surfaces."
                )
            if self.surface.type == SURFACE_PLANE_X and self.has_x_bounds:
                print_error("PlaneX surface tally bounds may only use y and/or z.")
            if self.surface.type == SURFACE_PLANE_Y and self.has_y_bounds:
                print_error("PlaneY surface tally bounds may only use x and/or z.")
            if self.surface.type == SURFACE_PLANE_Z and self.has_z_bounds:
                print_error("PlaneZ surface tally bounds may only use x and/or y.")

    def __repr__(self):
        text = super().__repr__()
        if self.spatial_filter_type == SPATIAL_FILTER_SURFACE:
            text += f"  - Surface: {self.spatial_filter.name}\n"
        elif self.spatial_filter_type == SPATIAL_FILTER_CELL:
            text += f"  - Cell filter: {self.spatial_filter.name}\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text


# ======================================================================================
# Collision tally
# ======================================================================================


class TallyCollision(Tally):
    label: str = "collision_tally"
    non_numba: list[str] = ["spatial_filter"]

    spatial_filter: Cell | MeshBase | NoneType
    spatial_filter_type: int
    spatial_filter_ID: int
    spatial_filter_subtype: int

    mesh_stride_z: int
    mesh_stride_y: int
    mesh_stride_x: int

    def __init__(
        self,
        cell: Cell | NoneType = None,
        mesh: MeshBase | NoneType = None,
        name: str = "",
        scores: list[str] = ["energy_deposition"],
        mu: Sequence[float] | NoneType = None,
        azi: Sequence[float] | NoneType = None,
        polar_reference: Sequence[float] | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
    ):
        type_ = TALLY_COLLISION
        spatial_shape = None
        if mesh is not None:
            spatial_shape = (mesh.Nx, mesh.Ny, mesh.Nz)

        super(Tally, self).__init__(type_)
        super().__init__(
            name,
            scores,
            mu=mu,
            azi=azi,
            polar_reference=polar_reference,
            energy=energy,
            time=time,
            spatial_shape=spatial_shape,
        )

        if len(self.scores) != 1 or SCORE_ENERGY_DEPOSITION not in self.scores:
            print_error(
                "Collision tally currently supports only scores=['energy_deposition']."
            )

        # Support check
        if SCORE_ENERGY_DEPOSITION in self.scores and mesh is None:
            print_error(
                "Score 'energy_deposition' is currently only supported with a mesh spatial filter."
            )

        # ==============================================================================
        # Set spatial filter
        # ==============================================================================

        # Default: no filter
        self.spatial_filter = None
        self.spatial_filter_type = SPATIAL_FILTER_NONE
        self.spatial_filter_subtype = -1
        self.spatial_filter_ID = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        # Cell filter
        if cell is not None:
            self.spatial_filter = cell
            self.spatial_filter_type = SPATIAL_FILTER_CELL
            self.spatial_filter_ID = cell.ID

            # Attach tally to the cell
            cell.tallies.append(self)

        # Mesh filter
        if mesh is not None:
            self.spatial_filter = mesh
            self.spatial_filter_type = SPATIAL_FILTER_MESH
            if isinstance(mesh, MeshStructured):
                self.spatial_filter_subtype = MESH_STRUCTURED
            elif isinstance(mesh, MeshUniform):
                self.spatial_filter_subtype = MESH_UNIFORM
            self.spatial_filter_ID = mesh.ID

            # Set the strides
            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

    def __repr__(self):
        text = super().__repr__()
        if self.spatial_filter_type == SPATIAL_FILTER_CELL:
            text += f"  - Cell: {self.spatial_filter.name}\n"
        elif self.spatial_filter_type == SPATIAL_FILTER_MESH:
            text += f"  - Mesh: {mesh_module.decode_type(self.spatial_filter.type)} (ID {self.spatial_filter.ID})\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text


# ======================================================================================
# Tracklength tally
# ======================================================================================


class TallyTracklength(Tally):
    # Annotations for Numba mode
    label: str = "tracklength_tally"
    non_numba: list[str] = ["spatial_filter"]
    #
    spatial_filter: Cell | MeshBase | NoneType
    spatial_filter_type: int
    spatial_filter_ID: int
    spatial_filter_subtype: int
    #
    mesh_stride_z: int
    mesh_stride_y: int
    mesh_stride_x: int

    def __init__(
        self,
        cell: Cell | NoneType = None,
        mesh: MeshBase | NoneType = None,
        name: str = "",
        scores: list[str] = ["flux"],
        mu: Sequence[float] | NoneType = None,
        azi: Sequence[float] | NoneType = None,
        polar_reference: Sequence[float] | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
    ):
        type_ = TALLY_TRACKLENGTH
        spatial_shape = None
        if mesh is not None:
            spatial_shape = (mesh.Nx, mesh.Ny, mesh.Nz)

        super(Tally, self).__init__(type_)
        super().__init__(
            name,
            scores,
            mu=mu,
            azi=azi,
            polar_reference=polar_reference,
            energy=energy,
            time=time,
            spatial_shape=spatial_shape,
        )

        # ==============================================================================
        # Set spatial filter
        # ==============================================================================

        # Default: no filter
        self.spatial_filter = None
        self.spatial_filter_type = SPATIAL_FILTER_NONE
        self.spatial_filter_subtype = -1
        self.spatial_filter_ID = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        # Cell filter
        if cell is not None:
            self.spatial_filter = cell
            self.spatial_filter_type = SPATIAL_FILTER_CELL
            self.spatial_filter_ID = cell.ID

            # Attach tally to the cell
            cell.tallies.append(self)

        # Mesh filter
        elif mesh is not None:
            self.spatial_filter = mesh
            self.spatial_filter_type = SPATIAL_FILTER_MESH
            if isinstance(mesh, MeshStructured):
                self.spatial_filter_subtype = MESH_STRUCTURED
            elif isinstance(mesh, MeshUniform):
                self.spatial_filter_subtype = MESH_UNIFORM
            self.spatial_filter_ID = mesh.ID

            # Set the strides
            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

    def __repr__(self):
        text = super().__repr__()
        if self.spatial_filter_type == SPATIAL_FILTER_CELL:
            text += f"  - Cell: {self.spatial_filter.name}\n"
        elif self.spatial_filter_type == SPATIAL_FILTER_MESH:
            text += f"  - Mesh: {mesh_module.decode_type(self.spatial_filter.type)} (ID {self.spatial_filter.ID})\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text
