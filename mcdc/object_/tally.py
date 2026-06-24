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
    SCORE_CURRENT_NET,
    SCORE_ENERGY_DEPOSITION,
    SCORE_CURRENT_IN,
    SCORE_CURRENT_OUT,
    SUPPORTED_SCORES,
    SUPPORTED_SCORES_SURFACE_CROSSING,
    SUPPORTED_SCORES_TRACKLENGTH,
    SUPPORTED_SCORES_COLLISION,
    TALLY_SURFACE_CROSSING,
    TALLY_COLLISION,
    TALLY_TRACKLENGTH,
)
from mcdc.object_.mesh import MeshBase, MeshStructured, MeshUniform
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array, print_error


class Tally(ObjectPolymorphic):
    # Annotations for Numba mode
    label: str = "tally"

    # Basic properties
    name: str
    scores: list[int]

    # Non-spatial filters
    filter_direction: bool
    filter_energy: bool
    filter_time: bool
    mu: NDArray[float64]
    azi: NDArray[float64]
    polar_reference: Annotated[NDArray[float64], (3,)]
    energy: NDArray[float64]
    time: NDArray[float64]

    # Score bins
    bin: NDArray[float64]
    bin_sum: NDArray[float64]
    bin_sum_square: NDArray[float64]
    bin_shape: list[int]

    # Filter strides
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
        spatial_shape: tuple[int, ...] | NoneType = None,
    ) -> TallySurfaceCrossing | TallyTracklength | TallyCollision:
        # Determine tally estimator type and create the instance based on the provided
        # spatial filter and scores

        # Check scores
        if len(scores) == 0:
            print_error(f"Tally needs a score.")
        if not (set(scores) <= SUPPORTED_SCORES):
            print_error(f"Unsupported tally scores: {set(scores) - SUPPORTED_SCORES}")

        # Determine the tally type based on the provided scores
        if set(scores) <= SUPPORTED_SCORES_SURFACE_CROSSING:
            tally_type = TALLY_SURFACE_CROSSING
        elif set(scores) <= SUPPORTED_SCORES_TRACKLENGTH:
            tally_type = TALLY_TRACKLENGTH
        elif set(scores) <= SUPPORTED_SCORES_COLLISION:
            tally_type = TALLY_COLLISION
        else:
            print_error(
                f"Cannot mix tally scores with different estimators.\n  Surfaces crossing: {set(scores) & SUPPORTED_SCORES_SURFACE_CROSSING}\n  Tracklength: {set(scores) & SUPPORTED_SCORES_TRACKLENGTH}\n  Collision: {set(scores) & SUPPORTED_SCORES_COLLISION}"
            )
            tally_type = -1

        # Check spatial filters
        if tally_type == TALLY_SURFACE_CROSSING:
            if surface is None and cell is None:
                print_error("Surface-crossing tally needs surface or cell filter.")
            if mesh is not None:
                print_error("Surface-crossing tally does not support mesh filter.")

        if tally_type == TALLY_COLLISION:
            if surface is not None:
                print_error("Collision tally does not support surface filter")

        if tally_type == TALLY_TRACKLENGTH:
            if surface is not None:
                print_error("Tracklength tally does not support surface filter")

        # Create the instance based on the tally type
        if tally_type == TALLY_SURFACE_CROSSING:
            return object.__new__(TallySurfaceCrossing)
        elif tally_type == TALLY_TRACKLENGTH:
            return object.__new__(TallyTracklength)
        else:  # tally_type == TALLY_COLLISION:
            return object.__new__(TallyCollision)

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
        spatial_shape: tuple[int, ...] | NoneType = None,
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
            elif score == "current-net":
                self.scores.append(SCORE_CURRENT_NET)
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
    elif type_ == SCORE_CURRENT_NET:
        return "Current net" if not lower_case else "current-net"
    elif type_ == SCORE_CURRENT_IN:
        return "Current in" if not lower_case else "current-in"
    elif type_ == SCORE_CURRENT_OUT:
        return "Current out" if not lower_case else "current-out"
    elif type_ == SCORE_ENERGY_DEPOSITION:
        return "Energy deposition" if not lower_case else "energy_deposition"


# ======================================================================================
# Surface-crossing tally
# ======================================================================================


class TallySurfaceCrossing(Tally):
    # Annotations for Numba mode
    label: str = "surface_crossing_tally"
    non_numba: list[str] = ["surface, cell"]

    # Spatial filters
    surface: Surface | NoneType
    surface_filtered: bool
    surface_filter_ID: int
    cell: Cell | NoneType
    cell_filtered: bool
    cell_filter_ID: int

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

        # ==============================================================================
        # Set spatial filters
        # ==============================================================================

        self.surface = surface
        self.cell = cell

        # Default, no filter
        self.surface_filtered = False
        self.surface_filter_ID = -1
        self.cell_filtered = False
        self.cell_filter_ID = -1

        # Surface filter
        if surface is not None:
            self.surface_filtered = True
            self.surface_filter_ID = surface.ID

            # Attach to the surface
            surface.tallies.append(self)

        # Cell filter
        if cell is not None:
            self.cell_filtered = True
            self.cell_filter_ID = cell.ID

            # Attach to all bounding surfaces of the cell if surface filter is not specified
            if surface is None:
                for boundary_surface in cell.surfaces:
                    boundary_surface.tallies.append(self)

    def __repr__(self):
        text = super().__repr__()
        if isinstance(self.surface, Surface):
            text += f"  - Surface filter: {self.surface.name}\n"
        if isinstance(self.cell, Cell):
            text += f"  - Cell filter: {self.cell.name}\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text


# ======================================================================================
# Collision tally
# ======================================================================================


class TallyCollision(Tally):
    # Annotations for Numba mode
    label: str = "collision_tally"
    non_numba: list[str] = ["cell", "mesh"]

    # Spatial filters
    cell: Cell | NoneType
    cell_filtered: bool
    cell_filter_ID: int
    mesh: MeshBase | NoneType
    mesh_filtered: bool
    mesh_filter_type: int
    mesh_filter_ID: int

    # Mesh filter strides
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

        # ==============================================================================
        # Set spatial filters
        # ==============================================================================

        self.cell = cell
        self.mesh = mesh

        # Default, no filter
        self.cell_filtered = False
        self.cell_filter_ID = -1
        self.mesh_filtered = False
        self.mesh_filter_ID = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        # Cell filter
        if cell is not None:
            self.cell_filtered = True
            self.cell_filter_ID = cell.ID

            # Attach to the cell
            cell.tallies.append(self)

        # Mesh filter
        if mesh is not None:
            self.mesh_filtered = True
            self.mesh_filter_ID = mesh.ID

            # Mesh type
            if isinstance(mesh, MeshStructured):
                self.mesh_filter_type = MESH_STRUCTURED
            elif isinstance(mesh, MeshUniform):
                self.spatial_filter_subtype = MESH_UNIFORM

            # Mesh strides
            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

        # Attach to all cells if cell filter is not specified
        if cell is None:
            for cell_ in simulation.cells:
                cell_.tallies.append(self)

    def __repr__(self):
        text = super().__repr__()
        if isinstance(self.cell, Cell):
            text += f"  - Cell filter: {self.cell.name}\n"
        if isinstance(self.mesh, MeshBase):
            text += f"  - Mesh: {mesh_module.decode_type(self.mesh.type)} (ID {self.mesh.ID})\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text


# ======================================================================================
# Tracklength tally
# ======================================================================================


class TallyTracklength(Tally):
    # Annotations for Numba mode
    label: str = "tracklength_tally"
    non_numba: list[str] = ["cell", "mesh"]

    # Spatial filters
    cell: Cell | NoneType
    cell_filtered: bool
    cell_filter_ID: int
    mesh: MeshBase | NoneType
    mesh_filtered: bool
    mesh_filter_type: int
    mesh_filter_ID: int

    # Mesh filter strides
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
        # Set spatial filters
        # ==============================================================================

        self.cell = cell
        self.mesh = mesh

        # Default, no filter
        self.cell_filtered = False
        self.cell_filter_ID = -1
        self.mesh_filtered = False
        self.mesh_filter_ID = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        # Cell filter
        if cell is not None:
            self.cell_filtered = True
            self.cell_filter_ID = cell.ID

            # Attach to the cell
            cell.tallies.append(self)

        # Mesh filter
        if mesh is not None:
            self.mesh_filtered = True
            self.mesh_filter_ID = mesh.ID

            # Mesh type
            if isinstance(mesh, MeshStructured):
                self.mesh_filter_type = MESH_STRUCTURED
            elif isinstance(mesh, MeshUniform):
                self.spatial_filter_subtype = MESH_UNIFORM

            # Mesh strides
            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

        # Attach to all cells if cell filter is not specified
        if cell is None:
            for cell_ in simulation.cells:
                cell_.tallies.append(self)

    def __repr__(self):
        text = super().__repr__()
        if isinstance(self.cell, Cell):
            text += f"  - Cell filter: {self.cell.name}\n"
        if isinstance(self.mesh, MeshBase):
            text += f"  - Mesh: {mesh_module.decode_type(self.mesh.type)} (ID {self.mesh.ID})\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text
