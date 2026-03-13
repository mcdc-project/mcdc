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
from typing import Annotated, Iterable
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
    SCORE_EDEP,
    SPATIAL_FILTER_CELL,
    SPATIAL_FILTER_MESH,
    SPATIAL_FILTER_NONE,
    TALLY_SURFACE,
    TALLY_COLLISION,
    TALLY_TRACKLENGTH,
)
from mcdc.object_.mesh import MeshBase, MeshStructured, MeshUniform
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array, print_error


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
        mu: Iterable[float] | NoneType = None,
        azi: Iterable[float] | NoneType = None,
        polar_reference: Iterable[float] | NoneType = None,
        energy: Iterable[float] | str | NoneType = None,
        time: Iterable[float] | NoneType = None,
    ) -> TallySurface | TallyTracklength | TallyCollision:
        # Determine type and create the tally self based on the provided
        # spatial filters and scores
        if surface is not None:
            return super().__new__(TallySurface)
        if "edep" in scores:
            if len(scores) > 1:
                print_error(
                    "Score 'edep' cannot be grouped with other scores yet. "
                    "Please request it in a separate tally."
                )
            else:
                return super().__new__(TallyCollision)
        else:
            return super().__new__(TallyTracklength)

    def __init__(
        self,
        name: str = "",
        scores: list[str] = ["flux"],
        surface: Surface | NoneType = None,
        cell: Cell | NoneType = None,
        mesh: MeshBase | NoneType = None,
        mu: Iterable[float] | NoneType = None,
        azi: Iterable[float] | NoneType = None,
        polar_reference: Iterable[float] | NoneType = None,
        energy: Iterable[float] | str | NoneType = None,
        time: Iterable[float] | NoneType = None,
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
            elif score == "edep":
                self.scores.append(SCORE_EDEP)
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
    elif type_ == TALLY_SURFACE:
        return "Surface tally"
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
    elif type_ == SCORE_EDEP:
        return "Energy deposition" if not lower_case else "edep"


# ======================================================================================
# Surface tally
# ======================================================================================


class TallySurface(Tally):
    # Annotations for Numba mode
    label: str = "surface_tally"
    #
    surface: Surface

    def __init__(
        self,
        surface: Surface,
        name: str = "",
        scores: list[str] = ["flux"],
        mu: Iterable[float] | NoneType = None,
        azi: Iterable[float] | NoneType = None,
        polar_reference: Iterable[float] | NoneType = None,
        energy: Iterable[float] | str | NoneType = None,
        time: Iterable[float] | NoneType = None,
    ):
        type_ = TALLY_SURFACE
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

        if SCORE_EDEP in self.scores:
            print_error(
                "Score 'edep' uses the collision estimator and is not supported "
                "for this tally type."
            )

        # Set surface and attach tally to the surface
        self.surface = surface
        surface.tallies.append(self)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Surface: {self.surface.name}\n"
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
        mu: Iterable[float] | NoneType = None,
        azi: Iterable[float] | NoneType = None,
        polar_reference: Iterable[float] | NoneType = None,
        energy: Iterable[float] | str | NoneType = None,
        time: Iterable[float] | NoneType = None,
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


class TallyCollision(Tally):
    label: str = "collision_tally"
    non_numba: list[str] = ["spatial_filter"]

    spatial_filter: MeshBase | NoneType
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
        scores: list[str] = ["edep"],
        mu: Iterable[float] | NoneType = None,
        azi: Iterable[float] | NoneType = None,
        polar_reference: Iterable[float] | NoneType = None,
        energy: Iterable[float] | str | NoneType = None,
        time: Iterable[float] | NoneType = None,
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

        if len(self.scores) != 1 or SCORE_EDEP not in self.scores:
            print_error("Collision tally currently supports only scores=['edep'].")

        self.spatial_filter = None
        self.spatial_filter_type = SPATIAL_FILTER_NONE
        self.spatial_filter_subtype = -1
        self.spatial_filter_ID = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        if cell is not None:
            print_error(
                "Collision tally with 'edep' is currently only supported "
                "with a mesh spatial filter."
            )

        if SCORE_EDEP in self.scores and mesh is None:
            print_error(
                "Score 'edep' is currently only supported with a mesh spatial filter."
            )

        if mesh is not None:
            self.spatial_filter = mesh
            self.spatial_filter_type = SPATIAL_FILTER_MESH
            if isinstance(mesh, MeshStructured):
                self.spatial_filter_subtype = MESH_STRUCTURED
            elif isinstance(mesh, MeshUniform):
                self.spatial_filter_subtype = MESH_UNIFORM
            self.spatial_filter_ID = mesh.ID

            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

    def __repr__(self):
        text = super().__repr__()
        if self.spatial_filter_type == SPATIAL_FILTER_MESH:
            text += (
                f"  - Mesh: {mesh_module.decode_type(self.spatial_filter.type)} "
                f"(ID {self.spatial_filter.ID})\n"
            )
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text
