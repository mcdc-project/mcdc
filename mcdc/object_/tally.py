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

from mcdc.constant import (
    INF,
    MESH_STRUCTURED,
    MESH_UNIFORM,
    PI,
    PARTICLE_ANY,
    PARTICLE_NEUTRON,
    PARTICLE_ELECTRON,
    PARTICLE_PROTON,
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
from mcdc.object_.base import MCDCPolymorphic
from mcdc.print_ import print_1d_array, print_error


class Tally(MCDCPolymorphic):
    """Quantities measured during the simulation.

    Parameters
    ----------
    name : str, optional
        User-facing tally name.
    scores : list of str, optional
        Scores to accumulate. Track-length scores are ``"flux"``, ``"density"``,
        ``"collision"``, ``"capture"``, and ``"fission"``; surface-crossing
        scores are ``"current-net"``, ``"current-in"``, and ``"current-out"``;
        the collision score is ``"energy_deposition"``, scored in eV. Scores
        from different estimator families cannot be mixed.
    surface : Surface, optional
        Surface filter. Required for a surface-crossing tally unless ``cell`` is
        provided.
    cell : Cell, optional
        Cell filter.
    mesh : MeshBase, optional
        Spatial mesh filter for track-length or collision tallies.
    mu : sequence of float, optional
        Polar-cosine bin boundaries.
    azi : sequence of float, optional
        Azimuthal-angle bin boundaries in radians.
    polar_reference : sequence of 3 float, optional
        Reference direction for the angular filters.
    particle_type : {"neutron", "electron", "proton"}, optional
        Particle type selected by the tally. If omitted, the tally accepts any
        transported particle type.
    energy : sequence of float or "all", optional
        Physical energy-bin boundaries in eV. In standard neutron multigroup
        transport, boundaries instead use the group-coordinate energy, and
        ``"all"`` creates one tally bin per energy group during simulation
        compilation.
    time : sequence of float, optional
        Time-bin boundaries in seconds.

    Returns
    -------
    TallySurfaceCrossing, TallyTracklength, or TallyCollision
        Concrete tally selected from ``scores``.

    Examples
    --------
    Score flux and fission on a structured mesh:

    >>> import numpy as np
    >>> import mcdc
    >>> mesh = mcdc.MeshStructured(z=np.linspace(0.0, 10.0, 101))
    >>> tally = mcdc.Tally(
    ...     name="Axial flux",
    ...     mesh=mesh,
    ...     scores=["flux", "fission"],
    ...     energy=[0.0, 1.0e6, 20.0e6],
    ... )

    Score net current crossing a surface:

    >>> boundary = mcdc.Surface.PlaneZ(z=10.0)
    >>> current = mcdc.Tally(surface=boundary, scores=["current-net"])

    Filter a track-length tally by cell, angle, and time:

    >>> material = mcdc.Material.multigroup(capture=np.array([1.0]))
    >>> lower = mcdc.Surface.PlaneZ(z=0.0)
    >>> upper = mcdc.Surface.PlaneZ(z=10.0)
    >>> cell = mcdc.Cell(region=+lower & -upper, fill=material)
    >>> filtered_flux = mcdc.Tally(
    ...     cell=cell,
    ...     scores=["flux", "capture"],
    ...     mu=np.linspace(-1.0, 1.0, 11),
    ...     azi=np.linspace(-np.pi, np.pi, 17),
    ...     time=[0.0, 1.0e-6, 2.0e-6],
    ... )

    Compile the geometry to resolve the cell's bounding surfaces:

    >>> model = mcdc.Simulation(name="Cell-current example")
    >>> model.set_model([cell])
    >>> model.compile()

    Score current entering and leaving the cell through any of its boundaries:

    >>> cell_current = mcdc.Tally(
    ...     cell=cell,
    ...     scores=["current-net", "current-in", "current-out"],
    ... )

    A cell-only surface-crossing filter scores genuine changes in cell membership.
    Crossings of surfaces inside the cell that do not enter or leave the cell are
    not scored.

    Restrict the cell current to one particular boundary surface:

    >>> upper_surface_current = mcdc.Tally(
    ...     surface=upper,
    ...     cell=cell,
    ...     scores=["current-net", "current-in", "current-out"],
    ... )
    >>> model.set_tallies([cell_current, upper_surface_current])

    With both filters, the surface selects where crossings are scored, while the
    cell determines whether each crossing is incoming or outgoing.

    Score energy deposition with both cell and mesh filters:

    >>> deposition = mcdc.Tally(
    ...     cell=cell,
    ...     mesh=mesh,
    ...     scores=["energy_deposition"],
    ... )

    Use one energy bin per group in standard neutron multigroup transport:

    >>> multigroup_flux = mcdc.Tally(
    ...     cell=cell,
    ...     scores=["flux"],
    ...     energy="all",
    ... )
    """

    # MC/DC framework metadata
    label = "tally"
    sub_type = -1  # Polymorphic base
    non_numba = ["_energy_all"]

    # Basic properties
    name: str
    scores: list[int]

    # Non-spatial filters
    particle_type: int
    filter_direction: bool
    filter_energy: bool
    filter_time: bool
    mu: NDArray[float64]
    azi: NDArray[float64]
    polar_reference: Annotated[NDArray[float64], (3,)]
    energy: NDArray[float64]
    time: NDArray[float64]
    _energy_all: bool  # Non-numba

    # Score bins
    bin: NDArray[float64]
    # Running mean and sum of squared deviations from the running mean.
    bin_mean: NDArray[float64]
    # Finalization replaces the squared-deviation sum with the standard error.
    bin_sum_squared_deviations: NDArray[float64]
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
        particle_type: str | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NDArray[float64] | NoneType = None,
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
        particle_type: str | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
        spatial_shape: tuple[int, ...] | NoneType = None,
    ) -> None:
        super().__init__()

        # Set name
        self.name = name or "(Unnamed tally)"

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

        # Particle filter
        if particle_type is None:
            self.particle_type = PARTICLE_ANY
        elif particle_type == "neutron":
            self.particle_type = PARTICLE_NEUTRON
        elif particle_type == "electron":
            self.particle_type = PARTICLE_ELECTRON
        elif particle_type == "proton":
            self.particle_type = PARTICLE_PROTON
        else:
            print_error(f"Unsupported tally particle type: {particle_type}")

        # Phase-space filters
        self.mu = np.array([-1.0, 1.0])
        self.azi = np.array([-PI, PI])
        self.polar_reference = np.array([0.0, 0.0, 1.0])
        self.energy = np.array([-1.0, INF])
        self.time = np.array([0.0, INF])
        self.filter_direction = False
        self.filter_energy = False
        self.filter_time = False
        self._energy_all = False
        if mu is not None:
            self.mu = np.array(mu)
            self.filter_direction = True
        if azi is not None:
            self.azi = np.array(azi)
            self.filter_direction = True
        if polar_reference is not None:
            polar_reference_arr = np.array(polar_reference)
            self.polar_reference = polar_reference_arr / np.linalg.norm(
                polar_reference_arr
            )
        if energy is not None:
            if isinstance(energy, str):
                if energy != "all":
                    print_error(f"Unsupported tally energy filter: {energy}")
                self._energy_all = True
                self.energy = np.array([0.0])  # Compilation placeholder
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

    def _set_bin_shape_and_strides(self, shape: tuple):
        # Set bins
        self.bin_shape = list(shape)

        # Set strides
        self.stride_time = reduce(operator.mul, shape[4:])
        self.stride_energy = reduce(operator.mul, shape[3:])
        self.stride_azi = reduce(operator.mul, shape[2:])
        self.stride_mu = reduce(operator.mul, shape[1:])

    def _use_census_based_tally(self, frequency: int, simulation):
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
        text += f"  - Scores: {', '.join(decode_score_type(x) for x in self.scores)}\n"
        particle_name = {
            PARTICLE_ANY: "Any",
            PARTICLE_NEUTRON: "Neutron",
            PARTICLE_ELECTRON: "Electron",
            PARTICLE_PROTON: "Proton",
        }.get(self.particle_type, "Unspecified")
        text += f"  - Particle: {particle_name}\n"
        if self.filter_time or self.filter_energy or self.filter_direction:
            text += f"  - Phase-space filters\n"
        if self.filter_time:
            text += f"    - Time {print_1d_array(self.time)} s\n"
        if self.filter_energy:
            if self._energy_all:
                text += f"    - Energy: All multigroup energy groups\n"
            else:
                text += f"    - Energy {print_1d_array(self.energy)} eV\n"
        if self.filter_direction:
            text += f"    - Direction\n"
            text += f"    -   Polar reference: {self.polar_reference}\n"
            text += f"    -   Polar cosine {print_1d_array(self.mu)}\n"
            text += f"    -   Azimuthal angle {print_1d_array(self.azi)}\n"
        return text

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        return True

    def _resolve_energy_filter(self, simulation) -> None:
        """Resolve energy filters that require the complete material model."""
        if self._energy_all:
            if simulation.technique.neutron_multigroup.hybrid:
                print_error(
                    'The energy="all" filter requires standard neutron multigroup '
                    "transport."
                )
            G = simulation.materials[0].neutron_multigroup.G
            self.energy = np.linspace(0, G, G + 1) - 0.5
            shape = list(self.bin_shape)
            shape[2] = G
            self._set_bin_shape_and_strides(tuple(shape))

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - Name: {self.name}\n"
        return text


def decode_score_type(type_, lower_case=False):
    """Return the display or input name for a packed tally-score code."""

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
    else:
        print_error(f"Unknown tally score code: {type_}")
        return "Unknown score"


# ======================================================================================
# Surface-crossing tally
# ======================================================================================


class TallySurfaceCrossing(Tally):
    """Surface-crossing current tally.

    Instances are normally created through :class:`Tally`, which selects this
    estimator for current scores.
    """

    # MC/DC framework metadata
    label = "surface_crossing_tally"
    sub_type = TALLY_SURFACE_CROSSING
    non_numba = ["surface", "cell"]

    surface: Surface | NoneType  # Non-numba
    surface_filtered: bool
    surface_filter_ID: int

    cell: Cell | NoneType  # Non-numba
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
        particle_type: str | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
    ) -> None:
        super().__init__(
            name,
            scores,
            mu=mu,
            azi=azi,
            polar_reference=polar_reference,
            particle_type=particle_type,
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

        # Set surface filter
        if surface:
            self.surface_filtered = True

            # Attach to surface
            surface.surface_crossing_tallies.append(self)

        # Set cell filter
        if cell:
            self.cell_filtered = True

            # Attach to all bounding surfaces if surface filter is not specified
            if not self.surface_filtered:
                for boundary_surface in cell.surfaces:
                    boundary_surface.surface_crossing_tallies.append(self)

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        # Set surface ID
        surface = self.surface
        if surface:
            surface._compile_into_simulation(simulation)
            self.surface_filter_ID = surface.ID

        # Set cell ID
        cell = self.cell
        if cell:
            cell._compile_into_simulation(simulation)
            self.cell_filter_ID = cell.ID

        return True

    def __repr__(self) -> str:
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
    """Collision-estimator tally.

    Instances are normally created through :class:`Tally`, which selects this
    estimator for the ``"energy_deposition"`` score.
    """

    # MC/DC framework metadata
    label = "collision_tally"
    sub_type = TALLY_COLLISION
    non_numba = ["cell", "mesh"]

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
        particle_type: str | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
    ) -> None:
        spatial_shape = None
        if mesh is not None:
            spatial_shape = (mesh.Nx, mesh.Ny, mesh.Nz)

        super().__init__(
            name,
            scores,
            mu=mu,
            azi=azi,
            polar_reference=polar_reference,
            particle_type=particle_type,
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
        self.mesh_filter_type = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        # Set cell filter
        if cell:
            self.cell_filtered = True

            # Attach to cell
            cell.collision_tallies.append(self)

        # Mesh filter
        if mesh:
            self.mesh_filtered = True
            self.mesh_filter_type = mesh.sub_type

            # Mesh strides
            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        # Set cell ID
        cell = self.cell
        if cell:
            cell._compile_into_simulation(simulation)
            self.cell_filter_ID = cell.ID

        # Set mesh ID
        mesh = self.mesh
        if mesh:
            mesh._compile_into_simulation(simulation)
            self.mesh_filter_ID = mesh.ID

        # Attach to all cells if cell filter is not specified
        if not self.cell_filtered:
            for cell in simulation.cells:
                cell.collision_tallies.append(self)

        return True

    def __repr__(self) -> str:
        text = super().__repr__()
        if self.cell:
            text += f"  - Cell filter: {self.cell.name}\n"
        if self.mesh:
            text += f"  - Mesh: {self.mesh.name}\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text


# ======================================================================================
# Tracklength tally
# ======================================================================================


class TallyTracklength(Tally):
    """Track-length estimator tally.

    Instances are normally created through :class:`Tally`, which selects this
    estimator for flux, density, reaction-rate, and collision scores.
    """

    # MC/DC framework metadata
    label = "tracklength_tally"
    sub_type = TALLY_TRACKLENGTH
    non_numba = ["cell", "mesh"]

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
        particle_type: str | NoneType = None,
        energy: Sequence[float] | str | NoneType = None,
        time: Sequence[float] | NoneType = None,
    ) -> None:
        spatial_shape = None
        if mesh is not None:
            spatial_shape = (mesh.Nx, mesh.Ny, mesh.Nz)

        super().__init__(
            name,
            scores,
            mu=mu,
            azi=azi,
            polar_reference=polar_reference,
            particle_type=particle_type,
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
        self.mesh_filter_type = -1
        self.mesh_stride_z = -1
        self.mesh_stride_y = -1
        self.mesh_stride_x = -1

        # Set cell filter
        if cell:
            self.cell_filtered = True

            # Attach to cell
            cell.tracklength_tallies.append(self)

        # Mesh filter
        if mesh:
            self.mesh_filtered = True
            self.mesh_filter_type = mesh.sub_type

            # Mesh strides
            N_score = len(self.scores)
            self.mesh_stride_z = N_score
            self.mesh_stride_y = N_score * mesh.Nz
            self.mesh_stride_x = N_score * mesh.Nz * mesh.Ny

    def _compile_into_simulation(self, simulation) -> bool:
        # Already compiled?
        if not super()._compile_into_simulation(simulation):
            return False

        # Set cell ID
        cell = self.cell
        if cell:
            cell._compile_into_simulation(simulation)
            self.cell_filter_ID = cell.ID

        # Set mesh ID
        mesh = self.mesh
        if mesh:
            mesh._compile_into_simulation(simulation)
            self.mesh_filter_ID = mesh.ID

        # Attach to all cells if cell filter is not specified
        if not self.cell_filtered:
            for cell in simulation.cells:
                cell.tracklength_tallies.append(self)

        return True

    def __repr__(self) -> str:
        text = super().__repr__()

        if self.cell:
            text += f"  - Cell filter: {self.cell.name}\n"
        if self.mesh:
            text += f"  - Mesh: {self.mesh.name}\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape [mu, azi, energy, time, score]: {self.bin_shape} \n"
        return text
