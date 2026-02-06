import numpy as np

from numpy import float64, int64
from numpy.typing import NDArray
from types import NoneType
from typing import Annotated, Iterable

####

import mcdc.object_.distribution as distribution

from mcdc.constant import PARTICLE_NEUTRON, INF, PI
from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.distribution import DistributionTabulated, DistributionPMF
from mcdc.object_.simulation import simulation
from mcdc.object_.util import move_object


def decode_particle_type(type_):
    if type_ == PARTICLE_NEUTRON:
        return "Neutron"


# ======================================================================================
# Source
# ======================================================================================


class Source(ObjectNonSingleton):
    # Annotations for Numba mode
    label: str = "source"
    #
    name: str
    # Position
    point_source: bool
    point: Annotated[NDArray[float64], (3,)]
    x: Annotated[NDArray[float64], (2,)]
    y: Annotated[NDArray[float64], (2,)]
    z: Annotated[NDArray[float64], (2,)]
    # Direction
    isotropic_direction: bool
    mono_direction: bool
    white_direction: bool
    direction: Annotated[NDArray[float64], (3,)]
    polar_cosine: Annotated[NDArray[float64], (2,)]
    azimuthal: Annotated[NDArray[float64], (2,)]
    # Energy
    mono_energetic: bool
    energy_group: int
    energy: float
    energy_group_pmf: DistributionPMF
    energy_pdf: DistributionTabulated
    # Time
    discrete_time: bool
    time: float
    time_range: Annotated[NDArray[float64], (2,)]
    #
    particle_type: int
    probability: float
    moving: bool
    N_move: int
    N_move_grid: int
    move_velocities: Annotated[NDArray[float64], ("N_move", 3)]
    move_durations: Annotated[NDArray[float64], ("N_move",)]
    move_time_grid: Annotated[NDArray[float64], ("N_move_grid",)]
    move_translations: Annotated[NDArray[float64], ("N_move_grid", 3)]

    def __init__(
        self,
        name: str = "",
        position: Iterable[float] | NoneType = None,
        x: Iterable[float] | NoneType = None,
        y: Iterable[float] | NoneType = None,
        z: Iterable[float] | NoneType = None,
        #
        direction: Iterable[float] | NoneType = None,
        white_direction: Iterable[float] | NoneType = None,
        isotropic: bool | NoneType = None,
        polar_cosine: Iterable[float] | NoneType = None,
        azimuthal: Iterable[float] | NoneType = None,
        #
        energy: float | NDArray[float64] | NoneType = None,
        energy_group: int | NDArray[int64] | NoneType = None,
        #
        time: float | Iterable[float] = 0.0,
        probability: float = 1.0,
    ):

        super().__init__()

        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.ID}"

        # ==============================================================================
        # Default attributes
        #   Point source at origin, isotropic, mono-energetic at 1 MeV or at group 0,
        #   time = 0, neutron
        # ==============================================================================

        # Position
        self.point_source = True
        self.point = np.zeros(3)
        self.x = np.array([0.0, 0.0])
        self.y = np.array([0.0, 0.0])
        self.z = np.array([0.0, 0.0])

        # Direction
        self.isotropic_direction = True
        self.mono_direction = False
        self.white_direction = False
        self.direction = np.array([0.0, 0.0, 0.0])
        self.polar_cosine = np.array([-1.0, 1.0])
        self.azimuthal = np.array([0.0, 2.0 * PI])

        # Energy
        self.mono_energetic = True
        self.energy_group = 0
        self.energy = 1.0e6
        self.energy_group_pmf = DistributionPMF(np.array([0.0]), np.array([1.0]))
        self.energy_pdf = DistributionTabulated(
            np.array([1.0e6 - 1.0, 1.0e6 + 1.0]), np.array([1.0, 1.0])
        )

        # Time
        self.discrete_time = True
        self.time = 0.0
        self.time_range = np.array([0.0, 0.0])

        # Particle type
        self.particle_type = PARTICLE_NEUTRON

        # Probability
        self.probability = probability

        # ==============================================================================
        # Assignment
        # ==============================================================================

        # Position
        if position is not None:
            self.point = np.array(position)
        else:
            self.point_source = False
            if x is not None:
                self.x = np.array(x)
            if y is not None:
                self.y = np.array(y)
            if z is not None:
                self.z = np.array(z)

        # Direction
        if isotropic is not None:
            pass
        elif direction is not None:
            self.isotropic_direction = False
            self.direction = np.array(direction)
            if polar_cosine is not None or azimuthal is not None:
                self.mono_direction = False
                if polar_cosine is not None:
                    self.polar_cosine = np.array(polar_cosine)
                if azimuthal is not None:
                    self.azimuthal = np.array(azimuthal)
            else:
                self.mono_direction = True
        elif white_direction is not None:
            self.isotropic_direction = False
            self.white_direction = True
            self.direction = np.array(white_direction)
        # Normalize direction
        self.direction /= np.linalg.norm(self.direction)

        # Energy
        if energy_group is not None:
            if type(energy_group) == int:
                self.energy_group = energy_group
            else:
                self.mono_energetic = False
                self.energy_group_pmf = DistributionPMF(
                    energy_group[0], energy_group[1]
                )
        elif energy is not None:
            if type(energy) == float:
                self.energy = energy
            else:
                self.mono_energetic = False
                self.energy_pdf = DistributionTabulated(energy[0], energy[1])

        # Time
        if type(time) == float:
            self.time = time
        else:
            self.discrete_time = False
            self.time_range = np.array(time)

        # Moving source parameters
        self.moving = False
        self.N_move = 1
        self.N_move_grid = 2
        self.move_velocities = np.zeros((1, 3))
        self.move_durations = np.array([INF])
        self.move_time_grid = np.array([0.0, INF])
        self.move_translations = np.zeros((2, 3))

    def __repr__(self):
        text = "\n"
        text += f"Source\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Particle: {decode_particle_type(self.particle_type)}\n"
        text += f"  - Probability: {self.probability * 100}%\n"
        if self.point_source:
            text += f"  - Position [x, y, z]: {self.point} cm\n"
        else:
            text += f"  - Position\n"
            text += f"    - x: {self.x} cm\n"
            text += f"    - y: {self.y} cm\n"
            text += f"    - z: {self.z} cm\n"
        if self.isotropic_direction:
            text += f"  - Direction: Isotropic\n"
        elif self.mono_direction:
            text += f"  - Direction [ux, uy, yz]: {self.direction}\n"
        elif self.white_direction:
            text += f"  - Isotropic halfspace: {self.direction}\n"
        if simulation.materials[0].label == "multigroup_material":
            if self.mono_energetic:
                text += f"  - Energy group: {self.energy_group} \n"
            else:
                text += f"  - Energy group: {distribution.decode_type(self.energy_group_pmf.type)} [ID: {self.energy_group_pmf.ID}]\n"
        else:
            if self.mono_energetic:
                text += f"  - Energy: {self.energy} eV\n"
            else:
                text += f"  - Energy: {distribution.decode_type(self.energy_pdf)} [ID: {self.energy_pdf.ID}]\n"
        if self.discrete_time:
            text += f"  - Time: {self.time} s\n"
        else:
            text += f"  - Time: {self.time_range} s\n"

        return text

    # ==================================================================================
    # Source moving
    # ==================================================================================

    def move(self, velocities, durations):
        """
        Define piecewise-constant motion for the source.

        Appends a final static segment (zero velocity, infinite duration) so that
        the motion covers the whole simulation time.

        Parameters
        ----------
        velocities : array_like, shape (N, 3) or list
            Per-segment velocity vectors [cm/s].
        durations : array_like, shape (N,) or list
            Per-segment durations [s].

        Notes
        -----
        - Internally converts lists to arrays and constructs
          ``move_time_grid`` and cumulative ``move_translations``.
        - Sets ``moving=True`` and ``N_move = len(durations) + 1``.

        Examples
        --------
        >>> src = mcdc.Source(z=[-0.1, 0.1], isotropic=True, energy=0, time=[0.0, 1.0])
        >>> src.move(velocities=[[0,0,1.0]], durations=[0.5]) # 0.5 s upward, then static
        >>> s.N_move
        2
        """
        move_object(self, velocities, durations)
