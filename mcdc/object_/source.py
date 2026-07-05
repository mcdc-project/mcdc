import numpy as np

from numpy import float64, int64
from numpy.typing import NDArray
from types import NoneType
from typing import Annotated, Iterable

####

import mcdc.object_.distribution as distribution

from mcdc.constant import (
    INTERPOLATION_LINEAR,
    PARTICLE_NEUTRON,
    PARTICLE_ELECTRON,
    PARTICLE_PROTON,
    INF,
    PI,
)
from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.distribution import DistributionTabulated, DistributionPMF
from mcdc.object_.simulation import simulation
from mcdc.object_.util import move_object
from mcdc.print_ import print_error


def decode_particle_type(type_):
    if type_ == PARTICLE_NEUTRON:
        return "Neutron"
    elif type_ == PARTICLE_ELECTRON:
        return "Electron"
    elif type_ == PARTICLE_PROTON:
        return "Proton"


# ======================================================================================
# Source
# ======================================================================================


class Source(ObjectNonSingleton):
    """
    Define a particle source.

    A source specifies the initial position, direction, energy, time, particle
    type, and relative sampling probability for emitted particles.

    Parameters
    ----------
    name : str, optional
        User label. If omitted, a default name is generated from the source ID.
    position : array_like of float, optional
        Point-source position ``[x, y, z]`` in cm. If provided, the source is
        treated as a point source.
    x, y, z : array_like of float, optional
        Spatial bounds of a box source in cm, given as ``[min, max]`` for each
        coordinate. These are used when ``position`` is not provided.
    direction : array_like of float, optional
        Source direction vector ``[ux, uy, uz]``. The vector is normalized
        internally. If provided without angular bounds, the source is
        mono-directional.

        When ``polar_cosine`` and/or ``azimuthal`` are specified, this vector
        defines the reference (polar) axis about which directions are sampled.
    white_direction : array_like of float, optional
        Outward normal direction for a white boundary source. The vector is
        normalized internally.
    isotropic : bool, optional
        If True, emit particles isotropically.
    polar_cosine : array_like of float, optional
        Bounds for the sampled polar cosine,
        ``[mu_min, mu_max]``, measured with respect to ``direction``.
        Defaults to ``[-1.0, 1.0]``.
    azimuthal : array_like of float, optional
        Bounds for the sampled azimuthal angle,
        ``[azi_min, azi_max]`` in radians, measured about ``direction``.
        Defaults to ``[0.0, 2π]``.
    energy : float or ndarray, optional
        Source energy in eV. A float defines a mono-energetic source. An array
        defines a tabulated energy distribution. Defaults to a mono-energetic
        source at **1 MeV**.

    energy_group : int or ndarray, optional
        Source energy group. An integer defines a mono-group source. An array
        defines a discrete group probability mass function. In multigroup
        simulations, the default is **group 0**.
    time : float or array_like of float, optional
        Emission time in seconds. A float defines a discrete emission time.
        A two-entry array-like value defines a time interval
        ``[t_min, t_max]``. Defaults to ``0.0``.
    particle_type : {"neutron", "electron", "proton"}, optional
        Type of emitted particle. Defaults to ``"neutron"``.
    probability : float, optional
        Relative source probability weight. Defaults to ``1.0``.

    Notes
    -----
    If ``position`` is provided, ``x``, ``y``, and ``z`` are ignored.

    When ``position`` is not provided, the source is treated as a box source.
    Any unspecified coordinate range defaults to ``[0.0, 0.0]`` cm. For
    example, if only ``z=[-1.0, 1.0]`` is specified, then the source occupies
    ``x=[0.0, 0.0]``, ``y=[0.0, 0.0]``, and ``z=[-1.0, 1.0]``.

    Direction options are interpreted in the following order:

    - if ``isotropic=True``, the source is isotropic;
    - else if ``direction`` is provided, the source uses that direction;
    - else if ``white_direction`` is provided, the source is a white boundary
      source;
    - otherwise, the default direction behavior is used.

    ``energy_group`` takes precedence over ``energy`` when both are provided.

    Examples
    --------
    Point source at the origin emitting mono-energetic neutrons isotropically:

    >>> src = mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True)

    Uniform box source distributed along z:

    >>> src = mcdc.Source(
    ...     z=[-1.0, 1.0],
    ...     isotropic=True,
    ...     energy=1.0e6,
    ... )

    The unspecified x and y ranges default to ``[0.0, 0.0]`` cm.

    Rectangular volume source:

    >>> src = mcdc.Source(
    ...     x=[-1.0, 1.0],
    ...     y=[-2.0, 2.0],
    ...     z=[0.0, 5.0],
    ...     isotropic=True,
    ... )

    Mono-directional source:

    >>> src = mcdc.Source(
    ...     position=[0.0, 0.0, 0.0],
    ...     direction=[0.0, 0.0, 1.0],
    ... )

    Directional source with angular spread:

    >>> src = mcdc.Source(
    ...     direction=[0.0, 0.0, 1.0],
    ...     polar_cosine=[0.8, 1.0],
    ...     azimuthal=[0.0, np.pi / 2],
    ... )

    Discrete energy-group source:

    >>> src = mcdc.Source(
    ...     energy_group=3,
    ... )

    Time-dependent source:

    >>> src = mcdc.Source(
    ...     time=[0.0, 1.0e-3],
    ... )
    """

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
        #
        particle_type: str = "neutron",
        #
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
        self.direction = np.array([0.0, 0.0, 1.0])
        self.polar_cosine = np.array([-1.0, 1.0])
        self.azimuthal = np.array([0.0, 2.0 * PI])

        # Energy
        self.mono_energetic = True
        self.energy_group = 0
        self.energy = 1.0e6
        self.energy_group_pmf = DistributionPMF(np.array([0.0]), np.array([1.0]))
        self.energy_pdf = DistributionTabulated(
            np.array([1.0e6 - 1.0, 1.0e6 + 1.0]),
            np.array([1.0, 1.0]),
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
        if isotropic is not None and isotropic:
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
                self.energy_pdf = DistributionTabulated(
                    np.array(energy[0]), np.array(energy[1])
                )

        # Time
        if type(time) == float:
            self.time = time
        else:
            self.discrete_time = False
            self.time_range = np.array(time)

        # Particle type
        if particle_type == "neutron":
            self.particle_type = PARTICLE_NEUTRON
        elif particle_type == "electron":
            self.particle_type = PARTICLE_ELECTRON
        elif particle_type == "proton":
            self.particle_type = PARTICLE_PROTON
        else:
            print_error(rf"Unsupported particle types: {particle_type}")

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

        The source moves through a sequence of constant-velocity segments. Each
        segment is defined by a velocity vector and its duration. After the last
        segment, a final static segment with zero velocity and infinite duration is
        appended automatically so that the source position remains well-defined for
        the remainder of the simulation.

        Parameters
        ----------
        velocities : array_like of float, shape (N, 3)
            Velocity vector ``[vx, vy, vz]`` in cm/s for each motion segment.
        durations : array_like of float, shape (N,)
            Duration of each motion segment in seconds. Must contain the same
            number of entries as ``velocities``.

        Notes
        -----
        This method

        - enables source motion by setting ``moving=True``;
        - constructs the internal time grid (``move_time_grid``);
        - computes the cumulative translation at the end of each segment
          (``move_translations``);
        - appends a final static segment with zero velocity and infinite duration.

        The resulting number of motion segments is ``len(durations) + 1``.

        Examples
        --------
        Move a source upward at 1 cm/s for 0.5 s, then keep it stationary:

        >>> src = mcdc.Source(
        ...     z=[-0.1, 0.1],
        ...     isotropic=True,
        ...     energy=1.0e6,
        ...     time=[0.0, 1.0],
        ... )
        >>> src.move(
        ...     velocities=[[0.0, 0.0, 1.0]],
        ...     durations=[0.5],
        ... )
        >>> src.N_move
        2

        Piecewise motion with two segments:

        >>> src.move(
        ...     velocities=[
        ...         [1.0, 0.0, 0.0],
        ...         [0.0, 1.0, 0.0],
        ...     ],
        ...     durations=[0.5, 1.0],
        ... )
        """
        move_object(self, velocities, durations)
