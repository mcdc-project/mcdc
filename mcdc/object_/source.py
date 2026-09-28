import numpy as np

from numbers import Real
from numpy import float64
from numpy.typing import ArrayLike, NDArray
from types import NoneType
from typing import Annotated, Sequence

####

from mcdc.constant import (
    PARTICLE_NEUTRON,
    PARTICLE_ELECTRON,
    PARTICLE_PROTON,
    INF,
    PI,
)
from mcdc.object_.base import MCDCObject
from mcdc.object_.distribution import DistributionTabulated, DistributionPMF
from mcdc.object_.util import move_object
from mcdc.print_ import print_error

# ======================================================================================
# Source
# ======================================================================================


class Source(MCDCObject):
    """Distributions of particles introduced into the simulation.

    A source specifies the position, direction, energy, time, particle type, and
    relative sampling probability for emitted particles.

    Parameters
    ----------
    name : str, optional
        User label. If omitted, a default name is generated from the source ID.
    position : array_like of float, optional
        Point-source position ``[x, y, z]`` in cm. If provided, the source is
        treated as a point source. Cannot be supplied with ``x``, ``y``, or
        ``z``.
    x, y, z : array_like of float, optional
        Spatial bounds of a box source in cm, given as ``[min, max]`` for each
        coordinate. Cannot be supplied with ``position``.
    direction : array_like of float, optional
        Source direction vector ``[ux, uy, uz]``. The vector is normalized
        internally. If provided without angular bounds, the source is
        mono-directional. Cannot be supplied with ``isotropic=True`` or
        ``white_direction``.

        When ``polar_cosine`` and/or ``azimuthal`` are specified, this vector
        defines the reference (polar) axis about which directions are sampled.
    white_direction : array_like of float, optional
        Outward normal direction for a white boundary source. The vector is
        normalized internally. Cannot be supplied with ``isotropic=True`` or
        ``direction``.
    isotropic : bool, optional
        If True, emit particles isotropically. Cannot be supplied with
        ``direction`` or ``white_direction``.
    polar_cosine : array_like of float, optional
        Bounds for the sampled polar cosine,
        ``[mu_min, mu_max]``, measured with respect to ``direction``.
        Requires ``direction``. Defaults to ``[-1.0, 1.0]``.
    azimuthal : array_like of float, optional
        Bounds for the sampled azimuthal angle,
        ``[azi_min, azi_max]`` in radians, measured about ``direction``.
        Requires ``direction``. Defaults to ``[0.0, 2π]``.
    energy : float, array_like of float, or int, optional
        Source energy in eV. A real scalar, including a NumPy scalar, defines a
        mono-energetic source. An array-like value with shape ``(2, N)`` defines
        a tabulated distribution: the first row contains energy values and the
        second row contains their probability density in ``eV^-1``. Defaults
        to a mono-energetic source at **1 MeV**. In standard neutron multigroup
        transport, an integer conventionally specifies a group-coordinate
        energy and is stored internally as a float. The coordinate must identify
        an available group. Continuous energy distributions are not supported
        in standard multigroup transport.
    discrete_energy : array_like of float, optional
        Discrete source-energy distribution with shape ``(2, N)``. The first
        row contains sampled energy values and the second row contains their
        probabilities. Values are physical energies in eV for continuous-energy
        transport and group-coordinate energies for standard multigroup
        transport. Standard-multigroup coordinates must be integer-valued and
        identify available groups. Cannot be supplied with ``energy``.
    time : real or array_like of float, optional
        Emission time in seconds. A real scalar, including a NumPy scalar,
        defines a discrete emission time. An array-like value with shape
        ``(2,)`` defines a uniform interval ``[t_min, t_max]``. Defaults to
        ``0.0``.
    particle_type : {"neutron", "electron", "proton"}, optional
        Type of emitted particle. Defaults to ``"neutron"``.
    probability : float, optional
        Relative source probability weight. Defaults to ``1.0``.

    Notes
    -----
    ``position`` and the box bounds ``x``, ``y``, and ``z`` are alternative
    spatial specifications and cannot be combined.

    When ``position`` is not provided, the source is treated as a box source.
    Any unspecified coordinate range defaults to ``[0.0, 0.0]`` cm. For
    example, if only ``z=[-1.0, 1.0]`` is specified, then the source occupies
    ``x=[0.0, 0.0]``, ``y=[0.0, 0.0]``, and ``z=[-1.0, 1.0]``.

    ``isotropic=True``, ``direction``, and ``white_direction`` are alternative
    angular specifications and cannot be combined. ``polar_cosine`` and
    ``azimuthal`` describe an angular spread about ``direction`` and therefore
    require it. If no angular specification is provided, the source is
    isotropic.

    Examples
    --------
    Point source at the origin emitting mono-energetic neutrons isotropically:

    >>> import numpy as np
    >>> import mcdc
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

    Sample discrete emission lines in a continuous-energy calculation:

    >>> decay_electrons = mcdc.Source(
    ...     particle_type="electron",
    ...     discrete_energy=(
    ...         [1.0e5, 2.0e5],
    ...         [0.8, 0.2],
    ...     ),
    ... )

    Sample a continuous-energy source from a tabulated probability density:

    >>> electron_source = mcdc.Source(
    ...     particle_type="electron",
    ...     energy=np.array([
    ...         [9_999.0, 10_001.0],
    ...         [0.5, 0.5],
    ...     ]),
    ...     direction=[0.0, 0.0, 1.0],
    ... )

    Sample between groups 0 and 1 in standard multigroup transport:

    >>> multigroup_source = mcdc.Source(
    ...     discrete_energy=(
    ...         [0.0, 1.0],
    ...         [0.25, 0.75],
    ...     ),
    ... )

    Time-dependent source:

    >>> src = mcdc.Source(
    ...     time=[0.0, 1.0e-3],
    ... )
    """

    # MC/DC framework metadata
    label = "source"

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
    discrete_energy: bool
    energy: float
    energy_pdf: DistributionTabulated
    energy_pmf: DistributionPMF

    # Time
    discrete_time: bool
    time: float
    time_range: Annotated[NDArray[float64], (2,)]

    # Misc.
    particle_type: int
    probability: float

    # Movement
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
        position: Sequence[float] | NoneType = None,
        x: Sequence[float] | NoneType = None,
        y: Sequence[float] | NoneType = None,
        z: Sequence[float] | NoneType = None,
        #
        direction: Sequence[float] | NoneType = None,
        white_direction: Sequence[float] | NoneType = None,
        isotropic: bool | NoneType = None,
        polar_cosine: Sequence[float] | NoneType = None,
        azimuthal: Sequence[float] | NoneType = None,
        #
        energy: float | ArrayLike | int | NoneType = None,
        discrete_energy: ArrayLike | NoneType = None,
        #
        time: ArrayLike = 0.0,
        #
        particle_type: str = "neutron",
        #
        probability: float = 1.0,
    ) -> None:
        super().__init__()

        self.name = name or "(Unnamed source)"

        # ==============================================================================
        # Default attributes
        #   Point source at origin,
        #   isotropic,
        #   mono-energetic at 1 MeV,
        #   time = 0,
        #   neutron
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
        self.discrete_energy = False
        self.energy = 1.0e6
        self.energy_pdf = DistributionTabulated(
            np.array([1.0e6 - 1.0, 1.0e6 + 1.0]),
            np.array([1.0, 1.0]),
        )
        self.energy_pmf = DistributionPMF(np.array([1.0e6]), np.array([1.0]))

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

        # Require one unambiguous source-position representation
        if position is not None and any(value is not None for value in (x, y, z)):
            print_error("Cannot specify position together with x, y, or z.")

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

        # Require one unambiguous source-direction representation
        isotropic_enabled = isotropic is not None and bool(isotropic)
        direction_modes = sum(
            (
                isotropic_enabled,
                direction is not None,
                white_direction is not None,
            )
        )
        if direction_modes > 1:
            print_error(
                "Cannot specify more than one of isotropic=True, direction, and "
                "white_direction."
            )
        if direction is None and (polar_cosine is not None or azimuthal is not None):
            print_error("polar_cosine and azimuthal require direction.")

        # Direction
        if isotropic_enabled:
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

        # Require one unambiguous source-energy representation
        if discrete_energy is not None and energy is not None:
            print_error("Cannot specify both energy and discrete_energy.")

        # Discrete energy
        if discrete_energy is not None:
            values, probabilities = _distribution_pair(
                discrete_energy, "Discrete energy"
            )
            self.mono_energetic = False
            self.discrete_energy = True
            self.energy_pmf = DistributionPMF(values, probabilities)

        # Energy
        if energy is not None:
            if isinstance(energy, Real) and not isinstance(energy, (bool, np.bool_)):
                self.energy = float(energy)
            else:
                values, pdf = _distribution_pair(energy, "Energy")
                self.mono_energetic = False
                self.energy_pdf = DistributionTabulated(values, pdf)

        # Time
        if isinstance(time, Real) and not isinstance(time, (bool, np.bool_)):
            self.time = float(time)
        else:
            self.discrete_time = False
            self.time_range = _time_range(time)

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

    def __repr__(self) -> str:
        text = super().__repr__()

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
        if self.mono_energetic:
            energy_text = f"{self.energy} eV"
        elif self.discrete_energy:
            energy_text = "PMF"
        else:
            energy_text = "PDF"
        text += f"  - Energy: {energy_text}\n"
        if self.discrete_time:
            text += f"  - Time: {self.time} s\n"
        else:
            text += f"  - Time: {self.time_range} s\n"

        return text

    # ==================================================================================
    # Source moving
    # ==================================================================================

    def move(self, velocities: ArrayLike, durations: ArrayLike) -> None:
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


def decode_particle_type(type_):
    """Return the display name for a packed particle-type code."""

    if type_ == PARTICLE_NEUTRON:
        return "Neutron"
    elif type_ == PARTICLE_ELECTRON:
        return "Electron"
    elif type_ == PARTICLE_PROTON:
        return "Proton"


# ======================================================================================
# Helper functions
# ======================================================================================


def _distribution_pair(
    value: ArrayLike, name: str
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Normalize a two-row distribution input and validate its shape."""
    try:
        array = np.asarray(value, dtype=float64)
    except (TypeError, ValueError):
        print_error(f"{name} distribution must be a rectangular array")

    if array.ndim != 2 or array.shape[0] != 2:
        print_error(f"{name} distribution must have shape (2, N)")

    return array[0], array[1]


def _time_range(value: ArrayLike) -> NDArray[float64]:
    """Normalize and validate a source time interval."""
    try:
        array = np.asarray(value, dtype=float64)
    except (TypeError, ValueError):
        print_error("Source time interval must be an array with shape (2,)")

    if array.shape != (2,):
        print_error("Source time interval must have shape (2,)")

    return array
