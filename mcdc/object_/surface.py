from typing import Annotated, Iterable
import numpy as np

from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import (
    BC_NONE,
    BC_REFLECTIVE,
    BC_VACUUM,
    INF,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_CYLINDER,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_PLANE,
    SURFACE_SPHERE,
    SURFACE_QUADRIC,
)
from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.cell import Region
from mcdc.object_.tally import TallySurface
from mcdc.object_.util import move_object

# ======================================================================================
# Surface
# ======================================================================================


class Surface(ObjectNonSingleton):
    """
    Geometric surface primitive with optional boundary condition and motion.

    Surfaces are registered non-singletons and receive a stable ``ID``. Factory
    constructors (:meth:`PlaneX`, :meth:`CylinderZ`, etc.) set the quadric
    coefficients (A..J) and linearity flag. Motion segments can be defined with
    :meth:`move`.

    Parameters
    ----------
    type_ : int
        One of ``SURFACE_*`` constants (e.g., ``SURFACE_PLANE_X``).
    name : str
        Optional label for reporting.
    boundary_condition : {"none","vacuum","reflective"}
        Boundary behavior at the surface.

    Attributes
    ----------
    ID : int
        Index in the global registry (assigned on construction).
    type : int
        Surface type code (``SURFACE_*``).
    name : str
        User label.
    boundary_condition : int
        One of ``BC_NONE``, ``BC_VACUUM``, ``BC_REFLECTIVE``.
    A,B,C,D,E,F,G,H,I,J : float
        Quadric coefficients defining the implicit surface.
    linear : bool
        True for linear (plane) surfaces; False for general quadrics.
    nx, ny, nz : float
        Outward normal components for linear planes.
    moving : bool
        True if :meth:`move` has been called.
    N_move : int
        Number of motion segments plus the final static segment.
    move_velocities : (N_move, 3) ndarray
        Per-segment velocity vectors.
    move_durations : (N_move,) ndarray
        Per-segment durations (s).
    move_time_grid : (N_move+1,) ndarray
        Cumulative time breakpoints.
    move_translations : (N_move+1, 3) ndarray
        Cumulative translations at each breakpoint.

    See Also
    --------
    Region
        Use unary ``+`` / ``-`` to form half-spaces: ``+surface`` or ``-surface``.
    decode_type
        Human-readable surface type.
    decode_BC_type
        Human-readable boundary condition name.
    """

    # Annotations for Numba mode
    label: str = "surface"
    #
    type: int
    name: str
    boundary_condition: int
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float
    I: float
    J: float
    linear: bool
    nx: float
    ny: float
    nz: float
    moving: bool
    N_move: int
    N_move_grid: int
    move_velocities: Annotated[NDArray[float64], ("N_move", 3)]
    move_durations: Annotated[NDArray[float64], ("N_move",)]
    move_time_grid: Annotated[NDArray[float64], ("N_move_grid",)]
    move_translations: Annotated[NDArray[float64], ("N_move_grid", 3)]
    tallies: list[TallySurface]

    def __init__(self, type_, name, boundary_condition):
        super().__init__()

        # Type and name
        self.type = type_
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.ID}"

        # Boundary condition
        if boundary_condition == "none":
            self.boundary_condition = BC_NONE
        elif boundary_condition == "vacuum":
            self.boundary_condition = BC_VACUUM
        elif boundary_condition == "reflective":
            self.boundary_condition = BC_REFLECTIVE

        # Quadric surface coefficients
        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.E = 0.0
        self.F = 0.0
        self.G = 0.0
        self.H = 0.0
        self.I = 0.0
        self.J = 0.0

        # Helpers
        self.linear = True
        # Surface normal direction (if linear)
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0

        # Moving surface parameters
        self.moving = False
        self.N_move = 1
        self.N_move_grid = 2
        self.move_velocities = np.zeros((1, 3))
        self.move_durations = np.array([INF])
        self.move_time_grid = np.array([0.0, INF])
        self.move_translations = np.zeros((2, 3))

        # Surface tallies
        self.tallies = []

    def __repr__(self):
        """
        Return a human-readable description including type-specific parameters.

        Returns
        -------
        str
            Multi-line formatted string with ID, name, BC, and geometry details.
        """
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Boundary condition: {decode_BC_type(self.boundary_condition)}\n"

        # ==============================================================================
        # Type-based repr
        # ==============================================================================

        if self.type == SURFACE_PLANE_X:
            text += f"  - x0: {-self.J} cm\n"
        elif self.type == SURFACE_PLANE_Y:
            text += f"  - y0: {-self.J} cm\n"
        elif self.type == SURFACE_PLANE_Z:
            text += f"  - z0: {-self.J} cm\n"
        elif self.type == SURFACE_PLANE:
            text += f"  - Coeffs.: {self.G}, {self.H}, {self.I}, {self.J}\n"
            text += f"  - Normal: ({self.nx}, {self.ny}, {self.nz})\n"
        elif self.type == SURFACE_CYLINDER_X:
            y = -0.5 * self.H
            z = -0.5 * self.I
            r = (y**2 + z**2 - self.J) ** 0.5
            text += f"  - Center (y, z): ({y}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CYLINDER_Y:
            x = -0.5 * self.G
            z = -0.5 * self.I
            r = (x**2 + z**2 - self.J) ** 0.5
            text += f"  - Center (x, z): ({x}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CYLINDER_Z:
            x = -0.5 * self.G
            y = -0.5 * self.H
            r = (x**2 + y**2 - self.J) ** 0.5
            text += f"  - Center (x, y): ({x}, {y}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CYLINDER:
            text += f"  - Coeffs.: {self.A}, {self.B}, {self.C},\n"
            text += f"             {self.G}, {self.H}, {self.I}, {self.J}\n"
        elif self.type == SURFACE_SPHERE:
            x = -0.5 * self.G
            y = -0.5 * self.H
            z = -0.5 * self.I
            r = (x**2 + y**2 + z**2 - self.J) ** 0.5
            text += f"  - Center (x, y, z): ({x}, {y}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_QUADRIC:
            text += f"  - Coeffs.: {self.A}, {self.B}, {self.C},\n"
            text += f"             {self.D}, {self.E}, {self.F},\n"
            text += f"             {self.G}, {self.H}, {self.I}, {self.J}\n"

        if len(self.tallies) > 0:
            text += f"  - Tallies: {[x.ID for x in self.tallies]}\n"

        return text

    # ==================================================================================
    # Type-based creation methods
    # ==================================================================================

    @classmethod
    def PlaneX(cls, name: str = "", x: float = 0.0, boundary_condition: str = "none"):
        """
        Create a plane perpendicular to +x at x = constant.

        Parameters
        ----------
        name : str, optional
        x : float, default 0.0
            Plane location (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Linear plane with normal ``(+1, 0, 0)``.
        """
        type_ = SURFACE_PLANE_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.G = 1.0
        surface.J = -x
        surface.nx = 1.0

        return surface

    @classmethod
    def PlaneY(cls, name: str = "", y: float = 0.0, boundary_condition: str = "none"):
        """
        Create a plane perpendicular to +y at y = constant.

        Parameters
        ----------
        name : str, optional
        y : float, default 0.0
            Plane location (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Linear plane with normal ``(0, +1, 0)``.
        """
        type_ = SURFACE_PLANE_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.H = 1.0
        surface.J = -y
        surface.ny = 1.0

        return surface

    @classmethod
    def PlaneZ(cls, name: str = "", z: float = 0.0, boundary_condition: str = "none"):
        """
        Create a plane perpendicular to +z at z = constant.

        Parameters
        ----------
        name : str, optional
        z : float, default 0.0
            Plane location (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Linear plane with normal ``(0, 0, +1)``.
        """
        type_ = SURFACE_PLANE_Z
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.I = 1.0
        surface.J = -z
        surface.nz = 1.0

        return surface

    @classmethod
    def Plane(
        cls,
        name: str = "",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        D: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a general plane defined by A x + B y + C z + D = 0.

        The normal is normalized to unit length and stored in ``(nx, ny, nz)``.

        Parameters
        ----------
        name : str, optional
        A, B, C, D : float
            Plane coefficients.
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Linear plane with normalized normal vector.
        """
        type_ = SURFACE_PLANE
        surface = cls(type_, name, boundary_condition)

        surface.linear = True

        # Normalize
        norm = (A**2 + B**2 + C**2) ** 0.5
        A /= norm
        B /= norm
        C /= norm
        D /= norm

        # Coefficients
        surface.G = A
        surface.H = B
        surface.I = C
        surface.J = D

        # Surface normal direction
        surface.nx = A
        surface.ny = B
        surface.nz = C
        return surface

    @classmethod
    def CylinderX(
        cls,
        name: str = "",
        center: Iterable[float] = [0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create an infinite cylinder aligned with the x-axis.

        Parameters
        ----------
        name : str, optional
        center : (2,) array_like of float, default (0, 0)
            Cylinder center in (y, z) (cm).
        radius : float, default 1.0
            Cylinder radius (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Quadratic cylinder surface.
        """
        type_ = SURFACE_CYLINDER_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Center and radius
        y, z = center
        r = radius

        # Coefficients
        surface.B = 1.0
        surface.C = 1.0
        surface.H = -2.0 * y
        surface.I = -2.0 * z
        surface.J = y**2 + z**2 - r**2
        return surface

    @classmethod
    def CylinderY(
        cls,
        name: str = "",
        center: Iterable[float] = [0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create an infinite cylinder aligned with the y-axis.

        Parameters
        ----------
        name : str, optional
        center : (2,) array_like of float
            Cylinder center in (x, z) (cm).
        radius : float
            Cylinder radius (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Quadratic cylinder surface.
        """
        type_ = SURFACE_CYLINDER_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Center and radius
        x, z = center
        r = radius

        # Coefficients
        surface.A = 1.0
        surface.C = 1.0
        surface.G = -2.0 * x
        surface.I = -2.0 * z
        surface.J = x**2 + z**2 - r**2
        return surface

    @classmethod
    def CylinderZ(
        cls,
        name: str = "",
        center: Iterable[float] = [0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create an infinite cylinder aligned with the z-axis.

        Parameters
        ----------
        name : str, optional
        center : (2,) array_like of float
            Cylinder center in (x, y) (cm).
        radius : float
            Cylinder radius (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Quadratic cylinder surface.
        """
        type_ = SURFACE_CYLINDER_Z
        surface = cls(type_, name, boundary_condition)
        surface.linear = False

        # Center and radius
        x, y = center
        r = radius

        # Coefficients
        surface.A = 1.0
        surface.B = 1.0
        surface.G = -2.0 * x
        surface.H = -2.0 * y
        surface.J = x**2 + y**2 - r**2

        return surface

    @classmethod
    def Cylinder(
        cls,
        name: str = "",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        G: float = 0.0,
        H: float = 0.0,
        I: float = 0.0,
        J: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a general infinite cylinder (diagonal quadric without cross terms):
            A x^2 + B y^2 + C z^2 + G x + H y + I z + J = 0

        Parameters
        ----------
        name : str, optional
        A, B, C, G, H, I, J : float
            Cylinder coefficients.
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            General cylinder surface.
        """
        type_ = SURFACE_CYLINDER
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.G = G
        surface.H = H
        surface.I = I
        surface.J = J
        return surface

    @classmethod
    def Sphere(
        cls,
        name: str = "",
        center: Iterable[float] = [0.0, 0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a sphere.

        Parameters
        ----------
        name : str, optional
        center : (3,) array_like of float
            Sphere center (x, y, z) in cm.
        radius : float
            Radius (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Quadratic spherical surface.
        """
        type_ = SURFACE_SPHERE
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Center and radius
        x, y, z = center
        r = radius

        # Coefficients
        surface.A = 1.0
        surface.B = 1.0
        surface.C = 1.0
        surface.G = -2.0 * x
        surface.H = -2.0 * y
        surface.I = -2.0 * z
        surface.J = x**2 + y**2 + z**2 - r**2
        return surface

    @classmethod
    def Quadric(
        cls,
        name: str = "",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        D: float = 0.0,
        E: float = 0.0,
        F: float = 0.0,
        G: float = 0.0,
        H: float = 0.0,
        I: float = 0.0,
        J: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a general quadric:
            A x^2 + B y^2 + C z^2 + D xy + E yz + F zx + G x + H y + I z + J = 0

        Parameters
        ----------
        name : str, optional
        A,B,C,D,E,F,G,H,I,J : float
            Quadric coefficients.
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            General quadratic surface.
        """
        type_ = SURFACE_QUADRIC
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.D = D
        surface.E = E
        surface.F = F
        surface.G = G
        surface.H = H
        surface.I = I
        surface.J = J
        return surface

    # ==================================================================================
    # Region building
    # ==================================================================================

    def __pos__(self):
        """
        Half-space on the **outward** side of the surface.

        Returns
        -------
        Region
            Region representing ``n · r + J >= 0`` (sign convention per type).
        """
        return Region.make_halfspace(self, +1)

    def __neg__(self):
        """
        Half-space on the **inward** side of the surface.

        Returns
        -------
        Region
            Region representing the complement half-space.
        """
        return Region.make_halfspace(self, -1)

    # ==================================================================================
    # Surface moving
    # ==================================================================================

    def move(self, velocities, durations):
        """
        Define piecewise-constant motion for the surface.

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
        >>> s = Surface.PlaneZ(z=0.0)
        >>> s.move(velocities=[[0,0,1.0]], durations=[0.5])  # 0.5 s upward, then static
        >>> s.N_move
        2
        """
        move_object(self, velocities, durations)


# ======================================================================================
# Type decoder
# ======================================================================================


def decode_type(type_):
    if type_ == SURFACE_PLANE_X:
        return "Plane-X surface"
    elif type_ == SURFACE_PLANE_Y:
        return "Plane-Y surface"
    elif type_ == SURFACE_PLANE_Z:
        return "Plane-Z surface"
    elif type_ == SURFACE_PLANE:
        return "Plane surface"
    elif type_ == SURFACE_CYLINDER_X:
        return "Infinite cylinder-X surface"
    elif type_ == SURFACE_CYLINDER_Y:
        return "Infinite cylinder-Y surface"
    elif type_ == SURFACE_CYLINDER_Z:
        return "Infinite cylinder-Z surface"
    elif type_ == SURFACE_CYLINDER:
        return "General cylinder surface"
    elif type_ == SURFACE_SPHERE:
        return "Sphere surface"
    elif type_ == SURFACE_QUADRIC:
        return "Quadric surface"


def decode_BC_type(type_):
    if type_ == BC_NONE:
        return "None"
    elif type_ == BC_VACUUM:
        return "Vacuum"
    elif type_ == BC_REFLECTIVE:
        return "Reflective"
