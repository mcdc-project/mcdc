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
    SURFACE_CONE_X,
    SURFACE_CONE_Y,
    SURFACE_CONE_Z,
    SURFACE_TORUS_X,
    SURFACE_TORUS_Y,
    SURFACE_TORUS_Z,
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
    type\\_ : int
        One of ``SURFACE_*`` constants (e.g., ``SURFACE_PLANE_X``).
    name : str
        Optional label for reporting.
    boundary_condition : str
        Boundary behavior at the surface (``"none"``, ``"vacuum"``, or ``"reflective"``).

    Attributes
    ----------
    ID : int
        Index in the global registry (assigned on construction).
    type\\_ : int
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
    R: float
    r: float
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

        # Torus surface parameters
        self.R = 0.0
        self.r = 0.0

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
        elif self.type == SURFACE_SPHERE:
            x = -0.5 * self.G
            y = -0.5 * self.H
            z = -0.5 * self.I
            r = (x**2 + y**2 + z**2 - self.J) ** 0.5
            text += f"  - Center (x, y, z): ({x}, {y}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CYLINDER:
            text += f"  - Coeffs.: {self.A}, {self.B}, {self.C},\n"
            text += f"             {self.D}, {self.E}, {self.F},\n"
            text += f"             {self.G}, {self.H}, {self.I}, {self.J}\n"
        elif self.type == SURFACE_CONE_X:
            t_sq = -self.A
            y0 = -0.5 * self.H
            z0 = -0.5 * self.I
            x0 = 0.0 if t_sq == 0.0 else 0.5 * self.G / t_sq
            text += f"  - Apex (x, y, z): ({x0}, {y0}, {z0}) cm\n"
            text += f"  - tan^2(theta): {t_sq}\n"
        elif self.type == SURFACE_CONE_Y:
            t_sq = -self.B
            x0 = -0.5 * self.G
            z0 = -0.5 * self.I
            y0 = 0.0 if t_sq == 0.0 else 0.5 * self.H / t_sq
            text += f"  - Apex (x, y, z): ({x0}, {y0}, {z0}) cm\n"
            text += f"  - tan^2(theta): {t_sq}\n"
        elif self.type == SURFACE_CONE_Z:
            t_sq = -self.C
            x0 = -0.5 * self.G
            y0 = -0.5 * self.H
            z0 = 0.0 if t_sq == 0.0 else 0.5 * self.I / t_sq
            text += f"  - Apex (x, y, z): ({x0}, {y0}, {z0}) cm\n"
            text += f"  - tan^2(theta): {t_sq}\n"
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
            User label.
        x : float, default 0.0
            Plane location (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
            User label.
        y : float, default 0.0
            Plane location (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
            User label.
        z : float, default 0.0
            Plane location (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
            User label.
        A, B, C, D : float
            Plane coefficients.
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
            User label.
        center : (2,) array_like of float, default (0, 0)
            Cylinder center in (y, z) (cm).
        radius : float, default 1.0
            Cylinder radius (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
            User label.
        center : (2,) array_like of float
            Cylinder center in (x, z) (cm).
        radius : float
            Cylinder radius (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
            User label.
        center : (2,) array_like of float
            Cylinder center in (x, y) (cm).
        radius : float
            Cylinder radius (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
        radius: float = 0.0,
        axis: Iterable[float] = [0.0, 0.0, 1.0],
        point: Iterable[float] = [0.0, 0.0, 0.0],
        boundary_condition: str = "none",
    ):
        """
        Create a general infinite cylinder with an arbitrary axis.

        Parameters
        ----------
        name : str, optional
        radius : float
            Cylinder radius (cm).
        axis : (3,) array_like of float
            Direction vector of the cylinder axis (normalized automatically).
        point : (3,) array_like of float
            A point on the cylinder axis (cm).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            General cylinder surface.
        """
        type_ = SURFACE_CYLINDER
        surface = cls(type_, name, boundary_condition)
        surface.linear = False

        # Axis and point
        ax, ay, az = axis
        norm = (ax**2 + ay**2 + az**2) ** 0.5
        dx, dy, dz = ax / norm, ay / norm, az / norm
        px, py, pz = point
        r = radius

        # Coefficients
        surface.A = 1.0 - dx**2
        surface.B = 1.0 - dy**2
        surface.C = 1.0 - dz**2
        surface.D = -2.0 * dx * dy
        surface.E = -2.0 * dx * dz
        surface.F = -2.0 * dy * dz
        Qpx = (1.0 - dx**2) * px - dx * dy * py - dx * dz * pz
        Qpy = -dx * dy * px + (1.0 - dy**2) * py - dy * dz * pz
        Qpz = -dx * dz * px - dy * dz * py + (1.0 - dz**2) * pz
        surface.G = -2.0 * Qpx
        surface.H = -2.0 * Qpy
        surface.I = -2.0 * Qpz
        pdotd = px * dx + py * dy + pz * dz
        surface.J = px**2 + py**2 + pz**2 - pdotd**2 - r**2

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
            User label.
        center : (3,) array_like of float
            Sphere center (x, y, z) in cm.
        radius : float
            Radius (cm).
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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
    def ConeX(
        cls,
        name: str = "",
        apex: Iterable[float] = [0.0, 0.0, 0.0],
        t_sq: float = 1.0,
        boundary_condition: str = "none",
    ):
        """
        Create an infinite cone with axis along the x-axis.

        Equation: (y - y0)^2 + (z - z0)^2 - t_sq * (x - x0)^2 = 0

        Parameters
        ----------
        name : str, optional
        apex : (3,) array_like of float
            Cone apex (x0, y0, z0) in cm.
        t_sq : float
            Squared tangent of the half-angle: t_sq = tan^2(theta).
            For a 45-degree half-angle use t_sq = 1.0.
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Cone-X surface.
        """
        type_ = SURFACE_CONE_X
        surface = cls(type_, name, boundary_condition)
        surface.linear = False

        x0, y0, z0 = apex

        surface.A = -t_sq
        surface.B = 1.0
        surface.C = 1.0
        surface.G = 2.0 * t_sq * x0
        surface.H = -2.0 * y0
        surface.I = -2.0 * z0
        surface.J = y0**2 + z0**2 - t_sq * x0**2

        return surface

    @classmethod
    def ConeY(
        cls,
        name: str = "",
        apex: Iterable[float] = [0.0, 0.0, 0.0],
        t_sq: float = 1.0,
        boundary_condition: str = "none",
    ):
        """
        Create an infinite cone with axis along the y-axis.

        Equation: (x - x0)^2 + (z - z0)^2 - t_sq * (y - y0)^2 = 0

        Parameters
        ----------
        name : str, optional
        apex : (3,) array_like of float
            Cone apex (x0, y0, z0) in cm.
        t_sq : float
            Squared tangent of the half-angle: t_sq = tan^2(theta).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Cone-Y surface.
        """
        type_ = SURFACE_CONE_Y
        surface = cls(type_, name, boundary_condition)
        surface.linear = False

        x0, y0, z0 = apex

        surface.A = 1.0
        surface.B = -t_sq
        surface.C = 1.0
        surface.G = -2.0 * x0
        surface.H = 2.0 * t_sq * y0
        surface.I = -2.0 * z0
        surface.J = x0**2 + z0**2 - t_sq * y0**2

        return surface

    @classmethod
    def ConeZ(
        cls,
        name: str = "",
        apex: Iterable[float] = [0.0, 0.0, 0.0],
        t_sq: float = 1.0,
        boundary_condition: str = "none",
    ):
        """
        Create an infinite cone with axis along the z-axis.

        Equation: (x - x0)^2 + (y - y0)^2 - t_sq * (z - z0)^2 = 0

        Parameters
        ----------
        name : str, optional
        apex : (3,) array_like of float
            Cone apex (x0, y0, z0) in cm.
        t_sq : float
            Squared tangent of the half-angle: t_sq = tan^2(theta).
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Cone surface.
        """
        type_ = SURFACE_CONE_Z
        surface = cls(type_, name, boundary_condition)
        surface.linear = False

        x0, y0, z0 = apex

        surface.A = 1.0
        surface.B = 1.0
        surface.C = -t_sq
        surface.G = -2.0 * x0
        surface.H = -2.0 * y0
        surface.I = 2.0 * t_sq * z0
        surface.J = x0**2 + y0**2 - t_sq * z0**2

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
            User label.
        A,B,C,D,E,F,G,H,I,J : float
            Quadric coefficients.
        boundary_condition : str, optional
            Boundary type (``"none"``, ``"vacuum"``, or ``"reflective"``).

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

    @classmethod
    def TorusX(
        cls,
        name: str = "",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        R: float = 0.0,
        r: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a torus radially symmetric around the x axis:
            f(x, y, z) = ( sqrt[(y - B)^2 + (z - C)^2] - R )^2 + (x - A)^2 - r^2

        Parameters
        ----------
        name : str, optional
        A,B,C,R,r : float
            A, B, C are displacement values for the torus in the x, y, z directions respectfully
            R is the radius around which a circle is revolved about the axis of revolution (parallel with the x-axis)
            r is the radius of the circle that is being revolved
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Torus surface.
        """
        type_ = SURFACE_TORUS_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.R = R
        surface.r = r

        return surface

    @classmethod
    def TorusY(
        cls,
        name: str = "",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        R: float = 0.0,
        r: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a torus radially symmetric around the y axis:
            f(x, y, z) = ( sqrt[(x - A)^2 + (z - C)^2] - R )^2 + (y - B)^2 - r^2

        Parameters
        ----------
        name : str, optional
        A,B,C,R,r : float
            A, B, C are displacement values for the torus in the x, y, z directions respectfully
            R is the radius around which a circle is revolved about the axis of revolution (parallel with the y-axis)
            r is the radius of the circle that is being revolved
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Torus surface.
        """
        type_ = SURFACE_TORUS_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.R = R
        surface.r = r

        return surface

    @classmethod
    def TorusZ(
        cls,
        name: str = "",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        R: float = 0.0,
        r: float = 0.0,
        boundary_condition: str = "none",
    ):
        """
        Create a torus radially symmetric around the z axis:
            f(x, y, z) = ( sqrt[(x - A)^2 + (y - B)^2] - R )^2 + (z - C)^2 - r^2

        Parameters
        ----------
        name : str, optional
        A,B,C,R,r : float
            A, B, C are displacement values for the torus in the x, y, z directions respectfully
            R is the radius around which a circle is revolved about the axis of revolution (parallel with the z-axis)
            r is the radius of the circle that is being revolved
        boundary_condition : {"none","vacuum","reflective"}, optional

        Returns
        -------
        Surface
            Torus surface.
        """
        type_ = SURFACE_TORUS_Z
        surface = cls(type_, name, boundary_condition)

        surface.linear = False

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.R = R
        surface.r = r

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
    elif type_ == SURFACE_CONE_X:
        return "Infinite cone-X surface"
    elif type_ == SURFACE_CONE_Y:
        return "Infinite cone-Y surface"
    elif type_ == SURFACE_CONE_Z:
        return "Infinite cone-Z surface"
    elif type_ == SURFACE_TORUS_X:
        return "Torus-X surface"
    elif type_ == SURFACE_TORUS_Y:
        return "Torus-Y surface"
    elif type_ == SURFACE_TORUS_Z:
        return "Torus-Z surface"


def decode_BC_type(type_):
    if type_ == BC_NONE:
        return "None"
    elif type_ == BC_VACUUM:
        return "Vacuum"
    elif type_ == BC_REFLECTIVE:
        return "Reflective"
