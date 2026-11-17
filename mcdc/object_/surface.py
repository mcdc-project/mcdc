import math
from typing import Annotated, Sequence

import numpy as np

from numpy import float64
from numpy.typing import ArrayLike, NDArray

####

from mcdc.constant import (
    BC_NONE,
    BC_REFLECTIVE,
    BC_VACUUM,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_CYLINDER,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_PLANE,
    SURFACE_SPHERE,
    SURFACE_CONE_X,
    SURFACE_CONE_Y,
    SURFACE_CONE_Z,
    SURFACE_CONE,
    SURFACE_QUADRIC,
    SURFACE_TORUS_X,
    SURFACE_TORUS_Y,
    SURFACE_TORUS_Z,
    SURFACE_TORUS,
)
from mcdc.object_.base import MCDCObject
from mcdc.object_.cell import Region
from mcdc.object_.tally import TallySurfaceCrossing
from mcdc.object_.util import initialize_motion, move_object
from mcdc.print_ import print_error

# ======================================================================================
# Surface
# ======================================================================================


class Surface(MCDCObject):
    """Geometric boundaries of simulation cells.

    Surfaces are created with class methods such as :meth:`PlaneX`,
    :meth:`CylinderZ`, and :meth:`Sphere`. Unary ``+`` and ``-`` return the
    corresponding positive and negative half-space
    :class:`~mcdc.object_.cell.Region`.

    Boundary conditions may be ``"none"``, ``"vacuum"``, or ``"reflective"``.
    A surface can also undergo piecewise-constant translational motion configured
    with :meth:`move`.

    Examples
    --------
    Create a vacuum x plane and select its positive half-space:

    >>> import numpy as np
    >>> import mcdc
    >>> plane = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
    >>> region = +plane

    Create a sphere and select its interior:

    >>> sphere = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=2.0)
    >>> interior = -sphere

    Create a cylinder parallel to the z axis:

    >>> cylinder = mcdc.Surface.CylinderZ(
    ...     center=[1.0, -1.0],
    ...     radius=0.5,
    ... )

    Create a cone about an arbitrary axis:

    >>> cone = mcdc.Surface.Cone(
    ...     apex=[0.0, 0.0, 0.0],
    ...     axis=[0.0, 1.0, 1.0],
    ...     half_angle=30.0,
    ... )

    Create an oblique plane from its equation coefficients:

    >>> oblique = mcdc.Surface.Plane(A=1.0, B=1.0, C=0.0, D=-2.0)

    Create a torus with an arbitrary symmetry axis:

    >>> torus = mcdc.Surface.Torus(
    ...     center=[0.0, 0.0, 0.0],
    ...     axis=[1.0, 1.0, 0.0],
    ...     R=2.0,
    ...     r=0.5,
    ... )

    Define piecewise-constant motion for a plane:

    >>> moving_plane = mcdc.Surface.PlaneX(x=0.0)
    >>> moving_plane.move(
    ...     velocities=np.array([[1.0, 0.0, 0.0]]),
    ...     durations=np.array([0.5]),
    ... )
    """

    # MC/DC framework metadata
    label = "surface"

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
    quadric: bool
    quartic: bool
    nx: float
    ny: float
    nz: float
    # Surface motion
    moving: bool
    N_move: int
    N_move_grid: int
    move_velocities: Annotated[NDArray[float64], ("N_move", 3)]
    move_durations: Annotated[NDArray[float64], ("N_move",)]
    move_time_grid: Annotated[NDArray[float64], ("N_move_grid",)]
    move_translations: Annotated[NDArray[float64], ("N_move_grid", 3)]
    surface_crossing_tallies: list[TallySurfaceCrossing]

    def __init__(self, type_: int, name: str, boundary_condition: str) -> None:
        super().__init__()

        self.type = type_
        self.name = name or "(Unnamed surface)"

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
        self.quadric = False
        self.quartic = False

        # Surface normal direction (if linear)
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0

        # Initialize stationary surface motion.
        initialize_motion(self)

        # Surface-crossing tallies
        self.surface_crossing_tallies = []

    def __repr__(self) -> str:
        text = super().__repr__()

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
            text += f"  - Point: ({self.A}, {self.B}, {self.C}) cm\n"
            text += f"  - Axis: ({self.nx}, {self.ny}, {self.nz})\n"
            text += f"  - Radius: {self.R} cm\n"
        elif self.type == SURFACE_SPHERE:
            x = -0.5 * self.G
            y = -0.5 * self.H
            z = -0.5 * self.I
            r = (x**2 + y**2 + z**2 - self.J) ** 0.5
            text += f"  - Center (x, y, z): ({x}, {y}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CONE_X:
            t_sq = -self.A
            y0 = -0.5 * self.H
            z0 = -0.5 * self.I
            x0 = 0.0 if t_sq == 0.0 else 0.5 * self.G / t_sq
            text += f"  - Apex (x, y, z): ({x0}, {y0}, {z0}) cm\n"
            text += f"  - Half-angle: {_get_cone_half_angle(t_sq):.12g} degrees\n"
        elif self.type == SURFACE_CONE_Y:
            t_sq = -self.B
            x0 = -0.5 * self.G
            z0 = -0.5 * self.I
            y0 = 0.0 if t_sq == 0.0 else 0.5 * self.H / t_sq
            text += f"  - Apex (x, y, z): ({x0}, {y0}, {z0}) cm\n"
            text += f"  - Half-angle: {_get_cone_half_angle(t_sq):.12g} degrees\n"
        elif self.type == SURFACE_CONE_Z:
            t_sq = -self.C
            x0 = -0.5 * self.G
            y0 = -0.5 * self.H
            z0 = 0.0 if t_sq == 0.0 else 0.5 * self.I / t_sq
            text += f"  - Apex (x, y, z): ({x0}, {y0}, {z0}) cm\n"
            text += f"  - Half-angle: {_get_cone_half_angle(t_sq):.12g} degrees\n"
        elif self.type == SURFACE_CONE:
            text += f"  - Apex: ({self.A}, {self.B}, {self.C}) cm\n"
            text += f"  - Axis: ({self.nx}, {self.ny}, {self.nz})\n"
            text += f"  - Half-angle: {_get_cone_half_angle(self.R):.12g} degrees\n"
        elif self.type == SURFACE_QUADRIC:
            text += f"  - Coeffs.: {self.A}, {self.B}, {self.C},\n"
            text += f"             {self.D}, {self.E}, {self.F},\n"
            text += f"             {self.G}, {self.H}, {self.I}, {self.J}\n"
        elif self.type == SURFACE_TORUS_X:
            text += f"  - A, B, C: {self.A}, {self.B}, {self.C}\n"
            text += f"  - R: {self.R} cm\n"
            text += f"  - r: {self.r} cm\n"
        elif self.type == SURFACE_TORUS_Y:
            text += f"  - A, B, C: {self.A}, {self.B}, {self.C}\n"
            text += f"  - R: {self.R} cm\n"
            text += f"  - r: {self.r} cm\n"
        elif self.type == SURFACE_TORUS_Z:
            text += f"  - A, B, C: {self.A}, {self.B}, {self.C}\n"
            text += f"  - R: {self.R} cm\n"
            text += f"  - r: {self.r} cm\n"
        elif self.type == SURFACE_TORUS:
            text += f"  - A, B, C: {self.A}, {self.B}, {self.C}\n"
            text += f"  - R: {self.R} cm\n"
            text += f"  - r: {self.r} cm\n"
        if len(self.surface_crossing_tallies) > 0:
            text += f"  - Surface-crossing tallies: {[x.name for x in self.surface_crossing_tallies]}\n"

        return text

    # ==================================================================================
    # Type-based creation methods
    # ==================================================================================

    @classmethod
    def PlaneX(
        cls,
        name: str = "",
        x: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create the plane ``x = constant``.

        Parameters
        ----------
        name : str, optional
            User-facing surface name.
        x : float, optional
            Plane position in cm.
        boundary_condition : {"none", "vacuum", "reflective"}, optional
            Boundary condition applied when a particle crosses the plane.
        """
        type_ = SURFACE_PLANE_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.quadric = False
        surface.quartic = False

        surface.G = 1.0
        surface.J = -x
        surface.nx = 1.0

        return surface

    @classmethod
    def PlaneY(
        cls,
        name: str = "",
        y: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create the plane ``y = constant``.

        Parameters are the surface ``name``, position ``y`` in cm, and a
        ``boundary_condition`` of ``"none"``, ``"vacuum"``, or
        ``"reflective"``.
        """
        type_ = SURFACE_PLANE_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.quadric = False
        surface.quartic = False

        surface.H = 1.0
        surface.J = -y
        surface.ny = 1.0

        return surface

    @classmethod
    def PlaneZ(
        cls,
        name: str = "",
        z: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create the plane ``z = constant``.

        Parameters are the surface ``name``, position ``z`` in cm, and a
        ``boundary_condition`` of ``"none"``, ``"vacuum"``, or
        ``"reflective"``.
        """
        type_ = SURFACE_PLANE_Z
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.quadric = False
        surface.quartic = False

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
    ) -> "Surface":
        """Create a general plane ``A*x + B*y + C*z + D = 0``.

        The coefficients are normalized internally. ``(A, B, C)`` must be a
        nonzero normal vector. Coordinates are evaluated in cm, so ``D`` must
        use the corresponding length scaling.
        """
        type_ = SURFACE_PLANE
        surface = cls(type_, name, boundary_condition)

        surface.linear = True
        surface.quadric = False
        surface.quartic = False

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
        center: Sequence[float] = [0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create an infinite cylinder parallel to the x axis.

        ``center`` gives ``[y, z]`` in cm and ``radius`` is in cm.
        """
        type_ = SURFACE_CYLINDER_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
        center: Sequence[float] = [0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create an infinite cylinder parallel to the y axis.

        ``center`` gives ``[x, z]`` in cm and ``radius`` is in cm.
        """
        type_ = SURFACE_CYLINDER_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
        center: Sequence[float] = [0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create an infinite cylinder parallel to the z axis.

        ``center`` gives ``[x, y]`` in cm and ``radius`` is in cm.
        """
        type_ = SURFACE_CYLINDER_Z
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
        axis: Sequence[float] = [0.0, 0.0, 1.0],
        point: Sequence[float] = [0.0, 0.0, 0.0],
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create an infinite cylinder with an arbitrary axis.

        Parameters
        ----------
        radius : float, optional
            Cylinder radius in cm.
        axis : sequence of 3 float, optional
            Nonzero vector parallel to the cylinder axis.
        point : sequence of 3 float, optional
            A point on the cylinder axis, in cm.
        """
        # Axis and point
        ax, ay, az = axis
        norm = (ax**2 + ay**2 + az**2) ** 0.5
        if norm == 0.0:
            print_error("Cylinder axis must be a nonzero vector.")

        type_ = SURFACE_CYLINDER
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

        surface.A, surface.B, surface.C = point
        surface.nx = ax / norm
        surface.ny = ay / norm
        surface.nz = az / norm
        surface.R = radius

        return surface

    @classmethod
    def Sphere(
        cls,
        name: str = "",
        center: Sequence[float] = [0.0, 0.0, 0.0],
        radius: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create a sphere from its center and radius in cm."""
        type_ = SURFACE_SPHERE
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
        apex: Sequence[float] = [0.0, 0.0, 0.0],
        half_angle: float = 45.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create a double cone aligned with the x axis.

        Parameters
        ----------
        apex : sequence of 3 float, optional
            Coordinates of the shared vertex of the two cone nappes, in cm.
        half_angle : float, optional
            Angle between the x axis and the cone surface, in degrees. Must be
            greater than zero and less than 90 degrees.
        """
        t_sq = _get_cone_t_sq(half_angle)
        type_ = SURFACE_CONE_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
        apex: Sequence[float] = [0.0, 0.0, 0.0],
        half_angle: float = 45.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create a double cone aligned with the y axis.

        Parameters
        ----------
        apex : sequence of 3 float, optional
            Coordinates of the shared vertex of the two cone nappes, in cm.
        half_angle : float, optional
            Angle between the y axis and the cone surface, in degrees. Must be
            greater than zero and less than 90 degrees.
        """
        t_sq = _get_cone_t_sq(half_angle)
        type_ = SURFACE_CONE_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
        apex: Sequence[float] = [0.0, 0.0, 0.0],
        half_angle: float = 45.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create a double cone aligned with the z axis.

        Parameters
        ----------
        apex : sequence of 3 float, optional
            Coordinates of the shared vertex of the two cone nappes, in cm.
        half_angle : float, optional
            Angle between the z axis and the cone surface, in degrees. Must be
            greater than zero and less than 90 degrees.
        """
        t_sq = _get_cone_t_sq(half_angle)
        type_ = SURFACE_CONE_Z
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
    def Cone(
        cls,
        name: str = "",
        apex: Sequence[float] = [0.0, 0.0, 0.0],
        axis: Sequence[float] = [0.0, 0.0, 1.0],
        half_angle: float = 45.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create a double cone with an arbitrary axis.

        Parameters
        ----------
        apex : sequence of 3 float, optional
            Coordinates of the shared vertex of the two cone nappes, in cm.
        axis : sequence of 3 float, optional
            Nonzero vector parallel to the cone's axis of symmetry. It is
            normalized internally; for a double cone, its sign does not change
            the surface.
        half_angle : float, optional
            Angle between ``axis`` and the cone surface, in degrees. Must be
            greater than zero and less than 90 degrees.
        """
        ax, ay, az = axis
        norm = (ax**2 + ay**2 + az**2) ** 0.5
        if norm == 0.0:
            print_error("Cone axis must be a nonzero vector.")
        t_sq = _get_cone_t_sq(half_angle)

        type_ = SURFACE_CONE
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

        surface.A, surface.B, surface.C = apex
        surface.nx = ax / norm
        surface.ny = ay / norm
        surface.nz = az / norm
        surface.R = t_sq

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
    ) -> "Surface":
        """Create a general second-degree surface.

        The coefficients define
        ``A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F*y*z
        + G*x + H*y + I*z + J = 0``. Coordinates are evaluated in cm; the
        coefficients must therefore use mutually consistent units.
        """
        type_ = SURFACE_QUADRIC
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = True
        surface.quartic = False

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
    ) -> "Surface":
        """Create a torus centered at ``(A, B, C)`` and aligned with x.

        ``(A, B, C)`` gives the center in cm. ``R`` is the major radius and
        ``r`` the minor radius, both in cm.
        """
        type_ = SURFACE_TORUS_X
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = False
        surface.quartic = True

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
    ) -> "Surface":
        """Create a torus centered at ``(A, B, C)`` and aligned with y.

        ``(A, B, C)`` gives the center in cm. ``R`` is the major radius and
        ``r`` the minor radius, both in cm.
        """
        type_ = SURFACE_TORUS_Y
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = False
        surface.quartic = True

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
    ) -> "Surface":
        """Create a torus centered at ``(A, B, C)`` and aligned with z.

        ``(A, B, C)`` gives the center in cm. ``R`` is the major radius and
        ``r`` the minor radius, both in cm.
        """
        type_ = SURFACE_TORUS_Z
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = False
        surface.quartic = True

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.R = R
        surface.r = r

        return surface

    @classmethod
    def Torus(
        cls,
        name: str = "",
        center: Sequence[float] = [0.0, 0.0, 0.0],
        axis: Sequence[float] = [0.0, 0.0, 1.0],
        R: float = 0.0,
        r: float = 0.0,
        boundary_condition: str = "none",
    ) -> "Surface":
        """Create a torus with an arbitrary axis.

        Parameters
        ----------
        center : sequence of 3 float, optional
            Torus center in cm.
        axis : sequence of 3 float, optional
            Nonzero symmetry-axis vector.
        R : float, optional
            Major radius in cm.
        r : float, optional
            Minor radius in cm.
        """
        x, y, z = center
        ax, ay, az = axis
        norm = (ax**2 + ay**2 + az**2) ** 0.5

        # if the axis is zero, we will get a division by zero when we try to normalize it
        if norm == 0.0:
            print_error("Torus axis must be a nonzero vector.")

        type_ = SURFACE_TORUS
        surface = cls(type_, name, boundary_condition)

        surface.linear = False
        surface.quadric = False
        surface.quartic = True

        surface.A = x
        surface.B = y
        surface.C = z
        surface.nx = ax / norm
        surface.ny = ay / norm
        surface.nz = az / norm
        surface.R = R
        surface.r = r

        return surface

    # ==================================================================================
    # Region building
    # ==================================================================================

    def __pos__(self) -> Region:
        return Region.make_halfspace(self, +1)

    def __neg__(self) -> Region:
        return Region.make_halfspace(self, -1)

    # ==================================================================================
    # Surface motion
    # ==================================================================================

    def move(self, velocities: ArrayLike, durations: ArrayLike) -> None:
        """Define piecewise-constant translation of the surface.

        Parameters
        ----------
        velocities : array_like of float, shape (N, 3)
            Velocity vector ``[vx, vy, vz]`` in cm/s for each motion segment.
        durations : array_like of float, shape (N,)
            Duration of each motion segment in seconds, starting at simulation
            time zero. Must contain the same number of entries as ``velocities``.

        Notes
        -----
        A final stationary segment is appended automatically for the remainder
        of the simulation. The supplied velocities and durations are not modified.
        Calling this method again replaces the previous motion schedule.
        """
        move_object(self, velocities, durations)


def _get_cone_t_sq(half_angle: float) -> float:
    """Convert a cone half-angle in degrees to its squared tangent."""

    if not 0.0 < half_angle < 90.0:
        print_error("Cone half-angle must be greater than 0 and less than 90 degrees.")
    return math.tan(math.radians(half_angle)) ** 2


def _get_cone_half_angle(t_sq: float) -> float:
    """Convert a cone's squared tangent to its half-angle in degrees."""

    return math.degrees(math.atan(math.sqrt(t_sq)))


def decode_BC_type(type_):
    """Return the display name for a packed boundary-condition code."""

    if type_ == BC_NONE:
        return "None"
    elif type_ == BC_VACUUM:
        return "Vacuum"
    elif type_ == BC_REFLECTIVE:
        return "Reflective"
