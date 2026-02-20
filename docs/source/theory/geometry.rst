.. _geometry:

======================
Geometry and CSG Model
======================

MC/DC uses **Constructive Solid Geometry** (CSG) to define the spatial
domain.  Complex geometries are built by combining simple surfaces with
Boolean region operators.

Surfaces
--------

A surface is a mathematical equation that divides space into a positive
half-space (:math:`f(\mathbf{r}) > 0`) and a negative half-space
(:math:`f(\mathbf{r}) < 0`).  MC/DC provides the following surface types:

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Surface
     - Equation
     - Constructor
   * - Plane X
     - :math:`x - x_0 = 0`
     - ``Surface.PlaneX(x=x0)``
   * - Plane Y
     - :math:`y - y_0 = 0`
     - ``Surface.PlaneY(y=y0)``
   * - Plane Z
     - :math:`z - z_0 = 0`
     - ``Surface.PlaneZ(z=z0)``
   * - Cylinder X
     - :math:`(y-y_0)^2 + (z-z_0)^2 - R^2 = 0`
     - ``Surface.CylinderX(center, radius)``
   * - Cylinder Y
     - :math:`(x-x_0)^2 + (z-z_0)^2 - R^2 = 0`
     - ``Surface.CylinderY(center, radius)``
   * - Cylinder Z
     - :math:`(x-x_0)^2 + (y-y_0)^2 - R^2 = 0`
     - ``Surface.CylinderZ(center, radius)``
   * - Sphere
     - :math:`|\mathbf{r} - \mathbf{r}_0|^2 - R^2 = 0`
     - ``Surface.Sphere(center, radius)``

**Boundary conditions** are set on outermost surfaces:

.. code-block:: python3

   s = mcdc.Surface.PlaneX(x=10.0, boundary_condition="vacuum")     # particles leak
   s = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")  # mirror reflection


Region Operators
-----------------

A **region** (also called a *half-space*) is obtained by applying the
unary ``+`` or ``-`` operator to a surface:

- ``+s`` — the positive half-space (:math:`f > 0`).
- ``-s`` — the negative half-space (:math:`f < 0`).

Regions are combined with Boolean operators:

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Operator
     - Meaning
     - Example
   * - ``&``
     - Intersection (AND)
     - ``+s1 & -s2`` — between planes ``s1`` and ``s2``
   * - ``|``
     - Union (OR)
     - ``region_A | region_B`` — either region
   * - ``~``
     - Complement (NOT)
     - ``~region_A`` — everything outside ``region_A``

**Operator precedence**: ``~`` > ``&`` > ``|``.  Use parentheses for
clarity.


Cells
------

A **cell** is a region filled with a material:

.. code-block:: python3

   mcdc.Cell(region=+s1 & -s2 & +s3 & -s4, fill=material)

Every point in the problem domain must belong to exactly one cell.
MC/DC does not verify non-overlapping coverage automatically — the user
is responsible for ensuring consistent cell definitions.

**Named cells** can be used for cell-based tallies:

.. code-block:: python3

   sphere_cell = mcdc.Cell(name="Fuel sphere", region=-sphere, fill=fuel)
   mcdc.TallyCell(cell=sphere_cell, scores=["fission"])


Universes and Packaging
------------------------

A **universe** groups a set of cells into a reusable geometry unit.
Universes can be **translated** and **rotated** when placed inside a
container cell, enabling duplication of complex assemblies without
redefining their internal geometry.

.. code-block:: python3

   assembly = mcdc.Universe(cells=[fuel_cell, clad_cell, water_cell])

   # Place two copies with different positions and rotations
   mcdc.Cell(region=left_region,  fill=assembly, translation=[-5, 0, 0])
   mcdc.Cell(region=right_region, fill=assembly, translation=[+5, 0, 0],
             rotation=[0, 10, 0])

When a particle enters a universe cell, MC/DC transforms coordinates
into the universe's local frame, tracks through local surfaces, then
transforms back.

For a complete example, see :ref:`example_fuel_array_packaged`.


Lattices
---------

A **lattice** is a regular array of universes arranged on a Cartesian
grid.  Each grid position is mapped to a universe by an integer index.

.. code-block:: python3

   lattice = mcdc.Lattice(
       x=[-5.0, 0.0, 5.0],   # 2 cells in x
       y=[-5.0, 0.0, 5.0],   # 2 cells in y
       universes=[[u1, u2],
                  [u3, u4]],
   )

Lattices are powerful for reactor-core models where fuel assemblies
repeat in a regular pattern.  See the :ref:`example_c5g7_k_eigenvalue`
for a full-core lattice example.

Root Universe
^^^^^^^^^^^^^

When using universes or lattices, the top-level cell collection is
registered as the **root universe**:

.. code-block:: python3

   mcdc.simulation.set_root_universe(cells=[cell_left, cell_right])


Geometry Visualization
-----------------------

MC/DC includes a built-in ray-casting visualizer for inspecting CSG
geometries before running a full transport simulation:

.. code-block:: python3

   mcdc.visualize(
       "xz",                         # projection plane
       y=0.0,                        # slice position
       x=[-10.0, 10.0],              # plot range
       z=[-5.0, 5.0],
       pixels=(400, 400),
       colors={fuel: "red", water: "blue"},
   )

This produces a pixel map showing which material fills each pixel.
Animated geometry (moving surfaces) can be visualized with the ``time``
argument:

.. code-block:: python3

   mcdc.visualize(..., time=np.linspace(0, 9, 19), save_as="geo_animation")

For details on moving surfaces and sources, see :ref:`cont_movement`.


Moving Surfaces
----------------

Any surface can be given a piecewise-constant velocity using the
``move()`` method, enabling time-dependent geometry:

.. code-block:: python3

   surface.move(velocities=[[vx, vy, vz]], durations=[dt])

MC/DC solves for the exact intersection of a particle trajectory with
the moving surface — no time-step discretization error is introduced.
For the mathematical formulation, see :ref:`cont_movement`.
