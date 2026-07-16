.. _example_sphere_in_cube:

=============================================
Sphere-in-Cube Fission Detector
=============================================

Problem Description
===================

A three-dimensional time-dependent problem with a homogeneous fissile
sphere embedded inside a scattering cube.  A cell-based tally records
the fission rate inside the sphere, demonstrating MC/DC's **cell tally**
functionality.

Geometry and Materials
======================

The computational domain is a cube:
:math:`x,y,z \in [0,4]` cm, with vacuum boundary conditions.

A sphere of radius 1.5 cm is centred at :math:`(2,2,2)` cm.

.. list-table:: Cross-section data (mono-energetic, cm\ :sup:`-1`)
   :widths: 25 15 15 15

   * - **Region**
     - :math:`\Sigma_s`
     - :math:`\Sigma_f`
     - :math:`\nu`
   * - Cube (outside sphere)
     - 1.0
     - —
     - —
   * - Sphere (fissile)
     - —
     - 1.0
     - 1.2

Physical Assumptions
====================

* Mono-energetic (one-speed) neutron transport.
* Isotropic scattering (cube) and isotropic fission (sphere).
* Time-dependent transport with a uniform isotropic source,
  :math:`t \in [0,50]` s.
* Implicit capture variance-reduction technique.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Tally type**
     - Cell tally on the spherical region
   * - **Tally score**
     - Fission rate
   * - **Source particles**
     - :math:`10^{3}` (demonstration)
   * - **Batches**
     - 2

Quantities of Interest
======================

* Volume-integrated fission rate inside the sphere.
* Statistical uncertainty (standard deviation) of the cell tally.

Reference Solution
==================

The problem can be verified analytically for simple cross-section
combinations using first-flight collision probabilities.

Step-by-Step Walkthrough
========================

**1. Import and Materials (lines 1–12)**

.. literalinclude:: ../../../examples/sphere_in_cube/input.py
   :language: python
   :lines: 1-12
   :linenos:
   :lineno-match:

Two mono-energetic materials: a purely fissile material (``pure_f``,
:math:`\Sigma_f = 1.0`, :math:`\nu = 1.2`) for the sphere, and a purely
scattering material (``pure_s``, :math:`\Sigma_s = 1.0`) for the cube.

**2. Surfaces and CSG Regions (lines 14–26)**

.. literalinclude:: ../../../examples/sphere_in_cube/input.py
   :language: python
   :lines: 14-26
   :linenos:
   :lineno-match:

Six planes define the cube, and a ``Sphere`` surface defines the
detector region.  The ``~`` (complement) operator carves out the
sphere from the cube.

**3. Source (lines 32–39)**

.. literalinclude:: ../../../examples/sphere_in_cube/input.py
   :language: python
   :lines: 32-39
   :linenos:
   :lineno-match:

A uniform isotropic source fills the cube over :math:`t \in [0,50]` s.

**4. Cell Tally, Settings, and Run (lines 45–55)**

.. literalinclude:: ../../../examples/sphere_in_cube/input.py
   :language: python
   :lines: 45-55
   :linenos:
   :lineno-match:

This example uses cell-filtered ``Tally`` — it tallies fission events inside a
specific cell (the sphere) rather than on a spatial mesh.
Implicit capture is enabled to keep particles alive longer.

**What to try:**

- Replace the cell filter with a mesh filter to visualise the 3-D flux.
- Change the sphere radius or :math:`\nu` to see how fission rate changes.
- Add a time grid to the cell tally for time-resolved data.

Full Input
==========

Click here to view the input file: `examples/sphere_in_cube/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/sphere_in_cube/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/sphere_in_cube/input.py
  :language: python
  :linenos:

How to Run
==========

From the repository root run::

  python examples/sphere_in_cube/input.py

Expected Output
===============

Volume-integrated fission rate time series saved by the tally and a
small printed summary from the example's ``process-output.py`` script.
