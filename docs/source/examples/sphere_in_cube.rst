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

Input
=====

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
