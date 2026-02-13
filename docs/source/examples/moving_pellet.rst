.. _example_moving_pellet:

=============================================
Moving Fuel Pellet
=============================================

Problem Description
===================

A three-dimensional time-dependent problem in which a cylindrical fuel
pellet traverses a box of air-like material along a piecewise-linear
trajectory.  Both the cylindrical surface and the bounding planes of the
pellet move, demonstrating MC/DC's **moving-surface** capability for
transient geometry.

Geometry and Materials
======================

The computational domain is a rectangular box:
:math:`x \in [-5,5]`, :math:`y \in [-5,5]`, :math:`z \in [-10,10]` cm,
with vacuum boundary conditions on all faces.

A z-aligned cylindrical fuel pellet (radius 1 cm) is initially located
at :math:`z \in [6,9]` cm.  The pellet moves in three consecutive phases:

.. list-table::
   :widths: 15 40 40 15

   * - **Phase**
     - **Cylinder velocity**
     - **End-cap velocity**
     - **Duration (s)**
   * - 1
     - :math:`(-0.5,\; 0,\; 0)`
     - :math:`(0,\; 0,\; -2)`
     - 2 / 5
   * - 2
     - :math:`(1,\; 0,\; 0)`
     - :math:`(0,\; 0,\; 4)`
     - 5 / 2
   * - 3
     - :math:`(-2,\; 0,\; 0)`
     - :math:`(0,\; 0,\; -10)`
     - 1 / 1

Two materials are used:

.. list-table:: Cross-section data (mono-energetic, cm\ :sup:`-1`)
   :widths: 20 15 15 15 15

   * - **Region**
     - :math:`\Sigma_c`
     - :math:`\Sigma_s`
     - :math:`\Sigma_f`
     - :math:`\nu`
   * - Fuel pellet
     - 0.50
     - —
     - 0.25
     - 1.5
   * - Air
     - 0.002
     - 0.008
     - —
     - —

Neutron speed: :math:`v = 2 \times 10^{5}` cm/s (both regions).

Physical Assumptions
====================

* Mono-energetic (one-speed) neutron transport.
* Isotropic scattering.
* Time-dependent transport with moving geometry surfaces.
* Fission in the pellet region only.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Spatial mesh (tally)**
     - :math:`201 \times 201` in the :math:`(x,z)`-plane
   * - **Time mesh (tally)**
     - 46 equally spaced bins over :math:`t \in [0,9]` s
   * - **Tally score**
     - Fission rate
   * - **Source particles**
     - :math:`10^{5}`
   * - **Batches**
     - 2

Quantities of Interest
======================

* Time-resolved 2-D fission rate distribution in the :math:`(x,z)`-plane.
* Animation of the fission rate tracking the moving pellet geometry.

Reference Solution
==================

No analytical reference.  The solution is validated by verifying that
the fission rate follows the pellet trajectory and that particle
conservation is maintained.

Input
=====

Click here to view the input file: `examples/moving_pellet/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/moving_pellet/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/moving_pellet/input.py
  :language: python
  :linenos:

How to Run
==========

From the repository root run::

  python examples/moving_pellet/input.py

Expected Output
===============

An HDF5 tally file with time-resolved fission rates and an optional
animation created by the example's post-processing script.
