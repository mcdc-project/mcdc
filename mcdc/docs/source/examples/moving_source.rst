.. _example_moving_source:

=============================================
Moving Neutron Source
=============================================

Problem Description
===================

A time-dependent problem in which an isotropic neutron source moves
through a three-dimensional box of air-like material along a prescribed
piecewise-linear trajectory.  This example demonstrates MC/DC's
**moving-source** capability.

Geometry and Materials
======================

The computational domain is a rectangular box:
:math:`x \in [-5,5]`, :math:`y \in [-5,5]`, :math:`z \in [-10,10]` cm,
with vacuum boundary conditions on all faces.

A single homogeneous material (air analogue) fills the entire domain:

.. list-table:: Cross-section data (mono-energetic, cm\ :sup:`-1`)
   :widths: 30 20 20

   * - **Region**
     - :math:`\Sigma_c`
     - :math:`\Sigma_s`
   * - Air
     - 0.002
     - 0.008

Neutron speed: :math:`v = 2 \times 10^{5}` cm/s.

Physical Assumptions
====================

* Mono-energetic (one-speed) neutron transport.
* Isotropic scattering.
* Time-dependent transport with a moving point-like source.
* No fission.

Numerical Setup
===============

The source moves in three consecutive phases:

.. list-table::
   :widths: 15 30 15

   * - **Phase**
     - **Velocity (cm/s)**
     - **Duration (s)**
   * - 1
     - :math:`(1, 0, 0)`
     - 7
   * - 2
     - :math:`(-0.5,\; 2,\; 0)`
     - 2
   * - 3
     - :math:`(0,\; -3,\; 0)`
     - 1

.. list-table::
   :widths: 35 65

   * - **Spatial mesh (tally)**
     - :math:`201 \times 201` in the :math:`(x,y)`-plane
   * - **Time mesh (tally)**
     - 46 equally spaced bins over :math:`t \in [0,10]` s
   * - **Tally score**
     - Scalar flux
   * - **Source particles**
     - :math:`10^{5}`
   * - **Batches**
     - 2

Quantities of Interest
======================

* Time-resolved 2-D flux distribution :math:`\phi(x,y,t)`.
* Animated GIF of the neutron cloud following the moving source.

Reference Solution
==================

No analytical reference.  The solution is validated qualitatively by
confirming that the flux maximum tracks the prescribed source trajectory.

Step-by-Step Walkthrough
========================

**1. Materials (lines 1–15)**

.. literalinclude:: ../../../examples/moving_source/input.py
   :language: python
   :lines: 1-15
   :linenos:
   :lineno-match:

A single air-like material with ``speed`` defined for time-dependent
transport.

**2. Geometry (lines 17–27)**

.. literalinclude:: ../../../examples/moving_source/input.py
   :language: python
   :lines: 17-27
   :linenos:
   :lineno-match:

A simple box with vacuum boundaries.  One cell fills the entire domain.

**3. Moving Source (lines 33–49)**

.. literalinclude:: ../../../examples/moving_source/input.py
   :language: python
   :lines: 33-49
   :linenos:
   :lineno-match:

The source is created with spatial and angular extent, then ``src.move()``
assigns a piecewise-linear trajectory: three velocity segments with their
durations.  The source physically translates through the domain over time.

**4. Tallies, Settings, and Run (lines 55–65)**

.. literalinclude:: ../../../examples/moving_source/input.py
   :language: python
   :lines: 55-65
   :linenos:
   :lineno-match:

A structured :math:`201 \times 201` mesh tally with 46 time bins captures
the evolving 2-D flux.  The companion ``process-output.py`` script
generates an animated GIF.

**What to try:**

- Change the velocity vectors to create a circular or zigzag path.
- Add more time-resolution bins for smoother animation.
- Compare with ``moving_pellet`` where the geometry moves instead of the source.

Full Input
==========

Click here to view the input file: `examples/moving_source/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/moving_source/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/moving_source/input.py
  :language: python
  :linenos:

How to Run
==========

From the repository root run::

  python examples/moving_source/input.py

Expected Output
===============

An HDF5 mesh tally with time-resolved 2-D flux slices and a GIF animation
produced by the example's post-processing script.
