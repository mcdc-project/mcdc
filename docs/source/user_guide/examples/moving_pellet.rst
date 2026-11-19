.. _example_moving_pellet:

=============================================
Moving Fuel Pellet
=============================================

Problem Description
===================

A three-dimensional time-dependent problem in which a cylindrical fuel
pellet traverses a box of air-like material along a piecewise-linear
trajectory.  Both the cylindrical surface and the bounding planes of the
pellet move. Delayed-neutron precursors follow the same translation through
``fuel_cell.move_product(velocities, durations)``, demonstrating moving
surfaces and moving delayed products together.

Geometry and Materials
======================

The computational domain is a rectangular box:
:math:`x \in [-5,5]`, :math:`y \in [-5,5]`, :math:`z \in [-10,10]` cm,
with vacuum boundary conditions on all faces.

A z-aligned cylindrical fuel pellet (radius 1 cm) is initially located
at :math:`z \in [6,9]` cm. The surfaces and fuel products share four consecutive
translation segments, followed by stationary motion from 8 to 9 s:

.. list-table::
   :widths: 15 60 25

   * - **Phase**
     - **Pellet velocity (cm/s)**
     - **Duration (s)**
   * - 1
     - :math:`(-0.5,\; 0,\; -2)`
     - 2
   * - 2
     - :math:`(1,\; 0,\; -2)`
     - 3
   * - 3
     - :math:`(1,\; 0,\; 4)`
     - 2
   * - 4
     - :math:`(-2,\; 0,\; -10)`
     - 1

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
The fuel's total yield of 1.5 consists of :math:`\nu_p = 1.0` prompt neutrons
and :math:`\nu_d = 0.5` delayed neutrons per fission. One illustrative delayed
group has decay constant :math:`\lambda = 1\ \mathrm{s}^{-1}` (mean lifetime
1 s). These are demonstration data rather than an isotope-specific model.

Physical Assumptions
====================

* Mono-energetic (one-speed) neutron transport.
* Isotropic scattering.
* Time-dependent transport with moving geometry surfaces.
* Fission in the pellet region only.
* Delayed-neutron precursors remain attached to the translating fuel until
  emission. Prompt neutrons are emitted at the fission position.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Spatial mesh (tally)**
     - :math:`200 \times 200` bins in the :math:`(x,z)`-plane
   * - **Time mesh (tally)**
     - 45 equally spaced bins over :math:`t \in [0,9]` s
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

No analytical reference is provided. The regression compares the fission
tally with a stored result, including the effect of moving delayed products.

Step-by-Step Walkthrough
========================

**1. Materials**

.. literalinclude:: ../../../../examples/moving_pellet/input.py
   :language: python
   :start-at: # Set materials
   :end-before: # Set surfaces
   :linenos:
   :lineno-match:

A fissile fuel pellet (:math:`\Sigma_f = 0.25`, :math:`\nu_p = 1.0`,
:math:`\nu_d = 0.5`) and an air-like background. Both include ``speed`` for
time-dependent transport. The delayed group also specifies ``decay_rate``.

**2. Surfaces and Moving Geometry**

.. literalinclude:: ../../../../examples/moving_pellet/input.py
   :language: python
   :start-at: # Set surfaces
   :end-before: # Set container cell surfaces
   :linenos:
   :lineno-match:

A z-cylinder and two z-planes define the pellet.  The key feature:
``surface.move(velocities, durations)`` makes these surfaces **time-dependent**.
The shared velocity vectors combine lateral and axial motion. Translation
along the cylinder's axis or parallel to an endcap leaves that surface's
equation unchanged, so all three surfaces can use the same schedule.

**3. Container, Cells, and Delayed Products**

.. literalinclude:: ../../../../examples/moving_pellet/input.py
   :language: python
   :start-at: # Set container cell surfaces
   :end-at: simulation.set_model([fuel_cell, air_cell])
   :linenos:
   :lineno-match:

The fuel pellet region is defined by the intersection of the cylinder
and the two planes.  The air fills the complement inside the bounding box.
The fuel cell's ``move_product`` call uses the same schedule as the surfaces.
For a fission at time :math:`t_b` and delayed emission at :math:`t_e`, the
emission position receives the displacement :math:`T(t_e) - T(t_b)`, where
:math:`T(t)` is the cumulative pellet translation.

**4. Source**

.. literalinclude:: ../../../../examples/moving_pellet/input.py
   :language: python
   :start-at: source = mcdc.Source(
   :end-at: simulation.set_sources([source])
   :linenos:
   :lineno-match:

A small box source near the pellet's trajectory, active over the
full simulation time :math:`t \in [0, 9]` s.

**5. Tallies and Settings**

.. literalinclude:: ../../../../examples/moving_pellet/input.py
   :language: python
   :start-at: # Tallies
   :end-at: simulation.settings.active_bank_buffer = 1000
   :linenos:
   :lineno-match:

A structured mesh tally in the :math:`(x,z)`-plane with 45 time bins
captures the fission rate as the pellet moves.

**What to try:**

- Change the pellet velocities to create different trajectories.
- Remove ``fuel_cell.move_product(...)`` to compare stationary precursor
  emission positions with precursors following the fuel.
- Set ``visualize = True`` to watch the geometry evolve with
  ``simulation.visualize_model(..., time=...)``.
- Compare with ``moving_source`` to see source motion vs. geometry motion.

Full Input
==========

Click here to view the input file: `examples/moving_pellet/input.py <https://github.com/mcdc-project/mcdc/blob/dev/examples/moving_pellet/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../../examples/moving_pellet/input.py
  :language: python
  :linenos:

How to Run
==========

From inside ``examples/moving_pellet`` run::

  python input.py

Expected Output
===============

An HDF5 tally file with time-resolved fission rates and an optional
animation created by the example's post-processing script.
