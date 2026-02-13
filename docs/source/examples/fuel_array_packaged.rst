.. _example_fuel_array_packaged:

=============================================
Packaged Fuel Array (Universe and Lattice)
=============================================

Problem Description
===================

A three-dimensional fixed-source problem featuring two identical fuel
assemblies placed side-by-side inside a water-filled box.  Each
assembly uses a composite "shooting-star" fuel geometry built from the
union of two orthogonal cylinders enclosed by a cladding sphere.

This example demonstrates MC/DC's **universe**, **translation**, and
**rotation** capabilities for constructive solid geometry (CSG)
packaging.

Geometry and Materials
======================

The global domain is a rectangular box:
:math:`x \in [-10,10]`, :math:`y \in [-5,5]`, :math:`z \in [-5,5]` cm,
with vacuum boundary conditions.

Each assembly is defined as a universe containing three cells:

1. **Fuel** — union of a z-aligned and an x-aligned cylinder
   (radius 1 cm, half-length 5 cm).
2. **Cladding** — spherical shell (radius 3 cm) surrounding the fuel.
3. **Water** — region outside the cladding sphere.

The left assembly is translated to :math:`(-5,0,0)` cm;
the right assembly is translated to :math:`(+5,0,0)` cm and rotated
:math:`10°` about the :math:`y`-axis.

.. list-table:: Cross-section data (mono-energetic, cm\ :sup:`-1`)
   :widths: 20 15 15 15 15

   * - **Region**
     - :math:`\Sigma_c`
     - :math:`\Sigma_s`
     - :math:`\Sigma_f`
     - :math:`\nu`
   * - Fuel
     - 0.45
     - —
     - 0.55
     - 2.5
   * - Cladding
     - 0.05
     - 0.95
     - —
     - —
   * - Water
     - 0.02
     - 0.08
     - —
     - —

Physical Assumptions
====================

* Mono-energetic (one-speed) neutron transport.
* Isotropic scattering.
* Steady-state fixed-source calculation.
* No delayed neutrons.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Spatial mesh (tally)**
     - :math:`201 \times 101` in the :math:`(x,z)`-plane
   * - **Tally score**
     - Fission rate
   * - **Source particles**
     - :math:`10^{3}` (demonstration)
   * - **Batches**
     - 2

Quantities of Interest
======================

* Two-dimensional fission rate distribution in the :math:`(x,z)`-plane.
* Relative standard deviation map for convergence assessment.

Reference Solution
==================

No analytical reference.  The geometry can be verified using MC/DC's
built-in ``mcdc.visualize()`` function to render the CSG model.

Input
=====

Click here to view the input file: `examples/fuel_array_packaged/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/fuel_array_packaged/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/fuel_array_packaged/input.py
  :language: python
  :linenos:

How to Run
==========

From the repository root run::

  python examples/fuel_array_packaged/input.py

Expected Output
===============

An HDF5 mesh tally and optional visualization images produced by the
``mcdc.visualize()`` helper when run with visualization enabled.
