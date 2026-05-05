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

Step-by-Step Walkthrough
========================

**1. Materials (lines 1–27)**

.. literalinclude:: ../../../examples/fuel_array_packaged/input.py
   :language: python
   :lines: 1-27
   :linenos:
   :lineno-match:

Three mono-energetic materials: fissile fuel, a scattering cladding, and
water moderator.

**2. Assembly Geometry — Shooting-Star CSG (lines 29–54)**

.. literalinclude:: ../../../examples/fuel_array_packaged/input.py
   :language: python
   :lines: 29-54
   :linenos:
   :lineno-match:

The fuel region is the **union** of a z-cylinder and an x-cylinder (the
"shooting star").  The cladding fills the sphere minus the fuel.
Water fills outside the sphere.  These three cells form a reusable
**universe**.

**3. Packaging with Universe, Translation, and Rotation (lines 56–80)**

.. literalinclude:: ../../../examples/fuel_array_packaged/input.py
   :language: python
   :lines: 56-80
   :linenos:
   :lineno-match:

The assembly universe is placed twice using ``mcdc.Cell(..., fill=assembly)``:

- **Left** — translated to :math:`(-5, 0, 0)`.
- **Right** — translated to :math:`(+5, 0, 0)` and rotated 10° about :math:`y`.

``set_root_universe()`` tells MC/DC these are the top-level cells.

**4. Source, Tallies, Settings, and Run (lines 82–105)**

.. literalinclude:: ../../../examples/fuel_array_packaged/input.py
   :language: python
   :lines: 82-105
   :linenos:
   :lineno-match:

A point-like source near the centre, a structured mesh tally for the
:math:`(x,z)`-plane fission rate, and 1 000 particles in 2 batches.
The ``active_bank_buffer`` accommodates fission-born particles.

**5. Optional Visualization (lines 107–end)**

.. literalinclude:: ../../../examples/fuel_array_packaged/input.py
   :language: python
   :lines: 107-
   :linenos:
   :lineno-match:

Set ``visualize = True`` to render the CSG geometry with
``mcdc.visualize()`` instead of running the transport.

**What to try:**

- Change the rotation angle and observe the effect on the fission map.
- Add a third assembly copy with a different translation.
- Use ``mcdc.Lattice`` instead of manual universe placement.

Full Input
==========

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
