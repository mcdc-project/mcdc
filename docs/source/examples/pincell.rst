.. _example_pincell:

=============================================
Continuous-Energy Pin Cell
=============================================

Problem Description
===================

A single light-water reactor fuel pin (pin cell) modelled with
continuous-energy nuclear data.  The problem solves the k-eigenvalue
equation to determine the neutron multiplication factor :math:`k_{\text{eff}}`
and the energy spectrum of the scalar flux :math:`\phi(E)`.

This example demonstrates MC/DC's continuous-energy transport capability
with nuclide-level material definitions.

Geometry and Materials
======================

A two-dimensional infinite lattice is represented by a single pin cell
with reflective boundary conditions on all four sides.

.. list-table::
   :widths: 30 40

   * - **Fuel radius**
     - 0.45720 cm
   * - **Pin pitch**
     - 1.25984 cm
   * - **Fuel composition (UO₂)**
     - U-235 (5.58 × 10\ :sup:`-4`), U-238 (2.240 × 10\ :sup:`-2`),
       O-16 (4.583 × 10\ :sup:`-2`)
   * - **Moderator composition (H₂O + B)**
     - H-1 (6.85 × 10\ :sup:`-2`), O-16 (3.279 × 10\ :sup:`-2`),
       B-10 (1.36 × 10\ :sup:`-4`)

Physical Assumptions
====================

* Continuous-energy neutron transport.
* Isotropic scattering in the centre-of-mass frame (converted to lab frame
  during tracking).
* Infinite axial extent (2-D geometry via reflective boundaries).
* k-eigenvalue (criticality) calculation.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Energy tally grid**
     - 362 bin boundaries (SHEM-361 structure, 0 eV – 19.6 MeV)
   * - **Source particles per cycle**
     - :math:`10^{2}` (demonstration; increase for production runs)
   * - **Inactive cycles**
     - 10
   * - **Active cycles**
     - 50
   * - **Source energy**
     - 1 MeV isotropic

Quantities of Interest
======================

* Effective multiplication factor :math:`k_{\text{eff}}`.
* Neutron flux energy spectrum :math:`E\,\phi(E)`.
* Neutron density as a function of time (when run in transient mode).

Reference Solution
==================

Continuous-energy pin-cell :math:`k_{\text{eff}}` values from established
Monte Carlo codes (e.g., OpenMC, MCNP) can be used for cross-verification.

References
==========

The SHEM-361 energy group structure is documented in the
`OpenMC documentation <https://docs.openmc.org/en/latest/pythonapi/mgxs.html>`__.

Input
=====

Click here to view the input file: `examples/pincell/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/pincell/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/pincell/input.py
  :language: python
  :linenos:

How to Run
==========

From the repository root run::

  python examples/pincell/input.py

Expected Output
===============

Global flux tallies and an energy-spectrum file using the energy grid in
``examples/pincell/energy_grid.txt``.
