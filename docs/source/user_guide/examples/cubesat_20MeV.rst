.. _example_cubesat_20MeV:

=============================================
20 MeV Neutron CubeSat
=============================================

Problem Description
===================

A continuous-energy neutron shielding problem for a simplified 1U CubeSat.
Neutrons from a space-environment spectrum enter the satellite from all
sides, and the current into four electronics sensitive volumes (SVs) is
tallied in energy and angle.  This example demonstrates MC/DC's
**continuous-energy materials**, **tabulated boundary sources**, and
**surface-and-cell current tallies**.

Geometry and Materials
======================

The CubeSat occupies :math:`x,y \in [0,10]` cm with rails spanning
:math:`z \in [0,11]` cm.  A 15 cm vacuum-bounded cube centred at
:math:`(5,5,5)` cm surrounds it.  All 40 parts are rectangular boxes:

.. list-table::
   :widths: 30 10 25

   * - **Part**
     - **Count**
     - **Material**
   * - Main and small rails
     - 12
     - Al7075
   * - Shear panels
     - 6
     - Al6061
   * - Solar panels
     - 10
     - Silicon
   * - Stowed antenna strips
     - 4
     - Copper
   * - Circuit boards
     - 4
     - FR4 proxy
   * - SVs (OBC, EPS, ADCS, Comms)
     - 4
     - Silicon, LiCoO\ :sub:`2`, Silicon, Silicon

Each SV sits directly on top of its board.  The remaining space is void.

.. list-table:: Material compositions
   :widths: 20 15 45

   * - **Material**
     - **Density (g/cm³)**
     - **Composition (wt%)**
   * - Al7075
     - 2.81
     - 90 Al-27, 5 Zn-64, 3 Mg-24, 2 Cu-63
   * - Al6061
     - 2.70
     - 98 Al-27, 1.2 Mg-24, 0.8 Si-28
   * - FR4 proxy
     - 1.85
     - 28.0 Si-28, 51.9 O-16, 18.3 C-12, 1.8 H-1
   * - Silicon
     - 2.329
     - 100 Si-28
   * - LiCoO\ :sub:`2` proxy
     - 5.05
     - 6.661 Li-7, 60.551 Co-59, 32.788 O-16
   * - Copper
     - 8.96
     - 68.4792 Cu-63, 31.5208 Cu-65

Nuclear data are ENDF/B-VIII.0 at 293.6 K.

The folder also contains ``Cubesat_Geometry_CAD.step``, a CAD model of the
same 40 parts, and a ``README.md`` describing how to convert it to MC/DC
geometry with GEOUNED.

Physical Assumptions
====================

* Continuous-energy neutron transport only; no secondary photons.
* Steady state (no time dependence).
* Source spectrum: ECSS neutron spectrum from 1 eV to 20 MeV.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Source**
     - Six inward white (cosine-weighted) sources, one on each face of
       the 15 cm cube, each with probability 1/6
   * - **Tally type**
     - Surface tally on each of the six faces of each SV, filtered by the SV cell
   * - **Tally scores**
     - ``current-in`` and ``current-out``
   * - **Tally grids**
     - 8 polar-cosine bins, 8 azimuthal bins, 69 logarithmic energy bins
       from :math:`10^{-5}` eV to 20 MeV
   * - **Source particles**
     - :math:`5 \times 10^{7}`

Quantities of Interest
======================

* Neutron current entering and leaving each SV, per source neutron.
* Incoming energy spectrum at each SV.

Reference Solution
==================

No reference solution.

Step-by-Step Walkthrough
========================

**1. Materials (lines 28–95)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 28-95
   :linenos:
   :lineno-match:

Each material is defined by nuclide number densities in atoms/barn-cm at
the library temperature.  The void cell uses a material with zero density.

**2. Box Helpers and Outer Boundary (lines 97–185)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 97-185
   :linenos:
   :lineno-match:

``box`` returns the region inside six planes.  ``stacked_board_sv`` builds a
board and its SV with a shared plane between them, and returns the SV's six
bounding surfaces for the current tallies.

**3. CubeSat Components (lines 187–333)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 187-333
   :linenos:
   :lineno-match:

Rails, shear panels, solar panels, antenna strips and the board stack, each
as a list of box cells.

**4. Void Fill (lines 335–354)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 335-354
   :linenos:
   :lineno-match:

The void cell is the outer cube minus every component region.

**5. Source (lines 356–433)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 356-433
   :linenos:
   :lineno-match:

The energy spectrum is read from ``ecss_20MeV.csv`` and passed as a
tabulated PDF.  Each face source is set just inside the vacuum boundary
with ``white_direction`` pointing inward.

**6. Tallies (lines 435–455)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 435-455
   :linenos:
   :lineno-match:

Each SV face gets a surface tally filtered by the SV cell, so ``current-in``
counts neutrons entering that SV through that face.  Summing the six faces
gives the SV total.

**7. Settings and Run (lines 458–471)**

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
   :language: python
   :lines: 458-471
   :linenos:
   :lineno-match:

**What to try:**

- Replace the per-face tallies with one cell tally per SV using
  ``mcdc.Tally(cell=..., scores=["current-in", "current-out"])``.
- Add a ``flux`` cell tally on each SV.
- Thicken the shear panels and compare the SV spectra.

Full Input
==========

Click here to view the input file: `examples/20MeV_Cubesat/input.py <https://github.com/mcdc-project/mcdc/blob/dev/examples/20MeV_Cubesat/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../../examples/20MeV_Cubesat/input.py
  :language: python
  :linenos:

How to Run
==========

Set ``MCDC_LIB`` to a directory holding ``neutron-{nuclide}-293.6K.h5`` for
H-1, Li-7, C-12, O-16, Mg-24, Al-27, Si-28, Co-59, Cu-63, Cu-65 and Zn-64,
prepared with ``tools/data_library_generator/neutron``.  Then, from inside
``examples/20MeV_Cubesat``, run::

  python input.py --mode=numba
  python process-output.py

The default :math:`5 \times 10^{7}` histories are meant for a parallel run,
for example ``mpiexec -n 32 python input.py --mode=numba``.  Use
``--N_particle`` for a quicker test.

Expected Output
===============

``cubesat_20MeV.h5`` with the SV current tallies.  ``process-output.py``
prints the total incoming current for each SV and saves
``cubesat_20MeV_sv_spectra.png``, the incoming energy spectrum of each SV.
