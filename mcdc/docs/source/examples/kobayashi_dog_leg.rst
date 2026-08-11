.. _example_kobayashi_dog_leg:

=============================================
Kobayashi Dog-Leg Void Benchmark
=============================================

Problem Description
===================

A mono-energetic, three-dimensional shielding benchmark featuring a
dog-leg vacuum channel embedded in a purely scattering/absorbing shield.
The problem evaluates the ability of a Monte Carlo code to transport
neutrons through deep-penetration streaming paths.

It is based on the NEA steady-state fixed-source benchmark problem suite
by Kobayashi *et al.* [Kobayashi2001]_.

Geometry and Materials
======================

The computational domain spans
:math:`x \in [0,60]`, :math:`y \in [0,100]`, :math:`z \in [0,60]` cm.
Boundary conditions are reflective on the three symmetry planes
(:math:`x=0`, :math:`y=0`, :math:`z=0`) and vacuum elsewhere.

Three regions are defined:

1. **Source region** — :math:`x \in [0,10]`, :math:`y \in [0,10]`,
   :math:`z \in [0,10]` cm.
2. **Dog-leg void channel** — an L-shaped duct connecting the source
   corner to the far side of the domain.
3. **Shield** — the remaining volume.

.. list-table:: Cross-section data (mono-energetic, cm\ :sup:`-1`)
   :widths: 30 20 20

   * - **Region**
     - :math:`\Sigma_c`
     - :math:`\Sigma_s`
   * - Shield
     - 0.05
     - 0.05
   * - Void channel
     - :math:`5\times10^{-5}`
     - :math:`5\times10^{-5}`

Physical Assumptions
====================

* Mono-energetic (one-speed) neutron transport.
* Isotropic scattering.
* Steady-state (time-independent) fixed-source problem.
* Implicit capture variance-reduction technique.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Spatial mesh (tally)**
     - :math:`60 \times 100 \times 60` uniform cells (1 cm spacing)
   * - **Tally score**
     - Scalar flux
   * - **Source particles**
     - :math:`10^{3}` (demonstration; increase for production runs)
   * - **Batches**
     - 2

Quantities of Interest
======================

* Three-dimensional scalar flux distribution :math:`\phi(x,y,z)`.
* Flux attenuation along the streaming channel and through the shield.

Reference Solution
==================

Reference solutions are tabulated in [Kobayashi2001]_ for several
axial slices.  Post-processing in MC/DC generates :math:`(x,y)` flux
maps at selected :math:`z`-planes for direct comparison.

References
==========

.. [Kobayashi2001] K. Kobayashi, N. Sugimura, and Y. Nagaya,
   "3D Radiation Transport Benchmark Problems and Results for Simple
   Geometries with Void Region,"
   *Progress in Nuclear Energy*, **39**:2, 119–144 (2001).
   `[link] <https://www.sciencedirect.com/science/article/abs/pii/S0149197001000075>`__

Step-by-Step Walkthrough
========================

This section walks through the input file block by block.

**1. Import and Materials (lines 1–13)**

.. literalinclude:: ../../../examples/kobayashi/input.py
   :language: python
   :lines: 1-13
   :linenos:
   :lineno-match:

Two mono-energetic multi-group materials are created:
``m`` for the shield (:math:`\Sigma_c = \Sigma_s = 0.05`) and
``m_void`` for the dog-leg channel (:math:`10^{-4}` total).

**2. Surfaces (lines 15–30)**

.. literalinclude:: ../../../examples/kobayashi/input.py
   :language: python
   :lines: 15-30
   :linenos:
   :lineno-match:

Fifteen planar surfaces define the 3-D bounding box and the internal
partitions.  Reflective conditions on ``sx1``, ``sy1``, ``sz1`` exploit
the quarter-symmetry; vacuum on the outer faces allows leakage.

**3. Cells — CSG Region Definitions (lines 32–44)**

.. literalinclude:: ../../../examples/kobayashi/input.py
   :language: python
   :lines: 32-44
   :linenos:
   :lineno-match:

Three cells cover the domain:

- The **source cell** (a small corner cube) filled with shield material.
- The **void channel** — four rectangular segments combined with the
  ``|`` (union) operator to form the L-shaped duct.
- The **shield** — the full box minus the void channel, using the
  ``~`` (complement) operator.

**4. Source (lines 50–57)**

.. literalinclude:: ../../../examples/kobayashi/input.py
   :language: python
   :lines: 50-57
   :linenos:
   :lineno-match:

An isotropic, uniformly distributed source fills the
:math:`10 \times 10 \times 10` cm corner cube.

**5. Tallies, Settings, Techniques, and Run (lines 63–74)**

.. literalinclude:: ../../../examples/kobayashi/input.py
   :language: python
   :lines: 63-74
   :linenos:
   :lineno-match:

- A uniform :math:`60 \times 100 \times 60` mesh tally records scalar flux.
- 1 000 source particles in 2 batches (increase for production).
- Implicit capture prevents particles from being absorbed prematurely.
- ``mcdc.run()`` launches the simulation.

**What to try:**

- Increase ``N_particle`` to :math:`10^5` or more for smoother flux maps.
- Change void-channel cross sections to see how attenuation changes.
- Add a time grid to the tally for a transient variant (see the TD example).

Full Input
==========

Click here to view the input file: `examples/kobayashi/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/kobayashi/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/kobayashi/input.py
  :language: python
  :linenos:

How to Run
==========

From the repository root run::

  python examples/kobayashi/input.py

Expected Output
===============

A mesh tally HDF5 file with 3-D flux data and example plotting using
the companion ``process-output.py`` in the same directory.
