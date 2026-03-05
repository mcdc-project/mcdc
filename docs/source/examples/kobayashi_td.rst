.. _example_kobayashi_td:

=============================================
Time-Dependent Kobayashi Dog-Leg
=============================================

Description
===========

Time-dependent variant of the Kobayashi dog-leg shielding benchmark.
See the steady-state Kobayashi example for geometry and material
specifications; this variant drives the problem with a pulsed source and
records time-resolved tallies.

Step-by-Step Walkthrough
========================

This example reuses the exact geometry from the steady-state Kobayashi
dog-leg benchmark.  The key differences are highlighted below.

**1. Source with a Time Window (line 54–60)**

.. literalinclude:: ../../../examples/kobayashi-TD/input.py
   :language: python
   :lines: 50-60
   :linenos:
   :lineno-match:

The source now has ``time=[0.0, 50.0]``, meaning particles are emitted
over a 50 s window rather than instantaneously.

**2. Time-Resolved Tallies (lines 66–69)**

.. literalinclude:: ../../../examples/kobayashi-TD/input.py
   :language: python
   :lines: 66-69
   :linenos:
   :lineno-match:

A ``time`` grid is added to both the mesh tally and a global density
tally.  This creates a time-resolved :math:`\phi(x, y, t)` dataset.
Global ``Tally`` with ``scores=["density"]`` tracks total neutron
population over time.

**What to try:**

- Shorten the source time to create a short pulse and watch the
  neutron cloud propagate.
- Add a finer time grid to capture early transient behaviour.
- Compare with the steady-state Kobayashi results.

Full Input
==========

Click here to view the input file: `examples/kobayashi-TD/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/kobayashi-TD/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/kobayashi-TD/input.py
   :language: python
   :linenos:

How to Run
==========

From the repository root run::

   python examples/kobayashi-TD/input.py

Expected Output
===============

Time-resolved mesh tally HDF5 file and an animation produced by
``process-output.py`` that visualises neutron density versus time.

References
==========

See: Kobayashi *et al.* (2001), Progress in Nuclear Energy.
