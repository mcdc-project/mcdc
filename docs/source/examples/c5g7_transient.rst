.. _example_c5g7_transient:

=============================================
C5G7 — Transient example
=============================================

Description
===========

Time-dependent C5G7-TD transient driven by control-rod movements and a
time-limited source.  Uses the packaged MGXS library in
``examples/c5g7`` and demonstrates moving surfaces and time-resolved
tallies.

Step-by-Step Walkthrough
========================

This example extends the C5G7 k-eigenvalue setup with time-dependent
features:

- **Moving surfaces** simulate control-rod insertion/withdrawal.
- **Time-resolved tallies** capture the transient fission rate.
- **Time census** checkpoints the particle population at specified
  intervals for population control.

The geometry and material setup is identical to the k-eigenvalue case.
The transient-specific additions are:

#. Surface velocities assigned via ``surface.move(...)``.
#. A ``time`` grid added to the mesh tally.
#. ``set_time_census(...)`` for time-step population control.

Refer to the embedded code below for the full implementation.

**What to try:**

- Change the rod insertion speed to see prompt vs. delayed transient response.
- Add more time census points for finer population control.
- Compare power history with published C5G7-TD benchmarks.

Full Input
==========

Click here to view the input file: `examples/c5g7/transient/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/c5g7/transient/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/c5g7/transient/input.py
   :language: python
   :linenos:

How to Run
==========

From the repository root run::

   python examples/c5g7/transient/input.py

Expected Output
===============

HDF5 tallies with time-resolved fission rates and PNG visualisations for
fission and relative standard deviation per time step produced by the
companion plotting scripts.
