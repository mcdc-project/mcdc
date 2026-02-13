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

Input
=====

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
