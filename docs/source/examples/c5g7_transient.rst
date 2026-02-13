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

Input
=====

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
