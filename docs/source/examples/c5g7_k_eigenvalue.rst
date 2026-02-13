.. _example_c5g7_k_eigenvalue:

=============================================
C5G7 — k-eigenvalue example
=============================================

Description
===========

Multigroup k-eigenvalue calculation for the C5G7 benchmark using the
packaged MGXS HDF5 library in ``examples/c5g7``.  This example performs a
static criticality calculation and reports :math:`k_{\mathrm{eff}}` and
gyration-radius diagnostics.

Input
=====

Click here to view the input file: `examples/c5g7/k-eigenvalue/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/c5g7/k-eigenvalue/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/c5g7/k-eigenvalue/input.py
   :language: python
   :linenos:

How to Run
==========

From the repository root run::

   python examples/c5g7/k-eigenvalue/input.py

Expected Output
===============

Eigenvalue history printed to stdout and HDF5 tally data for flux and
gyration-radius diagnostics saved in the build artifacts folder.
