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

Step-by-Step Walkthrough
========================

The C5G7 benchmark uses a pre-packaged 7-group cross-section library
(``MGXS-C5G7-TD.h5``).  The input file defines the full-core geometry
using MC/DC’s lattice and universe system.

Key concepts demonstrated:

- **Multi-group materials** loaded from an external HDF5 library via
  ``mcdc.MaterialMG(library=...)``.
- **Pin-cell universes** built from cylindrical fuel pins in square
  moderator cells.
- **Lattice assemblies** that tile pin-cell universes into fuel
  assemblies of different enrichments.
- **Core lattice** that arranges assemblies and reflector regions.
- **k-eigenvalue mode** with ``set_eigenmode()`` for criticality.

Refer to the embedded code below — comments in the source mark each
section (materials, pins, assemblies, core, source, tallies, settings).

**What to try:**

- Increase ``N_particle`` for better :math:`k_{\text{eff}}` statistics.
- Adjust the number of inactive/active cycles.
- Compare :math:`k_{\text{eff}}` with the published C5G7 reference value.

Full Input
==========

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
