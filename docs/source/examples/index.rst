.. _examples:

================
Example Problems
================

MC/DC ships with a curated set of benchmark and demonstration problems that
exercise the code's key capabilities—from simple mono-energetic fixed-source
calculations to full-core, continuous-energy reactor transients.
Each example is self-contained: an ``input.py`` script sets up the problem and
a companion ``process-output.py`` (or ``process.py``) script post-processes
the results.

The problems below mirror the concrete example folders present in the
`examples/` directory of this repository.  Each page embeds the
corresponding ``input.py`` so that the documented setup exactly matches
the code shipped in-tree.  A **Step-by-Step Walkthrough** section breaks
the input into annotated blocks, and a **What to try** box suggests
parameter changes for further exploration.

.. toctree::
   :maxdepth: 1

   kobayashi_dog_leg
   kobayashi_td
   pincell
   moving_source
   moving_pellet
   fuel_array_packaged
   sphere_in_cube
   c5g7_k_eigenvalue
   c5g7_transient
