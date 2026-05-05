.. _theory:

============
Theory Guide
============

We provided a brief theory guide into the methods, algorithms, and compilation
schemes in MC/DC.

New to Monte Carlo transport?  Start with :ref:`mc_basics` for the
fundamentals and :ref:`geometry` for how MC/DC represents problem domains.
Then explore the advanced topics below.

For an additional external resource, see the
`OpenMC theory guide <https://docs.openmc.org/en/latest/methods/index.html>`_.

Fundamentals
------------

.. toctree::
   :maxdepth: 1

   mc_basics
   geometry
   k_eigenvalue

Advanced Methods
----------------

.. toctree::
   :maxdepth: 1

   variance_reduction
   ana
   iqmc
   ww
   uq
   compressed_sensing

Implementation
--------------

.. toctree::
   :maxdepth: 1

   gpu
   cont_energy
   domain_decomp
   cont_movement
