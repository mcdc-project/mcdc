.. _project_vvp_fixed_source_inf_shem361_td:

=====================================
Transient Infinite-Medium SHEM-361
=====================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Infinite-medium SHEM-361 variants → Transient

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/inf_shem361_td>`_

Problem setup
-------------

This transient extends the subcritical infinite-medium :doc:`inf_shem361` problem by retaining group-dependent neutron speeds and delayed-neutron precursors.
An instantaneous isotropic source begins in group 361.
The tally uses 100 logarithmically distributed time intervals from :math:`10^{-8}` to :math:`10^1` seconds, preceded by time zero, and records the complete energy spectrum.

Both flux and total neutron density are evaluated.
The problem spans prompt slowing-down, delayed emission, and many decades of physical time in one simulation.

Exercised features
------------------

- Time-dependent 361-group neutron transport in an infinite medium.
- Group-dependent particle speeds and prompt slowing down.
- Delayed-neutron precursor creation, decay, and emission.
- Logarithmically spaced time-energy flux tallies.
- Total neutron-density tallying over the transient.

Mathematical derivation
-----------------------

.. include:: ../../../_derivations/shem361_transient.inc

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--inf_shem361_td_flux.png
         :alt: Flux convergence for transient infinite-medium SHEM-361 transport.

         Relative time-energy flux error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--inf_shem361_td_n.png
         :alt: Neutron-density convergence for transient infinite-medium SHEM-361 transport.

         Relative neutron-density-history error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--inf_shem361_td_flux.gif
         :alt: Animated spectrum comparison for transient infinite-medium SHEM-361 transport.

         Highest-statistics spectrum and matrix-exponential reference throughout the transient.

The relative 2-norm decreases for both quantities as sampling increases.
The maximum flux error is much noisier because it is sensitive to sparsely populated time-energy bins where the reference approaches zero, so the global norm and density history provide the more stable convergence indicators.

Reference
---------

- A. Hébert and A. Santamarina, *Refinement of the Santamarina-Hfaiedh Energy Mesh Between 22.5 eV and 11.4 keV*, PHYSOR 2008, 2008.
