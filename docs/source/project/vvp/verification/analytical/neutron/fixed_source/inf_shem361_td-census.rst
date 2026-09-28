.. _project_vvp_fixed_source_inf_shem361_td_census:

=======================================================
Transient Infinite-Medium SHEM-361 with Time Censuses
=======================================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Infinite-medium SHEM-361 variants → Transient with time censuses

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/inf_shem361_td-census>`_

Problem setup
-------------

This case preserves the multigroup physics, delayed-neutron treatment, logarithmic time tally, and instantaneous group-361 source of :doc:`inf_shem361_td`.
Six logarithmically spaced census times from :math:`10^{-5}` through :math:`10^1` seconds divide the transient, and population control regulates the census bank between intervals.
Expanded source and census buffers accommodate changes in population across the prompt and delayed regimes.

The case verifies census storage and restart behavior over six orders of magnitude in time while retaining the full 361-group energy structure.

Exercised features
------------------

- Time-dependent 361-group infinite-medium transport with delayed-neutron precursors.
- Logarithmically spaced time censuses over prompt and delayed regimes.
- Census-bank storage, restart, and population control.
- Source- and census-bank buffer management.
- Time-energy flux and total neutron-density tallies.

Mathematical derivation
-----------------------

.. include:: ../../../_derivations/shem361_transient.inc

The census times only partition particle execution and do not alter this continuous reference solution.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--inf_shem361_td-census_flux.png
         :alt: Flux convergence for transient infinite-medium SHEM-361 transport with censuses.

         Relative time-energy flux error with population control.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--inf_shem361_td-census_n.png
         :alt: Neutron-density convergence for transient infinite-medium SHEM-361 transport with censuses.

         Relative neutron-density-history error with population control.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--inf_shem361_td-census_flux.gif
         :alt: Animated spectrum comparison for transient infinite-medium SHEM-361 transport with censuses.

         Highest-statistics censused spectrum and matrix-exponential reference.

The global flux and density errors decrease with increasing histories and the animation remains continuous across census boundaries.
As in the uninterrupted transient, the pointwise maximum is sensitive to nearly empty time-energy bins and is less regular than the 2-norm.

Reference
---------

- A. Hébert and A. Santamarina, *Refinement of the Santamarina-Hfaiedh Energy Mesh Between 22.5 eV and 11.4 keV*, PHYSOR 2008, 2008.
