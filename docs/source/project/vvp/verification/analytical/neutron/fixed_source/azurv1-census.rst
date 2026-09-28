.. _project_vvp_fixed_source_azurv1_census:

==========================
AZURV1 with Time Censuses
==========================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → AZURV1 variants → Time censuses

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/azurv1-census>`_

Problem setup
-------------

This case preserves the critical physics, source, and space-time tally of :doc:`azurv1` while introducing time censuses at every integer time from 1 through 19.
Particles crossing a census time are stored and become the source population for the following interval.
Population control regulates that bank between intervals, and enlarged census and source-bank buffers accommodate transient fluctuations.

The case verifies that repeatedly stopping, storing, redistributing, and restarting histories preserves the physical solution across the complete transient.

Exercised features
------------------

- Critical AZURV1 transient transport and space-time flux tallying.
- Repeated time censuses and census-bank persistence.
- Population control and particle redistribution between census intervals.
- Source- and census-bank buffer management.

Mathematical derivation
-----------------------

The census algorithm does not change the physical equation, so the reference uses :math:`c=1` in the uninterrupted critical solution derived below.

.. include:: ../../../_derivations/azurv1.inc

The resulting bin averages are evaluated across the census boundaries without modification.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--azurv1-census_flux.png
         :alt: Flux convergence for AZURV1 with time censuses.

         Relative space-time flux error with 19 census boundaries.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--azurv1-census_flux.gif
         :alt: Animated flux comparison for AZURV1 with time censuses.

         Highest-statistics censused solution and analytical reference.

The expected inverse-square-root trend is retained, and the animation shows no visible discontinuity or bias at the census times.

Reference
---------

- B. D. Ganapol, R. S. Baker, J. A. Dahl, and R. E. Alcouffe, `Homogeneous Infinite Media Time-Dependent Analytical Benchmarks <https://www.osti.gov/biblio/975281>`_, LA-UR-01-1854, 2001.
