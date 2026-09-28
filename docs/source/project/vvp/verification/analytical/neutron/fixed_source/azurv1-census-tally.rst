.. _project_vvp_fixed_source_azurv1_census_tally:

====================================
AZURV1 with Census-Based Tallying
====================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → AZURV1 variants → Census-based tallying

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/azurv1-census-tally>`_

Problem setup
-------------

This variant uses the critical :doc:`azurv1` model with censuses at times 5, 10, 15, and 20.
Tallying is performed in census-length chunks, with five unit-width time bins collected during each interval and recombined into the full 20-bin result during processing.
Population control is applied to the census bank between intervals.

The study spans :math:`10^4` through :math:`10^6` source particles.
It specifically tests tally allocation, indexing, normalization, and recombination when one physical transient is divided into multiple census executions.

Exercised features
------------------

- Critical AZURV1 transient transport with time censuses.
- Population control across census intervals.
- Interval-local time-tally allocation and indexing.
- Recombination and normalization of independently accumulated tally segments.

Mathematical derivation
-----------------------

The tally segmentation does not change the physical equation, so the reference uses :math:`c=1` in the uninterrupted critical solution derived below.

.. include:: ../../../_derivations/azurv1.inc

Each recombined MC/DC bin is compared directly with the corresponding double integral, including bins on both sides of every census boundary.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--azurv1-census-tally_flux.png
         :alt: Flux convergence for AZURV1 with census-based tallying.

         Relative error of the recombined space-time tally.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--azurv1-census-tally_flux.gif
         :alt: Animated flux comparison for AZURV1 with census-based tallying.

         Highest-statistics recombined tally and analytical reference.

Both the global and maximum errors decrease with sampling effort, demonstrating that the interval tallies are recombined with the correct shape and normalization.

Reference
---------

- B. D. Ganapol, R. S. Baker, J. A. Dahl, and R. E. Alcouffe, `Homogeneous Infinite Media Time-Dependent Analytical Benchmarks <https://www.osti.gov/biblio/975281>`_, LA-UR-01-1854, 2001.
