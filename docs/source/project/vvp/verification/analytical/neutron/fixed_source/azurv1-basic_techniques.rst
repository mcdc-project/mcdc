.. _project_vvp_fixed_source_azurv1_basic_techniques:

========================================
AZURV1 with Basic Variance Reduction
========================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → AZURV1 variants → Basic variance reduction

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/azurv1-basic_techniques>`_

Problem setup
-------------

This case retains the critical :doc:`azurv1` physical problem while enabling three general variance-reduction techniques.
Implicit capture reduces particle weight instead of terminating histories at capture events, weighted emission targets an emitted weight of 0.5, and global weight roulette uses a threshold of 0.75 with surviving particles restored to weight 1.0.

Because the analog and variance-reduced cases share the same source, transport data, and space-time tally, the analytical comparison tests whether the weight transformations remain unbiased throughout the transient.

Exercised features
------------------

- Critical AZURV1 transient transport and space-time flux tallying.
- Implicit capture through continuous particle-weight reduction.
- Weighted emission at scattering and fission events.
- Global weight roulette for low-weight particles.
- Combined variance-reduction weight accounting.

Mathematical derivation
-----------------------

The variance-reduction techniques do not change the expected transport equation, so the reference uses :math:`c=1` in the critical solution derived below.

.. include:: ../../../_derivations/azurv1.inc

Implicit capture, weighted emission, and roulette must reproduce these same bin-averaged expectations despite changing individual history weights and paths.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--azurv1-basic_techniques_flux.png
         :alt: Flux convergence for AZURV1 with basic variance-reduction techniques.

         Relative space-time flux error with all three techniques enabled.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--azurv1-basic_techniques_flux.gif
         :alt: Animated flux comparison for AZURV1 with basic variance reduction.

         Highest-statistics variance-reduced solution and analytical reference.

The errors decrease without a persistent offset, supporting unbiased implementation of implicit capture, weighted emission, and global weight roulette in combination.

Reference
---------

- B. D. Ganapol, R. S. Baker, J. A. Dahl, and R. E. Alcouffe, `Homogeneous Infinite Media Time-Dependent Analytical Benchmarks <https://www.osti.gov/biblio/975281>`_, LA-UR-01-1854, 2001.
