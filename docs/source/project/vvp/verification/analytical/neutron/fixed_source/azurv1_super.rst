.. _project_vvp_fixed_source_azurv1_super:

======================
Supercritical AZURV1
======================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → AZURV1 variants → Supercritical medium

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/azurv1_super>`_

Problem setup
-------------

This variant retains the infinite one-group medium, instantaneous isotropic point source, and space-time tally of the base :doc:`azurv1` case.
Capture, scattering, and fission cross sections remain :math:`1/3`, but the prompt fission yield is increased from 2.0 to 2.3.
The combined scattering and fission production ratio is consequently 1.1, producing a growing supercritical response.

Exercised features
------------------

- One-dimensional time-dependent transport in an effectively infinite medium.
- Instantaneous isotropic point-source sampling.
- Capture, isotropic scattering, and prompt-fission branching in a supercritical medium.
- Space-time mesh-flux tallying over a growing particle population.

Mathematical derivation
-----------------------

For the supercritical problem, the production ratio in the following derivation is :math:`c=1.1`.

.. include:: ../../../_derivations/azurv1.inc

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--azurv1_super_flux.png
         :alt: Flux convergence for supercritical AZURV1.

         Relative space-time flux error for the supercritical medium.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--azurv1_super_flux.gif
         :alt: Animated flux comparison for supercritical AZURV1.

         Highest-statistics MC/DC and analytical flux throughout the transient.

The error trend remains statistically consistent as the particle count increases, and the evolving profile agrees with the analytical supercritical response.

Reference
---------

- B. D. Ganapol, R. S. Baker, J. A. Dahl, and R. E. Alcouffe, `Homogeneous Infinite Media Time-Dependent Analytical Benchmarks <https://www.osti.gov/biblio/975281>`_, LA-UR-01-1854, 2001.
