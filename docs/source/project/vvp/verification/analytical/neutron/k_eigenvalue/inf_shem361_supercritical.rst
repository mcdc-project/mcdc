.. _project_vvp_k_eigenvalue_inf_shem361_supercritical:

=======================================
Supercritical Infinite-Medium SHEM-361
=======================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → k-eigenvalue suite → Infinite-medium SHEM-361 variants → Supercritical

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/k_eigenvalue/cases/inf_shem361_supercritical>`_

Problem setup
-------------

This case solves the fundamental mode of the effectively infinite, homogeneous 361-group pin-cell medium with its original capture data.
Removing the 50 percent capture increase used in :doc:`inf_shem361_subcritical` produces a supercritical system while leaving the scattering and fission data unchanged.
Reflecting boundaries suppress leakage, and the initial source is isotropic in group 361.

Each cycle uses one million particles, with 20 inactive cycles followed by the varied number of active cycles.
The tally records the complete energy-dependent fundamental spectrum.

Exercised features
------------------

- :math:`k`-eigenvalue transport with 361 energy groups in an infinite medium.
- Group-to-group scattering and fission production in a supercritical medium.
- Reflecting boundaries for an effectively infinite homogeneous system.
- Fission-source banking, population control, and inactive-cycle convergence.
- Energy-dependent fundamental-mode tallying and multiplication-factor uncertainty estimation.

Mathematical derivation
-----------------------

The supercritical case uses :math:`\alpha=1` in the following derivation.

.. include:: ../../../_derivations/shem361_eigenvalue.inc

The resulting reference multiplication factor is approximately 1.16018.

Results
-------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--inf_shem361_supercritical_flux.png
         :alt: Spectrum convergence for supercritical infinite-medium SHEM-361.

         Relative fundamental-spectrum error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--inf_shem361_supercritical_k-effective.png
         :alt: Multiplication-factor convergence for supercritical infinite-medium SHEM-361.

         Relative multiplication-factor error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--inf_shem361_supercritical_k-effective_errorbar.png
         :alt: Multiplication-factor estimates and uncertainties for supercritical infinite-medium SHEM-361.

         MC/DC estimates, reported uncertainties, and the matrix reference.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--comparison--inf_shem361_supercritical_flux.png
         :alt: Fundamental spectrum comparison for supercritical infinite-medium SHEM-361.

         Highest-statistics normalized spectrum and generalized-eigenvalue reference.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--comparison--inf_shem361_supercritical_k_history.png
         :alt: Cycle-by-cycle multiplication-factor history for supercritical infinite-medium SHEM-361.

         Inactive and active cycle history, active mean, uncertainty, and reference.

The multiplication-factor uncertainty narrows around the supercritical matrix reference and the normalized energy spectrum converges toward the dominant mode.
Together with the subcritical variant, the result exercises the same multigroup operator on both sides of unity.

Reference
---------

- A. Hébert and A. Santamarina, *Refinement of the Santamarina-Hfaiedh Energy Mesh Between 22.5 eV and 11.4 keV*, PHYSOR 2008, 2008.
