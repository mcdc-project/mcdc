.. _project_vvp_fixed_source_inf_shem361:

======================================
Steady-State Infinite-Medium SHEM-361
======================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Infinite-medium SHEM-361 variants → Steady-state

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/inf_shem361>`_

Problem setup
-------------

This steady-state problem transports neutrons in an effectively infinite, homogeneous 361-group medium.
The multigroup constants were produced by homogenizing a continuous-energy calculation of an infinite lattice of borated pressurized-water-reactor pin cells onto the SHEM-361 energy structure.
The dataset includes capture, scattering, fission, prompt and delayed emission, spectra, group speeds, and precursor decay data.

For this fixed-source case, every capture cross section is increased by 50 percent to make the system subcritical.
Reflecting boundaries at very large coordinates approximate the infinite medium, and an isotropic source emits in group 361.
The tally records the steady-state spectrum over all energy groups.

Exercised features
------------------

- Steady-state 361-group neutron transport in an infinite medium.
- Group-to-group scattering and fission production.
- Isotropic mono-group source sampling.
- Reflecting boundaries for an effectively infinite homogeneous medium.
- Energy-resolved flux-spectrum tallying.

Mathematical derivation
-----------------------

.. include:: ../../../_derivations/shem361_fixed.inc

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--inf_shem361_flux.png
         :alt: Spectrum convergence for steady-state infinite-medium SHEM-361.

         Relative spectrum error over the source-particle study.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--inf_shem361_flux.png
         :alt: Spectrum comparison for steady-state infinite-medium SHEM-361.

         Highest-statistics MC/DC spectrum and matrix reference.

The global spectrum error decreases with sampling effort, with modest pointwise fluctuation in the maximum metric across groups having very different populations.
The comparison resolves the detailed slowing-down and thermal spectrum over all 361 groups.

Reference
---------

- A. Hébert and A. Santamarina, *Refinement of the Santamarina-Hfaiedh Energy Mesh Between 22.5 eV and 11.4 keV*, PHYSOR 2008, 2008.
