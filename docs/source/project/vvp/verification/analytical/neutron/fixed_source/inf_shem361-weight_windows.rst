.. _project_vvp_fixed_source_inf_shem361_weight_windows:

=========================================================
Steady-State Infinite-Medium SHEM-361 with Weight Windows
=========================================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Infinite-medium SHEM-361 variants → Steady-state with weight windows

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/inf_shem361-weight_windows>`_

Problem setup
-------------

This case retains the subcritical infinite-medium physics and group-361 isotropic source of :doc:`inf_shem361` while enabling energy-dependent weight windows.
The target weight in each group is shaped by the analytical steady-state spectrum, normalized to its largest value, and limited by a floor of :math:`10^{-3}`.
A width factor of 2.5 defines the lower and upper window bounds around each target.

The campaign uses :math:`10^3` through :math:`10^5` source particles.
It tests energy-indexed window lookup, splitting, roulette, and particle-weight conservation across a spectrum spanning many orders of magnitude.

Exercised features
------------------

- Steady-state 361-group transport in an effectively infinite medium.
- Group-to-group scattering and fission production.
- Energy-dependent weight-window lookup.
- Particle splitting and weight roulette across energy groups.
- Analytical-spectrum-derived importance mapping and weight conservation.

Mathematical derivation
-----------------------

.. include:: ../../../_derivations/shem361_fixed.inc

The resulting spectrum also defines the energy-dependent windows.
With :math:`\epsilon=10^{-3}` and :math:`w=2.5`,

.. math::

   W_g=\epsilon+(1-\epsilon)\frac{\phi_g}{\max_h\phi_h},
   \qquad
   W_g^- =\frac{W_g}{w},
   \qquad
   W_g^+=wW_g.

These sampling controls must preserve the same linear-system solution.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--inf_shem361-weight_windows_flux.png
         :alt: Spectrum convergence for infinite-medium SHEM-361 with energy weight windows.

         Relative spectrum error with energy-dependent windows.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--inf_shem361-weight_windows_flux.png
         :alt: Spectrum comparison for infinite-medium SHEM-361 with energy weight windows.

         Highest-statistics weight-window spectrum and matrix reference.

The lower-statistics campaign is noisier than the analog study, but its errors decrease without a persistent offset and the highest-statistics spectrum follows the matrix solution across the group structure.

Reference
---------

- A. Hébert and A. Santamarina, *Refinement of the Santamarina-Hfaiedh Energy Mesh Between 22.5 eV and 11.4 keV*, PHYSOR 2008, 2008.
