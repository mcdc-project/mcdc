.. _project_vvp_fixed_source_azurv1_weight_windows:

==========================
AZURV1 with Weight Windows
==========================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → AZURV1 variants → Weight windows

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/azurv1-weight_windows>`_

Problem setup
-------------

This case applies spatial weight windows to the critical :doc:`azurv1` transient.
The target weight in each spatial cell is inversely shaped by the analytical flux averaged over the entire simulation interval from time zero through 20.
The target distribution is normalized, limited by a floor of :math:`10^{-3}`, and expanded into lower and upper bounds with a width factor of 2.5.

The fixed time-averaged map deliberately uses one set of spatial windows throughout the transient.
The study spans :math:`10^4` through :math:`10^6` source particles and tests splitting, roulette, weight conservation, and mesh lookup against an unchanged physical solution.

Exercised features
------------------

- Critical AZURV1 transient transport and space-time flux tallying.
- Mesh-based spatial weight-window lookup.
- Particle splitting above the upper weight bound.
- Weight roulette below the lower weight bound.
- Analytical-solution-derived importance mapping and weight conservation.

Mathematical derivation
-----------------------

The weight windows do not change the expected transport equation, so the reference uses :math:`c=1` in the critical solution derived below.

.. include:: ../../../_derivations/azurv1.inc

The window construction first forms the full-time average

.. math::

   \phi_j^{\mathrm{avg}}
   =\frac{1}{T}\sum_k\Delta t_k\overline{\phi}_{k,j}.

With floor :math:`\epsilon=10^{-3}` and width :math:`w=2.5`, the target and bounds are

.. math::

   W_j=\epsilon+(1-\epsilon)
       \frac{\phi_j^{\mathrm{avg}}}{\max_l\phi_l^{\mathrm{avg}}},
   \qquad
   W_j^- =\frac{W_j}{w},
   \qquad
   W_j^+=wW_j.

The comparisons still use the original time-dependent bin averages rather than the time-averaged window field.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--azurv1-weight_windows_flux.png
         :alt: Flux convergence for AZURV1 with spatial weight windows.

         Relative space-time flux error with analytical-solution-derived windows.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--azurv1-weight_windows_flux.gif
         :alt: Animated flux comparison for AZURV1 with spatial weight windows.

         Highest-statistics weight-window solution and analytical reference.

The error continues to decrease with particle count and shows no persistent bias from the splitting and roulette operations.

Reference
---------

- B. D. Ganapol, R. S. Baker, J. A. Dahl, and R. E. Alcouffe, `Homogeneous Infinite Media Time-Dependent Analytical Benchmarks <https://www.osti.gov/biblio/975281>`_, LA-UR-01-1854, 2001.
