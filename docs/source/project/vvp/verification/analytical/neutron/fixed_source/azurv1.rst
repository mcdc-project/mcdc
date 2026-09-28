.. _project_vvp_fixed_source_azurv1:

======
AZURV1
======

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → AZURV1 variants → Base critical problem

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/azurv1>`_

Problem setup
-------------

AZURV1 is a one-group, time-dependent Green's-function problem in an infinite homogeneous medium.
MC/DC represents the infinite domain with reflecting planes at very large positive and negative :math:`x` coordinates.
Capture, scattering, and fission cross sections are each :math:`1/3`, the prompt fission yield is 2, and the total cross section is therefore one.
Scattering and fission together give an effective particle-production ratio of one, making the base problem critical.

An instantaneous isotropic point source is placed at :math:`x=0` and :math:`t=0`.
The space-time flux is tallied over 201 cells from -20.5 to 20.5 and 20 unit-width time intervals from zero to 20.

Exercised features
------------------

- One-dimensional time-dependent transport in an effectively infinite medium.
- Instantaneous isotropic point-source sampling.
- Capture, isotropic scattering, and prompt-fission branching at critical balance.
- Space-time mesh-flux tallying.

Mathematical derivation
-----------------------

For the base problem, the production ratio in the following derivation is :math:`c=1`.

.. include:: ../../../_derivations/azurv1.inc

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--azurv1_flux.png
         :alt: Flux convergence for the base AZURV1 problem.

         Relative space-time flux error for the critical problem.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--azurv1_flux.gif
         :alt: Animated flux comparison for the base AZURV1 problem.

         Highest-statistics MC/DC and analytical flux throughout the transient.

The convergence metrics decrease at the expected Monte Carlo rate, and the animation shows agreement as the pulse spreads and attenuates through the medium.

Reference
---------

- B. D. Ganapol, R. S. Baker, J. A. Dahl, and R. E. Alcouffe, `Homogeneous Infinite Media Time-Dependent Analytical Benchmarks <https://www.osti.gov/biblio/975281>`_, LA-UR-01-1854, 2001.
