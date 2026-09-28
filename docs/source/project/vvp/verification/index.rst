.. _project_vvp_verification:

============
Verification
============

Verification assesses whether MC/DC correctly solves its stated mathematical and numerical problems.
:doc:`Unit tests <../../../developer_guide/contributing/unit_testing>` verify individual components, interfaces, and edge cases, while VVP uses analytical and code-to-code comparisons to test complete simulations through full-transport problems.
These separate activities provide verification evidence at complementary scales.

Verification suites and cases
-----------------------------

The current VVP verification program is organized first by verification method, then by particle type, suite, and case.

- **Analytical verification** — Provides the strongest direct test when an analytical, manufactured, or accurately converged semi-analytical reference is available.

  - **Neutron** — Covers fixed-source and :math:`k`-eigenvalue neutron transport.

    - **Fixed-source suite** — Tests error convergence as the source-particle count increases.

      - **Slab problems** — Exercise one-dimensional geometry, materials, sources, energy groups, and time dependence.

        - :doc:`Absorbium slab <analytical/neutron/fixed_source/slab_absorbium>`
        - :doc:`Manufactured two-group slab <analytical/neutron/fixed_source/mms_two_group_slab>`
        - :doc:`Time-dependent isotropic-beam slab <analytical/neutron/fixed_source/slab_isobeam_td>`
        - :doc:`Reed's slab problem <analytical/neutron/fixed_source/reed>`

      - **AZURV1 variants** — Test criticality, censuses, tally recombination, and variance reduction on one transient problem.

        - :doc:`Base critical problem <analytical/neutron/fixed_source/azurv1>`
        - :doc:`Subcritical medium <analytical/neutron/fixed_source/azurv1_sub>`
        - :doc:`Supercritical medium <analytical/neutron/fixed_source/azurv1_super>`
        - :doc:`Time censuses <analytical/neutron/fixed_source/azurv1-census>`
        - :doc:`Census-based tallying <analytical/neutron/fixed_source/azurv1-census-tally>`
        - :doc:`Basic variance reduction <analytical/neutron/fixed_source/azurv1-basic_techniques>`
        - :doc:`Weight windows <analytical/neutron/fixed_source/azurv1-weight_windows>`

      - **Infinite-medium SHEM-361 variants** — Test 361-group steady-state and transient transport, weight windows, and censuses without spatial leakage.

        - :doc:`Steady-state <analytical/neutron/fixed_source/inf_shem361>`
        - :doc:`Steady-state with weight windows <analytical/neutron/fixed_source/inf_shem361-weight_windows>`
        - :doc:`Transient <analytical/neutron/fixed_source/inf_shem361_td>`
        - :doc:`Transient with time censuses <analytical/neutron/fixed_source/inf_shem361_td-census>`

    - **k-eigenvalue suite** — Tests multiplication-factor and fundamental-mode convergence with active cycles.

      - **Finite-slab criticality problems** — Test leakage-dependent modes in homogeneous and heterogeneous one-group slabs.

        - :doc:`Homogeneous one-group slab criticality <analytical/neutron/k_eigenvalue/one_group_slab>`
        - :doc:`Kornreich-Parsons slab <analytical/neutron/k_eigenvalue/kornreich>`

      - **Infinite-medium SHEM-361 variants** — Test 361-group eigenpairs below and above criticality without spatial leakage.

        - :doc:`Subcritical <analytical/neutron/k_eigenvalue/inf_shem361_subcritical>`
        - :doc:`Supercritical <analytical/neutron/k_eigenvalue/inf_shem361_supercritical>`

- **Code-to-code verification** — Extends verification to complex problems without tractable analytical references and tests whether differences between independent codes decrease at the expected statistical rate.
  Convergence increases confidence in the implementations and helps expose modeling or numerical discrepancies.
  Agreement does not establish correctness because the codes may share assumptions, defects, or biases.

  - **Neutron suite** — Covers multidimensional reactor and shielding transients.

    - :doc:`C5G7 four-phase transient <code_to_code/neutron/c5g7-4phase>`
    - :doc:`Kobayashi dog-leg transient <code_to_code/neutron/kobayashi>`

Reading verification results
----------------------------

Monte Carlo estimates fluctuate statistically, so a single close comparison is not sufficient evidence of convergence.
The verification suites repeat each case over a sequence of sampling levels and compare the observed error or difference with the expected inverse-square-root behavior.
For a source-particle count :math:`N`, ordinary Monte Carlo sampling predicts errors proportional to :math:`N^{-1/2}` when statistical uncertainty dominates.
The fixed-source suites therefore vary the number of source particles and report a relative 2-norm together with a maximum relative error or difference.

The :math:`k`-eigenvalue suite instead varies the number of active cycles while holding the particles per cycle and inactive cycles fixed within each case.
Its convergence plots test the corresponding :math:`N_\mathrm{active}^{-1/2}` behavior, while its uncertainty plots show whether the reference multiplication factor is consistent with the reported MC/DC uncertainty.

Solution plots use the task with the largest sampling effort to show where MC/DC agrees with or departs from the reference.
Time-dependent solution comparisons animate spatial or energy-dependent behavior throughout the transient.

For code-to-code cases, the arithmetic mean of the participating-code estimates at the largest common sampling level defines one fixed comparison reference for the entire convergence study.
The pairwise relative differences at every sampling level are normalized by that reference.
Decay proportional to :math:`N^{-1/2}` indicates that the codes are approaching the same solution at the expected statistical rate, whereas a persistent plateau may indicate a modeling discrepancy, implementation bias, or insufficiently resolved reference.

.. toctree::
   :hidden:

   analytical/neutron/fixed_source/slab_absorbium
   analytical/neutron/fixed_source/mms_two_group_slab
   analytical/neutron/fixed_source/slab_isobeam_td
   analytical/neutron/fixed_source/reed
   analytical/neutron/fixed_source/azurv1
   analytical/neutron/fixed_source/azurv1_sub
   analytical/neutron/fixed_source/azurv1_super
   analytical/neutron/fixed_source/azurv1-census
   analytical/neutron/fixed_source/azurv1-census-tally
   analytical/neutron/fixed_source/azurv1-basic_techniques
   analytical/neutron/fixed_source/azurv1-weight_windows
   analytical/neutron/fixed_source/inf_shem361
   analytical/neutron/fixed_source/inf_shem361-weight_windows
   analytical/neutron/fixed_source/inf_shem361_td
   analytical/neutron/fixed_source/inf_shem361_td-census
   analytical/neutron/k_eigenvalue/one_group_slab
   analytical/neutron/k_eigenvalue/kornreich
   analytical/neutron/k_eigenvalue/inf_shem361_subcritical
   analytical/neutron/k_eigenvalue/inf_shem361_supercritical
   code_to_code/neutron/c5g7-4phase
   code_to_code/neutron/kobayashi
