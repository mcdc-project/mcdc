.. _project_vvp_code_to_code_neutron_kobayashi:

================================
Kobayashi Dog-Leg Transient
================================

**VVP map:** :doc:`Verification <../../index>` → Code-to-code → Neutron suite → Kobayashi dog-leg transient

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/code_to_code/neutron/cases/kobayashi>`_

Problem setup
-------------

This one-group problem adapts the Kobayashi three-dimensional shielding benchmark by pulsing a source into a low-density dog-leg channel.
The first octant spans 60 by 100 by 60 cm, with reflection on the three symmetry planes and vacuum on the outer boundaries.
The shield material has capture and scattering cross sections of 0.05 cm\ :sup:`-1`, while both values are :math:`5\times10^{-5}` cm\ :sup:`-1` in the nominal void.

The source occupies the 10 cm cube at the origin and emits isotropically from time zero through 50.
Four connected rectangular channel segments turn first through the :math:`x`–:math:`y` plane and then upward in :math:`z`, producing severe streaming and large attenuation away from the source.

The space-time flux is tallied from zero through 200 in 100 time intervals on a uniform 1 cm mesh containing 60 by 100 by 60 cells.
A second tally records the total neutron-density history over the same time grid.

Exercised features
------------------

- Three-dimensional one-group time-dependent transport.
- Streaming through connected low-density channel segments.
- Pulsed isotropic volume-source sampling.
- Reflective symmetry and vacuum outer boundaries.
- Large structured-mesh space-time flux tallying.
- Global neutron-density tallying.

Comparison data
---------------

MC/DC and OpenMC use equivalent benchmark definitions and 30 batches at five particle levels.
The configured particles per batch range from :math:`10^8` through :math:`10^{10}`, corresponding to :math:`3\times10^9` through :math:`3\times10^{11}` total histories.
The large history counts are required to resolve particles that stream through the channel and reach remote regions.

The archived OpenMC calculations are preserved in the associated Zenodo record.
For each scored quantity, the arithmetic mean of the largest MC/DC and OpenMC arrays supplies the fixed normalization reference used at all sampling levels.

Comparison derivation
---------------------

.. include:: ../../_derivations/code_to_code.inc

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--convergence--kobayashi_convergence_flux.png
         :alt: MC/DC and OpenMC flux convergence for the Kobayashi transient.

         Relative 2-norm and maximum space-time flux differences.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--convergence--kobayashi_convergence_density.png
         :alt: MC/DC and OpenMC neutron-density convergence for the Kobayashi transient.

         Relative 2-norm and maximum neutron-density-history differences.

.. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--reference--kobayashi_reference_flux.gif
   :alt: Animated fixed flux reference for the Kobayashi transient.

   Fixed largest-sample reference formed by averaging the MC/DC and OpenMC flux estimates, shown through its orthogonal projections together with the corresponding reference neutron-density history.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--comparison--kobayashi_comparison.gif
         :alt: Animated MC/DC and OpenMC flux comparison for the Kobayashi transient.

         Orthogonal flux projections and neutron-density histories from the largest shared sample.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--comparison--kobayashi_difference.gif
         :alt: Animated relative differences for the Kobayashi transient.

         Spatial flux differences and neutron-density differences throughout the transient.

The reference animation presents the common space-time solution against which the convergence metrics are normalized.
The flux differences decrease cleanly at approximately the inverse-square-root rate across the five sampling levels.
The density differences decrease through most of the campaign but fluctuate at the largest level, illustrating why the complete trend and spatial animations are more informative than any single scalar comparison.

References
----------

- I. Variansyah, `Time-Dependent Kobayashi Dog-Leg Benchmark for Neutron Transport <https://doi.org/10.5281/zenodo.15069882>`_, Zenodo, 2025.
- K. Kobayashi, N. Sugimura, and Y. Nagaya, `3D Radiation Transport Benchmark Problems and Results for Simple Geometries with Void Region <https://doi.org/10.1016/S0149-1970(01)00007-5>`_, Progress in Nuclear Energy, 2001.
