.. _project_vvp_code_to_code_neutron_c5g7_4phase:

==========================
C5G7 Four-Phase Transient
==========================

**VVP map:** :doc:`Verification <../../index>` → Code-to-code → Neutron suite → C5G7 four-phase transient

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/code_to_code/neutron/cases/c5g7-4phase>`_

Problem setup
-------------

This seven-group problem adapts the heterogeneous C5G7 time-dependent benchmark into four source- and control-rod-driven phases over a 20-second calculation.
The modeled quadrant contains four 17-by-17 fuel assemblies—two uranium-dioxide and two mixed-oxide assemblies—surrounded on its outer sides and bottom by moderator reflectors.
Symmetry planes are reflective, while the exterior radial and axial boundaries are vacuum.

The pin pitch is 1.26 cm, the fuel radius is 0.54 cm, the active core height is 128.52 cm, and the reflector thickness is 21.42 cm.
The material set contains uranium-dioxide fuel, three mixed-oxide enrichments, guide tubes, fission chambers, control rods, and moderator.
Four control-rod banks follow independent piecewise-linear insertion histories, exercising continuously moving surfaces throughout the transient.

An isotropic highest-energy-group source occupies the central active pin of the fourth assembly from time zero through 15 seconds.
The quantity of interest is the seven-group fission rate tallied from zero through 20 seconds in 200 time intervals on a 34-by-34-by-102 pin-pitch-resolved spatial mesh.

Exercised features
------------------

- Three-dimensional seven-group time-dependent transport.
- Nested pin-cell, lattice, assembly, and reflector geometry.
- Continuously moving control-rod surfaces with independent motion schedules.
- Prompt and delayed fission-neutron physics.
- Time-dependent localized source sampling.
- Pin-pitch-resolved space-time fission-rate tallying.
- Reflective symmetry and vacuum leakage boundaries.

Comparison data
---------------

MC/DC and OpenMC use equivalent benchmark definitions and 30 batches at five particle levels.
The configured particles per batch range from :math:`10^5` through :math:`10^7`, corresponding to :math:`3\times10^6` through :math:`3\times10^8` total histories.
The archived OpenMC results and benchmark definition are preserved in the associated Zenodo record.

The fixed normalization field is the arithmetic mean of the largest MC/DC and OpenMC fission-rate arrays.
All lower-sampling comparisons use that same field, preventing the reference itself from changing with particle count.

Comparison derivation
---------------------

.. include:: ../../_derivations/code_to_code.inc

Results
-------

.. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--convergence--c5g7-4phase_convergence_fission.png
   :alt: MC/DC and OpenMC fission-rate convergence for the C5G7 four-phase transient.

   Relative 2-norm and maximum fission-rate differences as the common sampling effort increases.

.. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--reference--c5g7-4phase_reference_fission.gif
   :alt: Animated fixed fission-rate reference for the C5G7 four-phase transient.

   Fixed largest-sample reference formed by averaging the MC/DC and OpenMC fission-rate estimates, shown through its orthogonal projections together with the integrated fission-rate history.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--comparison--c5g7-4phase_comparison.gif
         :alt: Animated MC/DC and OpenMC fission-rate comparison for the C5G7 four-phase transient.

         Orthogonal fission-rate projections from the largest shared sample.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--code_to_code--neutron--comparison--c5g7-4phase_difference.gif
         :alt: Animated relative fission-rate differences for the C5G7 four-phase transient.

         Spatial relative differences and their root-mean-square history.

The reference animation presents the common space-time solution against which the convergence metrics are normalized.
Both difference metrics decline with increasing histories at approximately the expected statistical rate.
The animations show how the fission-rate field responds to the four independently moving rod banks and expose where the remaining MC/DC–OpenMC differences are concentrated at each time.

References
----------

- I. Variansyah, `Four-Phase C5G7 Transient Benchmark for Neutron Transport <https://doi.org/10.5281/zenodo.15719118>`_, Zenodo, 2025.
- J. Hou, K. N. Ivanov, V. F. Boyarinov, and P. A. Fomichenko, `OECD/NEA Benchmark for Time-Dependent Neutron Transport Calculations Without Spatial Homogenization <https://doi.org/10.1016/j.nucengdes.2017.02.008>`_, Nuclear Engineering and Design, 2017.
