.. _continuous_energy:

=================
Continuous Energy
=================

MC/DC supports continuous energy (CE) neutron transport using pointwise nuclear data libraries.
Cross sections and outgoing-particle distributions are evaluated at the particle's energy rather than averaged over energy groups.
The random walk follows :doc:`Monte Carlo transport basics <monte_carlo>`.
For library generation, see :doc:`Neutron data preparation <../user_guide/data_preparation/neutron>`; for stored data structures, see the :doc:`data format reference <../reference/data_format/index>`.

Cross Section Evaluation
------------------------

Microscopic cross sections are evaluated by binary search on the shared energy grid, using the interpolation rules stored with each reaction's cross section.
Macroscopic cross sections for a material are computed as the sum over constituent nuclides:

.. math::

   \Sigma(E) = \sum_i N_i \, \sigma_i(E)

where :math:`N_i` is the atom density and :math:`\sigma_i(E)` is the microscopic cross section of nuclide :math:`i` at energy :math:`E`.

Elastic Scattering
------------------

Elastic scattering accounts for target thermal motion by sampling a collision-weighted Maxwellian target velocity at the evaluation temperature.
The neutron velocity is transformed into the center-of-mass frame, its direction is sampled from the reaction's angular distribution, and the outgoing velocity is transformed back to the laboratory frame.
At energies above the thermal-motion threshold, the target is treated as stationary.

The :doc:`ACE distribution reference <../reference/ace_distributions>` describes outgoing-energy and angle sampling for the supported reaction distributions.

Neutron Speed
-------------

Relativistic particle speed is computed as:

.. math::

   v = c \, \frac{\sqrt{E(E + 2m_n c^2)}}{E + m_n c^2}

where :math:`E` is the neutron kinetic energy, :math:`m_n` is its rest mass, and :math:`c` is the speed of light.
