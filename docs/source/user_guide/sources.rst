.. _user_sources:

================
Particle Sources
================

A :class:`mcdc.Source` describes the position, direction, energy, time, particle type, and relative probability of particles introduced into a simulation.
Position and spatial bounds use cm, physical energy uses eV, time uses seconds, and direction vectors are dimensionless.
Only sources passed to :meth:`mcdc.Simulation.set_sources` participate in transport.

Energy Distributions
--------------------

A scalar ``energy`` defines a mono-energetic source:

.. code-block:: python

   physical_source = mcdc.Source(energy=1.0e6)

An ``energy`` array with shape ``(2, N)`` defines a continuous probability density with energy values in the first row in eV and density in the second row in eV\ :sup:`-1`.
Use ``discrete_energy`` for a probability mass function over physical emission lines:

.. code-block:: python

   emission_lines = mcdc.Source(
       particle_type="electron",
       discrete_energy=(
           [1.0e5, 2.0e5],
           [0.8, 0.2],
       ),
   )

The values in the second row are dimensionless relative probabilities.
The ``energy`` and ``discrete_energy`` inputs are alternative source-energy specifications and cannot be combined.

.. _user_standard_multigroup_sources:

Standard Neutron Multigroup Sources
-----------------------------------

Standard neutron multigroup transport uses a dimensionless group coordinate in place of physical source energy.
A scalar integer selects one group:

.. code-block:: python

   group_zero_source = mcdc.Source(energy=0)

Use ``discrete_energy`` to sample among groups:

.. code-block:: python

   group_mixture = mcdc.Source(
       discrete_energy=(
           [0, 1],
           [0.25, 0.75],
       ),
   )

Group coordinates must be finite, integer-valued, and satisfy ``0 <= energy < G``.
Continuous ``energy`` distributions are not valid in standard neutron multigroup transport.
Native and hybrid transport use physical source energy in eV, as described in :doc:`materials_and_multigroup`.
