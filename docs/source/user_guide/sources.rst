.. _user_sources:

================
Particle Sources
================

A :class:`mcdc.Source` describes the position, direction, energy, time, particle type, and relative probability of particles introduced into a simulation.
Position and spatial coordinates use cm, physical energy uses eV, time uses seconds, and direction vectors are dimensionless.
Only sources passed to :meth:`mcdc.Simulation.set_sources` participate in transport.

Spatial Distributions
---------------------

The ``x``, ``y``, and ``z`` source coordinates are sampled independently.
An unspecified coordinate is fixed at zero, and a scalar fixes a coordinate at the specified value.
Two values define a uniform interval:

.. code-block:: python

   uniform_line = mcdc.Source(x=[0.0, 10.0])

A coordinate array with shape ``(2, N)`` defines a piecewise-linear probability density.
The first row contains coordinates in cm, and the second row contains nonnegative relative densities:

.. code-block:: python

   nonuniform_line = mcdc.Source(
       x=(
           [0.0, 5.0, 10.0],
           [0.2, 1.0, 0.4],
       ),
   )

The coordinates must be finite and strictly increasing.
MC/DC normalizes the probability density internally.
Independent distributions may be specified for multiple coordinates to define a separable multidimensional source.
Use ``position=[x, y, z]`` instead when all three coordinates are fixed.

Angular Distributions
---------------------

With a reference ``direction``, scalar ``polar_cosine`` and ``azimuthal``
values fix the corresponding angle, two values define a uniform interval, and
an array with shape ``(2, N)`` defines a piecewise-linear probability density.
The two angular variables are sampled independently. The optional
``energy_at_polar_cosine`` input correlates source energy with a tabulated
polar-cosine distribution. A one-dimensional array assigns one deterministic
energy in eV to each polar-cosine grid point; MC/DC linearly interpolates the
energy at the sampled polar cosine. An array with shape
``(N_mu, 2, N_energy)`` assigns a conditional energy PDF to each polar-cosine
grid point. The second dimension contains the energy grid and PDF,
respectively.

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
The ``energy``, ``discrete_energy``, and ``energy_at_polar_cosine`` inputs are
alternative source-energy specifications and cannot be combined.

Time Distributions
------------------

A scalar ``time`` fixes the emission time in seconds, two values define a
uniform interval, and an array with shape ``(2, N)`` defines a piecewise-linear
probability density. For a tabulated distribution, the first row contains
strictly increasing times and the second contains nonnegative relative
densities. MC/DC normalizes the density internally.

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
