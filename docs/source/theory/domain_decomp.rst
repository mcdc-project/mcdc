.. _dd:

====================
Domain Decomposition
====================

Domain decomposition (DD) enables MC/DC to distribute large transport problems across multiple MPI ranks by partitioning the spatial domain.
Each rank is responsible for tracking particles within its assigned subdomain, and particles that cross subdomain boundaries are communicated to the appropriate neighbor.

Algorithm
---------

The domain decomposition workflow in MC/DC proceeds as follows:

#. **Domain check-in** (``dd_check_in``): At the start of each transport step, each particle is verified to belong to the local subdomain. Particles that have drifted out of bounds are flagged for transfer.
#. **Particle transport**: Standard Monte Carlo transport is performed on all local particles.
#. **Particle send** (``dd_particle_send``): Particles that have crossed a subdomain boundary during transport are packed and sent to the neighboring rank via MPI communication.
#. **Source resolution** (``source_dd_resolution``): After the source iteration or cycle completes, any remaining load imbalances are resolved by redistributing source particles across ranks.

Each rank maintains its own particle banks (active, census, source, future) and only tracks particles physically located in its subdomain.

Usage
-----

Domain decomposition is activated as a technique flag within the simulation.
The decomposition is spatial, meaning the global geometry mesh is divided along one or more axes, and each MPI rank handles a contiguous block of the mesh.

.. note::

   Domain decomposition support is under active development.
   More sophisticated partitioning strategies (e.g., load-balanced or graph-based decomposition) are planned for future releases.
