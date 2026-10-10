.. _particle_transport_architecture:

===============================
Particle Transport Architecture
===============================

``mcdc/transport/simulation.py`` coordinates particle advancement, physics, geometry, techniques, and tally scoring.
It operates on the prepared runtime state described in :doc:`runtime_data_layout`.
This page explains the transport algorithm and its ownership boundaries; :doc:`transport_execution` explains how the execution modes run it.

The Particle Step
-----------------

``step_particle`` organizes one step into the following stages, returning early when the particle is lost or terminated:

#. Reset event flags, locate the particle as needed, and inspect its geometry.
#. Determine the step distance and endpoint event flags.
#. Score track-length tallies and move the particle.
#. Apply condensed interactions when enabled, then score their interaction contribution.
#. Terminate at the final time boundary, or process a discrete collision or geometry crossing.
#. Apply transport techniques to the surviving particle.
#. Bank the surviving particle at a census when required.

The CPU ``particle_loop`` also applies techniques when a particle first enters the active transport loop.
The step orchestration owns when each process runs; particle-specific physics implements the physical changes.

Step Limits and Endpoint Events
-------------------------------

``determine_next_events`` compares distances to geometry crossing, discrete collision, census, final time, and the maximum condensed step.
The smallest distance limits the flight.
Coincidence checks determine which endpoint flags accompany it:

- ``EVENT_LOST`` terminates a particle whose geometry cannot be resolved or whose selected distance reaches ``INF``.
- ``EVENT_TIME_BOUNDARY`` overrides all coincident endpoint events.
  Movement and enabled condensed interactions still account for the segment before termination.
- ``EVENT_GEOMETRY_CROSSING`` takes precedence over a coincident ``EVENT_COLLISION``.
- ``EVENT_TIME_CENSUS`` can accompany either geometry crossing or collision, but not the final time boundary.
- A condensed-step limit alone leaves ``EVENT_NONE``.

Condensed interactions are applied over the traveled segment regardless of which distance limits it.
Reaching their step limit does not itself imply particle termination.

Discrete and Condensed Interactions
-----------------------------------

``physics.collision_distance`` samples the distance to a discrete collision, and ``physics.collision`` performs the selected reaction.
Here, collision includes discrete absorption and particle-production reactions as well as scattering.
``physics.max_condensed_step_distance`` limits the extent of condensed treatment, while ``physics.condensed_interactions`` applies the accumulated changes over the actual step.
The exposed physics API is flat; particle-specific implementations remain organized under the physics package.

Both treatments use ``InteractionData`` to report scoring information.
The wrappers in ``transport/simulation.py`` each allocate a separate record, initialize its deposited energy, and save the incoming particle state before calling physics.
Physics changes the particle and accumulates the contribution; the wrapper then calls ``score_interaction_tallies``.
If condensed interactions and a discrete collision both occur during one step, their contributions are scored separately.
The discrete collision's incoming state includes the preceding condensed changes.

``InteractionData.energy_deposition`` contains weighted deposited energy in eV.
``InteractionData.incident_particle`` supplies the tally filters, preserving energy, direction, and particle type even when physics changes or replaces the active particle.
For condensed interactions, the snapshot is taken after movement but before the condensed physics is applied: position and time are at the endpoint, while energy and direction precede the condensed changes.
Assigning the condensed contribution to this endpoint assumes steps are small relative to the spatial, temporal, and energy scales resolved by the tally.

Cross-Species Production
------------------------

``physics.produce_cross_species`` checks transport activation, samples product counts and phase space, accounts for outgoing energy, and banks cross-species products.
Particle-specific reactions supply the available-energy deposition balance and handle same-species products separately.
The shared function uses ``InteractionData.incident_particle`` for the incident state and the active particle's RNG stream for sampling.
Products that are not transported leave their energy in the local deposition balance.
Currently, proton inelastic reactions call this function for prompt, single-spectrum products, retaining the existing nuclear frame transformation and census routing.
Photon reactions do not call it: their only transported products are photons, so photoelectric fluorescence lines and pair-production annihilation photons are banked directly by the photon physics as same-species products, and every charged product leaves its energy in the local deposition balance.

Tally Triggers and Scoring
--------------------------

Tally types identify where scoring is triggered during transport.
They do not by themselves specify an estimation method; individual scores implement the corresponding estimators.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Tally type
     - Transport trigger
     - Scoring inputs
   * - ``TallyTracklength``
     - Before movement along a flight segment
     - Starting particle state and segment length
   * - ``TallySurfaceCrossing``
     - During surface-crossing processing
     - Particle state and crossed surface
   * - ``TallyInteraction``
     - After a discrete collision or condensed treatment
     - Saved incoming state and contributions in ``InteractionData``

Interaction scoring uses the incident cell's tally registrations and the saved particle state for filtering.
It runs before the wrapper returns, including when the treatment terminates the particle.
The track-length ``"collision"`` score remains a discrete collision-rate estimate and is distinct from the interaction tally type.

Geometry After Condensed Deflection
-----------------------------------

Geometry inspection selects a crossing using the direction before condensed interactions.
A condensed deflection at the endpoint can turn the particle back toward its incident region.
``geometry.surface_crossing_valid`` compares incident and outgoing surface-normal components in the incident geometry's coordinate frame.
It retraces the incident hierarchy to account for nested rotations and lattice translations.

If the crossing is no longer valid, the condensed-interaction wrapper clears ``surface_ID`` to suppress the surface action.
It retains the geometry-crossing event so that geometry IDs are invalidated and the particle's region is located again.
This check uses endpoint directions; it does not reconstruct a curved microscopic trajectory.
