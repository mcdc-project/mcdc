.. _pythonapi:

================
Input Definition
================

Full API documentation.


Defining materials
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Material
   mcdc.MaterialMG


Defining geometry
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Cell
   mcdc.Lattice
   mcdc.Surface
   mcdc.Universe

Defining meshes
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.MeshUniform
   mcdc.MeshStructured

Defining sources
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Source

Defining tallies
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Tally

Defining simulation settings
-----------------------------

Settings are configured by assigning attributes on the ``mcdc.settings`` singleton.
Key attributes include:

- ``mcdc.settings.N_particle`` — Number of particles.
- ``mcdc.settings.N_batch`` — Number of batches.
- ``mcdc.settings.rng_seed`` — RNG seed.
- ``mcdc.settings.output_name`` — Output file name (default: ``"output"``).
- ``mcdc.settings.time_boundary`` — Time boundary.

Methods:

- ``mcdc.settings.set_eigenmode(N_inactive=..., N_active=..., k_init=...)`` — Enable k-eigenvalue mode.
- ``mcdc.settings.set_time_census(time, tally_frequency=...)`` — Set time census parameters.
- ``mcdc.settings.set_source_file(source_file_name)`` — Load source particles from file.

Defining techniques
-------------------

Techniques are enabled by calling methods on the ``mcdc.simulation`` singleton:

- ``mcdc.simulation.implicit_capture(active=True)``
- ``mcdc.simulation.weighted_emission(active=True, weight_target=1.0)``
- ``mcdc.simulation.weight_roulette(weight_threshold=0.0, weight_target=1.0)``
- ``mcdc.simulation.weight_windows(mesh, weight_windows)``
- ``mcdc.simulation.population_control(active=True)``

Running
-------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcfunction.rst

   mcdc.run








