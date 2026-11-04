.. _simulation_compilation:

======================
Simulation Compilation
======================

MC/DC begins with normal Python objects.
Materials, surfaces, cells, sources, tallies, settings, and techniques can contain nested references and arbitrary-sized arrays.
A :class:`mcdc.Simulation` owns the roots of one model and turns that connected object graph into a deterministic, simulation-local snapshot.

This first compilation stage is entirely a Python operation.
It discovers the model, finalizes object-local and model-wide state, and establishes the snapshot that runtime preparation will pack for Python, Numba-CPU, or Numba-GPU execution.

Why a Simulation Context Is Needed
----------------------------------

A Monte Carlo transport model is a connected system rather than a collection of independent definitions.
A cell is not complete without its bounding surfaces and fill; a material may depend on nuclides, elements, reactions, and their data; and sources and tallies may refer to distributions, meshes, or geometry objects.
The same object may also be shared by several parts of the model.

These relationships are assembled incrementally with ordinary Python references.
While the model is being built, an individual object cannot know its final position among all objects of the same kind, whether another branch of the model will refer to it, or which subtype-specific collection will contain it.
Finalizing each object when it is created would therefore make model construction depend on definition order and require process-wide registration.

:class:`mcdc.Simulation` provides the owning context for this delayed finalization.
After the model cells, sources, tallies, settings, and techniques have been configured, compilation can inspect the complete reachable object graph at once.
It registers shared objects only once, resolves relationships into simulation-local identifiers, and produces a consistent snapshot for runtime preparation.

Here, *context* means the explicit model scope owned by a ``Simulation`` instance, not a Python ``with`` context manager.

Terminology
-----------

MC/DC uses several related forms of compilation:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Term
     - Responsibility
     - Primary implementation
   * - Model compilation
     - Discover objects, register them once, assign local IDs, and finalize model-dependent state.
     - ``mcdc/object_/``, ``Simulation.compile``, and ``mcdc/code_factory/python_objects_compiler.py``
   * - Runtime preparation
     - Pack the finalized model, allocate execution resources, and configure the selected backend.
     - ``mcdc.main.prepare`` and ``mcdc/code_factory/numba_layers_generator.py``
   * - Backend compilation
     - Compile shared transport functions for a CPU or GPU target.
     - Numba and, for GPUs, Harmonize

Unless otherwise qualified, this page uses *compilation* to mean model compilation.

Configuration Boundary
----------------------

``mcdc.config`` connects command-line execution controls to the simulation lifecycle.
When imported, it parses the command-line arguments known to MC/DC, establishes process-wide choices such as Python or Numba mode, CPU or GPU target, caching behavior, and GPU execution options, then configures Numba and generated-code caches accordingly.
Unknown arguments are retained rather than rejected so MC/DC can run under notebooks, test runners, and higher-level Python drivers.

Simulation settings remain owned by :class:`mcdc.Simulation` and its embedded ``Settings`` object.
At the beginning of ``Simulation.compile``, ``config.override_settings`` applies the supported command-line overrides before object discovery and model-wide finalization.
Consequently, derived state such as particle-bank capacities is calculated from the effective settings that will be used for the run.

This separation distinguishes process-wide execution configuration from simulation-owned model configuration.
A new physical or numerical setting should normally be added to ``Settings`` or another simulation-owned object.
Add a ``config.py`` option only when that setting also needs a command-line override or when the choice controls the execution framework itself.

Ownership and Roots
-------------------

Every calculation begins with an explicit simulation:

.. code-block:: python

   simulation = mcdc.Simulation("Shielding")
   simulation.set_model([source_cell, shield_cell])
   simulation.set_sources([source])
   simulation.set_tallies([flux_tally])

The setter calls identify the roots of the user model:

- ``set_model`` places cells in the simulation's root universe.
- ``set_sources`` retains the sources sampled by transport.
- ``set_tallies`` retains the requested scoring definitions.
- Settings and transport techniques are embedded objects already owned by the simulation.

There is no process-wide model singleton.
Each ``Simulation`` owns the model objects reachable from its roots or embedded configuration.

Ownership Boundary and Process Model
------------------------------------

MC/DC intentionally uses a serial-in-process execution model: one ``Simulation`` is compiled, prepared, or executed at a time within a Python process.
A model-object instance belongs to one simulation context.
Repeated in-process calculations should update and recompile the same ``Simulation``.
See :ref:`example_iterative_source_reweighting` for a complete iterative simulation and result-comparison example.

References may be shared freely within that context.
For example, several cells may use the same material, and compilation will register that material once.
The same material instance must not be attached to a second ``Simulation``.
Compilation metadata such as ``compile_ID``, ``ID``, and ``sub_ID`` is stored on the Python object and describes only its owning simulation's current snapshot.

This ownership rule keeps ordinary model construction and recursive compilation lightweight.
Applications that need concurrent calculations should create an independent model graph in each process and manage those processes from an outer Python driver, workflow system, or batch scheduler.
Process-level orchestration is the supported parallelism boundary for multiple independent simulations.

The MC/DC Object Hierarchy
--------------------------

The classes in ``mcdc/object_/base.py`` give every part of the model a common compilation lifecycle:

.. code-block:: text

   MCDCBase
   └── MCDCObject
       └── MCDCPolymorphic

The distinction between these classes is about identity in a compiled simulation.
All three can contribute fields to the packed runtime layout, but only ``MCDCObject`` and ``MCDCPolymorphic`` instances are registered as independently addressable model entities.

``MCDCBase``: Embedded State and Traversal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MCDCBase`` is the foundation for Python-side model and runtime definitions.
Each subclass declares a ``label`` and type-annotated fields.
The annotations serve two related purposes:

- Assignment validation catches incompatible model values during construction.
- The Numba-layer generator uses the annotations to derive structured fields, object references, and variable-length data access.

An ``MCDCBase`` instance participates in recursive compilation and carries a ``compile_ID``, but it does not receive an object ``ID`` or occupy a simulation-level registry.
This behavior is appropriate for state that belongs to a parent rather than representing a separately addressable model entity.
Examples include ``Simulation``, ``Settings``, transport-technique configuration, particle-bank metadata, and GPU metadata.
``Simulation`` is the special root of this hierarchy: it starts compilation and owns the resulting registries instead of being registered in one.

The default compilation method visits object-valued members and lists of members recursively.
Fields named in ``non_numba`` are excluded from this default traversal and packed representation; the owning class handles them explicitly when special processing is required.

``MCDCObject``: Registered Model Entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MCDCObject`` extends ``MCDCBase`` for entities that must be collected and referenced by transport.
Examples include cells, surfaces, universes, lattices, sources, nuclides, and elements.

When compilation reaches an ``MCDCObject``, ``register_object`` places it in the corresponding collection owned by the current ``Simulation`` and assigns its simulation-local ``ID``.
Registration happens before the object's members are traversed.
Consequently, another reference to the same Python instance can recognize that it is already registered, which both preserves sharing and terminates cycles.

An ``ID`` identifies object position within one compiled simulation; it is not a permanent identity belonging to the Python object.
Recompiling its owning simulation may assign a different ``ID``.
Attaching the same object instance to another simulation is outside the ownership model.

``MCDCPolymorphic``: Base and Concrete Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MCDCPolymorphic`` extends ``MCDCObject`` for categories with multiple runtime representations.
Meshes, distributions, tallies, and neutron and electron reactions use this pattern.
For example, ``Tally`` defines the shared tally category, while ``TallySurfaceCrossing``, ``TallyInteraction``, and ``TallyTracklength`` provide tally-type-specific representations distinguished by their scoring triggers during transport.

Each concrete polymorphic class declares a ``sub_type`` code.
During registration, an instance receives two positions:

- ``ID`` locates its base record in the heterogeneous category collection.
- ``sub_ID`` locates its concrete record among objects with the same ``sub_type``.

By convention, the shared category class uses ``sub_type = -1`` and concrete subclasses use named integer constants.
Their distinct ``label`` values also name the corresponding structured layouts and generated accessor modules.

The packed base record stores ``sub_type`` and ``sub_ID``; the concrete record stores ``base_ID``.
Transport can therefore move between the common category view and the subtype-specific data without retaining Python references or depending on Python dynamic dispatch.

For the implementation steps required to add a field, registered category, or polymorphic subtype, see :doc:`../extending/extending_the_object_model`.

Recursive Discovery
-------------------

``Simulation.compile`` assigns a new ``compile_ID``, clears the previous registries, and walks the object graph:

#. Reserved ``None`` representations are registered.
#. The root universe recursively reaches cells, regions, surfaces, fills, universes, lattices, materials, nuclear data, and distributions.
#. Sources and tallies are traversed from their explicit root lists.
#. Settings, techniques, and their members are traversed from the simulation.

The common lifecycle defined by the object hierarchy makes this traversal uniform: embedded ``MCDCBase`` members are visited in place, while ``MCDCObject`` members are registered before their descendants are explored.

Discovery follows actual Python references.
For example, a root cell reaches its region, the region reaches its surfaces, and the cell's fill reaches its material or child universe.
Users therefore attach root cells rather than manually registering every referenced object.

Object and Model Finalization
-----------------------------

Discovery and finalization occur within the same model-compilation snapshot.
An object's ``_compile_into_simulation`` hook handles work owned by that object, including canonicalizing Python inputs, compiling excluded references, and deriving fields from assigned IDs.

Some values require the complete discovered model rather than one object.
After the explicit model roots have been traversed, ``Simulation._finalize_compilation`` resolves these model-wide relationships and invariants, such as completing material data, normalizing source probabilities, adapting tally shapes, deriving particle-bank capacities, and initializing state from the final settings and MPI decomposition.
The compiler traverses embedded simulation configuration before model-wide finalization, and finalization logic explicitly compiles any runtime-visible dependencies it creates.

This boundary keeps scientific model rules in ``mcdc/object_/``.
The orchestration in ``python_objects_compiler.py`` changes only when the compilation framework gains a new phase or registered category, while ``mcdc.main.prepare`` remains responsible for framework-level runtime setup after the model is complete.

Neutron Multigroup Finalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For neutron multigroup transport, model-wide finalization selects standard transport only when every material has multigroup data without native composition and all multigroup energy grids are identical.
An omitted grid is represented internally by a zero-valued ``G + 1`` placeholder, allowing materials that all omit the grid to share one standard multigroup structure.
All other material combinations select hybrid transport, for which each multigroup dataset requires an explicit physical energy grid.
Standard transport interprets the particle's ``E`` field as a dimensionless group coordinate, while hybrid transport interprets it as physical energy in eV and maps it through ``energy_grid`` and ``energy_representation``.
Finalization also validates standard-multigroup source energies and resolves ``energy="all"`` tally filters.

Deduplication and Cycles
------------------------

An object's ``compile_ID`` records the compilation snapshot in which it most recently participated.
When the traversal reaches the same object again during that snapshot, registration stops for that branch.

This behavior has two purposes:

- A material, surface, mesh, or other object shared in several places is registered once.
- Cycles among embedded configuration objects do not cause infinite recursion.

Deduplication is based on Python object identity within one compilation, not on equal field values.
Two separately constructed materials with identical cross sections remain two objects unless a class performs its own canonicalization.
For example, nuclide compilation canonicalizes element references by chemical symbol, so all isotopes and temperatures of an element share one element object within the simulation.

Simulation-local IDs
--------------------

Registered objects receive identifiers that are meaningful only within the current compiled simulation:

``ID``
   Position in the corresponding simulation collection, such as all surfaces or all materials.

``sub_type``
   Integer code identifying the concrete representation of a polymorphic object, such as a surface-crossing, collision, or track-length tally.

``sub_ID``
   Position in the collection for that concrete subtype.

The packed child record also carries ``base_ID`` so transport can move between the subtype-specific and base representations.

For example, transport can inspect a tally base record's ``sub_type`` and ``sub_ID`` and then select the record for its tally type.
These IDs replace Python object references in the runtime layer.

Snapshot Lifecycle
------------------

``set_model``, ``set_sources``, and ``set_tallies`` invalidate the compiled state.
``run`` and ``visualize_model`` call ``compile`` automatically when the simulation is not compiled, so most user inputs do not need an explicit call.
Command-line overrides are applied before this compilation so model-wide derived state reflects the effective settings.

An explicit call is useful for inspecting discovered objects and assigned IDs:

.. code-block:: python

   simulation.compile()

   print(simulation.materials)
   print(source_cell.ID)
   print(source_region_material.ID)

IDs can change whenever the simulation is compiled again.
They must not be stored as durable identifiers outside the current snapshot.

Directly mutating an already attached object does not notify its owning simulation.
After such a change to an explicitly compiled or visualized model, call ``simulation.compile()`` again before inspecting or visualizing the new snapshot.
``simulation.run()`` marks the snapshot uncompiled after execution, so a subsequent run recompiles it.

Iterative Partial Updates
^^^^^^^^^^^^^^^^^^^^^^^^^

An iterative study can keep one simulation and change only part of its owned model between runs.
For example, two sources can be reweighted while the geometry, materials, and tallies remain unchanged:

.. code-block:: python

   simulation.set_sources([source_left, source_right])

   for iteration, left_fraction in enumerate([0.2, 0.5, 0.8]):
       source_left.probability = left_fraction
       source_right.probability = 1.0 - left_fraction
       simulation.settings.output_name = f"source_mix_{iteration}"

       simulation.compile()
       simulation.run()

Each call to ``compile`` assigns a new ``compile_ID`` and rebuilds a complete, consistent snapshot even though only the source probabilities changed.
The objects remain owned by the same ``Simulation`` throughout the study.

From Objects to Runtime Data
----------------------------

Model compilation finalizes the original Python objects and populates the simulation's ordered registries.
Runtime preparation then reads that complete snapshot, derives structured dtypes from its annotations, and packs its values while allocating framework-owned execution resources.

Continue with :doc:`runtime_data_layout` for that conversion.
For the user-facing construct-to-output workflow, see :doc:`../../user_guide/simulation_lifecycle`.
