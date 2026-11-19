.. _first_simulation:

======================
First MC/DC Simulation
======================

This tutorial constructs and runs a one-group shielding calculation.
It assumes familiarity with the basic concepts of Monte Carlo radiation transport.

Monte Carlo transport inputs are generally assembled from the same core components: materials, geometry, particle sources, tallies, and simulation settings.
In MC/DC, these components are represented by Python objects and assembled into a :class:`mcdc.Simulation`.

This tutorial applies the general :doc:`../simulation_lifecycle` to one complete problem.

The example uses multigroup data defined directly in the input, so it does not require an external nuclear-data library.
Its complete, executable source is available in ``examples/slab_shielding``.

MC/DC Workflow
--------------

An MC/DC calculation follows five main steps:

#. Construct the materials, geometry, sources, and tallies.
#. Create a simulation and attach the model objects to it.
#. Configure settings and compile the connected object graph.
#. Visualize or run the prepared model.
#. Generate and post-process the output.

.. important::

   Validate the geometry, source distribution, tally definitions, and settings before interpreting simulation results.
   A successfully executed calculation is not necessarily a correctly specified physical model.

Problem Description
-------------------

The model contains two adjacent slab regions:

- A mostly scattering source region over :math:`0 < z < 2` cm.
- A more strongly absorbing shield over :math:`2 < z < 6` cm.

Both outer boundaries are vacuum.
Particles are emitted isotropically throughout the source region, and a mesh tally records the flux across the entire domain.

.. list-table:: One-group material data
   :header-rows: 1
   :widths: 30 20 20 20

   * - Region
     - Range (cm)
     - :math:`\Sigma_c` (cm\ :sup:`-1`)
     - :math:`\Sigma_s` (cm\ :sup:`-1`)
   * - Source region
     - :math:`0 < z < 2`
     - 0.1
     - 0.9
   * - Shield
     - :math:`2 < z < 6`
     - 0.7
     - 0.3

The total cross section is :math:`1.0\ \text{cm}^{-1}` in both regions.
Changing the capture-to-scatter ratio isolates the effect of the shield on the flux distribution.

Building the Input
------------------

Imports and Simulation
~~~~~~~~~~~~~~~~~~~~~~

NumPy provides the numerical arrays used for cross sections and tally grids.
The ``Simulation`` instance collects the model and controls its execution:

.. code-block:: python3

   import numpy as np

   import mcdc


   simulation = mcdc.Simulation("One-group slab shielding")

Materials
~~~~~~~~~

``Material.multigroup()`` creates a material for neutron multigroup transport and attaches the underlying ``NeutronMultigroupData``.
A one-element capture array and a :math:`1 \times 1` scattering matrix define one-group neutron data:

.. code-block:: python3

   source_region_material = mcdc.Material.multigroup(
       capture=np.array([0.1]),
       scatter=np.array([[0.9]]),
   )
   shield_material = mcdc.Material.multigroup(
       capture=np.array([0.7]),
       scatter=np.array([[0.3]]),
   )

:ref:`Native compositions <user_native_transport>` use the same :class:`mcdc.Material` interface and require an MC/DC nuclear-data library.
See :ref:`install-data-library` for configuration instructions.

Geometry
~~~~~~~~

Three z-planes define the two slab regions.
The outer planes use vacuum boundary conditions; the plane at :math:`z=2` cm is an internal interface:

.. code-block:: python3

   left = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
   interface = mcdc.Surface.PlaneZ(z=2.0)
   right = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")

A cell combines a region with the material that fills it.
A positive half-space selects points above a ``PlaneZ``, while a negative half-space selects points below it:

.. code-block:: python3

   source_cell = mcdc.Cell(
       region=+left & -interface,
       fill=source_region_material,
   )
   shield_cell = mcdc.Cell(
       region=+interface & -right,
       fill=shield_material,
   )
   simulation.set_model([source_cell, shield_cell])

The cells are the roots of the geometry.
MC/DC reaches their materials and surfaces when it compiles the simulation, so those objects do not require separate setter calls.

Source
~~~~~~

In standard multigroup transport, ``energy=0`` emits particles at group 0 isotropically and uniformly throughout the first cell:

.. code-block:: python3

   source = mcdc.Source(
       z=[0.0, 2.0],
       isotropic=True,
       energy=0,
   )
   simulation.set_sources([source])

Tallies
~~~~~~~

The structured mesh divides the 6 cm domain into 60 equal spatial bins.
The tally scores the track-length estimate of flux in each bin:

.. code-block:: python3

   mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
   flux_tally = mcdc.Tally(
       name="slab_flux",
       mesh=mesh,
       scores=["flux"],
   )
   simulation.set_tallies([flux_tally])

Naming the tally makes its location in the output file predictable: ``tallies/slab_flux``.

Settings and Execution
~~~~~~~~~~~~~~~~~~~~~~

This example runs 1,000 particle histories in each of 10 statistically independent batches.
Multiple batches allow MC/DC to estimate the standard deviation of each tally bin:

.. code-block:: python3

   simulation.settings.N_particle = 1_000
   simulation.settings.N_batch = 10
   simulation.settings.output_name = "slab_shielding"

   simulation.run()

The calculation writes its results to ``slab_shielding.h5``.

Complete Input
--------------

The complete runnable input is embedded directly from ``examples/slab_shielding/input.py``:

.. literalinclude:: ../../../../examples/slab_shielding/input.py
   :language: python
   :linenos:

Visualizing the Model
---------------------

Before running transport, insert the following call immediately before ``simulation.run()`` to render an x-z slice of the material geometry.
Spatial coordinates are in cm and snapshot times are in seconds:

.. code-block:: python3

   simulation.visualize_model(
       vis_plane="xz",
       x=[-1.0, 1.0],
       y=0.0,
       z=[0.0, 6.0],
       pixels=(100, 300),
       colors=None,
       time=[0.0],
       save_as="slab_shielding_geometry",
   )

The image is saved as ``slab_shielding_geometry.png``.
Visualization compiles the current model when necessary.

Running the Example
-------------------

Enter the problem directory, then run the input in pure Python mode:

.. code-block:: sh

   cd examples/slab_shielding
   python input.py

Pure Python mode avoids compilation overhead and is suitable for checking a small model.
For accelerated or parallel calculations, see :doc:`../execution/cpu`, :doc:`../execution/gpu`, and :doc:`../execution/batch_systems`.

Post-processing
---------------

MC/DC writes tally results and runtime information to HDF5.
To identify the MC/DC version and Git commit that produced a result, see :ref:`Reading the output's version information <output-source-revision>`.
The companion script reads the spatial grid, normalizes the flux and standard deviation by the mesh-bin widths, and plots the result:

.. literalinclude:: ../../../../examples/slab_shielding/process-output.py
   :language: python
   :linenos:

After the transport calculation finishes, run the companion script from the same problem directory:

.. code-block:: sh

   python process-output.py

The script writes ``slab_shielding_flux.png``.
The dashed line marks the material interface at :math:`z=2` cm.
The flux is expected to decrease more rapidly in the shield because capture accounts for a larger fraction of its total cross section.

Next Steps
----------

After running the original problem, useful variations include:

- Increase ``N_particle`` and compare the reported standard deviation.
- Change the shield capture and scattering cross sections.
- Move the material interface and observe the change in attenuation.
- Add an energy group or another spatial region.
- Add a surface-crossing tally at the material interface.

See :doc:`../examples/index` for examples involving lattices, moving geometry, time-dependent transport, and reactor benchmarks.
