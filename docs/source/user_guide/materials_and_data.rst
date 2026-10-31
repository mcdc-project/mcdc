.. _user_materials_and_data:

============================
Materials and Transport Data
============================

A :class:`mcdc.Material` describes a physical medium.
Materials can be specified using multigroup data, or using native continuous-energy data.
Defining a material using elements or nuclides will define cross sections using :ref:`native transport data <user_native_transport>`.
Materials can also carry particle-specific data that augments native interaction data, for example with semi-empirical information, or supports specialized and reduced transport treatments.

.. _user_native_transport:

Continuous-Energy Data
----------------------


In MC/DC, *native* refers to data-library-backed transport physics derived from a material's nuclide or element composition.
The composition identifies the physical constituents whose library records provide the interaction data.
Continuous-energy transport physics data are read from a library that uses MC/DC's internal cross-section format.
Material data is loaded by listing either isotopes (e.g. :code:`U235`) or elements (e.g. :code:`C`), which uses that element's natural isotopic abundance for neutron transport, or that element's data for electron and photon transport.
Material composition is given by the atom densities (``number_density``) in atoms/(barn cm), mass densities (``mass_density``) in g/cc, or mass fractions (``mass_fractions``) combined with total material ``density`` in g/cc.
Mass inputs use library nuclide weights, with natural-element weights averaged over the natural isotope abundances.
The corresponding nuclide files must be available even for transport using element data.

If both an element and one or more of its isotopes are specified, the densities are summed.
Element/nuclide names are not checked when the Material is created, so custom elements or nuclides can be used; a name missing from the data library raises an error when the simulation is compiled.
Material temperature is specified in K:

.. code-block:: python

   fuel = mcdc.Material(
       name="UO2",
       number_density={
           "U235": 5.0e-4,
           "U238": 2.2e-2,
           "O": 4.5e-2, # natural oxygen isotopic abundance
       },
       temperature=293.6,
   )

The same material can be specified using mass fractions:

.. code-block:: python

    fuel = mcdc.Material(
        name="UO2",
        mass_fractions={"U235": 0.0193, "U238": 0.862, "O": 0.1184},
        density=10.09,
    )

Set ``MCDC_LIB`` to the directory containing the prepared projectile files for these targets.
See :doc:`data_preparation/index` for library generation and the :doc:`data format reference <../reference/data_format/index>` for file naming and contents.

Multigroup Transport and Data
-----------------------------

Neutron multigroup transport represents neutron energy with discrete groups and describes interactions using groupwise macroscopic data.
It is widely used in general transport pedagogy and in nuclear engineering applications.

:meth:`mcdc.Material.multigroup` creates a material with :class:`mcdc.NeutronMultigroupData`, which holds the cross sections, group speeds, production spectra, and delayed-precursor data.
Macroscopic cross sections use cm\ :sup:`-1`, group speeds use cm/s, and precursor decay rates use s\ :sup:`-1`:

.. code-block:: python

   moderator = mcdc.Material.multigroup(
       name="Moderator",
       capture=np.array([0.1, 0.2]),
       scatter=np.array([
           [0.7, 0.1],
           [0.2, 0.5],
       ]),
       energy_grid=np.array([1.0e-5, 1.0, 20.0e6]),
   )

Combining Native and Multigroup Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Construct :class:`mcdc.NeutronMultigroupData` directly when attaching it to a material with a :ref:`native composition <user_native_transport>`:

.. code-block:: python

   hybrid_fuel = mcdc.Material(
       name="Hybrid fuel",
       number_density={"U235": 5.0e-4, "U238": 2.2e-2},
       neutron_multigroup=mcdc.NeutronMultigroupData(
           capture=np.array([0.10]),
           fission=np.array([0.20]),
           nu_p=np.array([2.50]),
           energy_grid=np.array([1.0e-5, 20.0e6]),
       ),
   )

The explicit energy grid is required whenever a native composition and neutron multigroup data are combined.
Its bounds identify the energy interval where the multigroup transport model is available.

See :ref:`user_standard_multigroup_sources` for the corresponding source-energy convention.

Multigroup Energy Grids and Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An explicit ``energy_grid`` contains ``G + 1`` physical energy boundaries in eV.
Group ``g`` covers ``energy_grid[g] <= E < energy_grid[g + 1]``.
The grid both maps continuous energy to a group and bounds continuous energy reconstructed from a group.

The grid may be omitted for standard multigroup transport when every material omits ``energy_grid``.

For lower and upper group boundaries :math:`E_g` and :math:`E_{g+1}`, the energy-representation policies are:

- ``"midpoint"``: use the arithmetic midpoint, :math:`E=(E_g+E_{g+1})/2`;
- ``"log_midpoint"``: use the geometric midpoint, :math:`E=\sqrt{E_g E_{g+1}}`;
- ``"uniform"``: sample :math:`E` uniformly between the two boundaries; and
- ``"log_uniform"``: sample :math:`\log E` uniformly between their logarithms, equivalently :math:`E=E_g(E_{g+1}/E_g)^\xi` for :math:`\xi\sim\mathcal{U}(0,1)`.

The policies apply when transport reconstructs physical energy from a group, including after a hybrid multigroup interaction.
The logarithmic policies require strictly positive energy boundaries.
Midpoint policies reconstruct one deterministic value per group, while uniform policies sample a new value when continuous energy is reconstructed.
Standard multigroup transport retains the group coordinate instead and does not apply a physical-energy reconstruction policy during particle transport.

MC/DC determines the neutron multigroup transport organization when the simulation is compiled.
Standard neutron multigroup transport applies when every material uses neutron multigroup data without native composition and all materials either omit ``energy_grid`` or share the same explicit grid.
In standard multigroup transport, particle energy uses a dimensionless group coordinate: energy ``0.0`` represents group 0, energy ``1.0`` represents group 1, and so forth.

Transport is hybrid when native composition data is present or multigroup materials use different grids.
Every multigroup dataset participating in hybrid transport requires an explicit physical energy grid because continuous energy selects the applicable material-local group.
In hybrid transport, particle energy remains physical energy in eV, and ``energy_grid`` maps it to a material-local group.
The material's ``energy_representation`` policy maps an outgoing group back to physical energy.
If a particle lies outside that grid, MC/DC uses the material's native neutron data when present; without a native composition, the material has zero interaction cross section at that energy.
