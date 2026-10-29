=========
Reactions
=========

Each group under ``/reactions/{reaction}`` in a projectile file represents one reaction channel.
Reaction names describe the channel, such as ``elastic``, ``radiative-capture``, or ``inelastic-level-1``.
The transport category is stored separately in the ``category`` attribute.

Basic fields
------------

All reactions have the following basic fields, subject to the conditions shown.

.. list-table:: Basic reaction fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Type / unit
     - Meaning
   * - ``ENDF-MT``
     - integer attribute, optional
     - Original source reaction number.
   * - ``GNDS-label``
     - string attribute, optional
     - Original GNDS reaction label, preserved exactly when available.
   * - ``category``
     - string attribute
     - Transport category from the tables below.
   * - ``xs``
     - CrossSection, barn
     - Reaction cross section on the projectile's shared energy grid.
   * - ``reference-frame``
     - string
     - ``LAB`` for laboratory frame or ``COM`` for center-of-mass frame.
   * - ``Q-value``
     - float, eV, conditional
     - Nuclear reaction energy release; positive for an exothermic reaction.
   * - ``cross-products/``
     - group
     - Indexed additional product records; empty when none are specified.

``Q-value`` is required for neutron, proton, and ion nuclear reactions.
It is not required for the electron and photon interactions listed below.
``reference-frame`` applies to outgoing kinematic quantities and distributions in the reaction.
Isotropy is represented explicitly by a Distribution with ``type = isotropic``.

Source identifiers are metadata, not HDF5 group keys.
For example, neutron MT 102 can be stored as ``radiative-capture`` with ``category = capture``.
An unavailable source label is omitted.

Neutron reactions
-----------------

These reactions reside in ``/reactions`` within ``neutron-{nuclide}-{temperature}K.h5``.
Multiplicity counts outgoing neutrons.
Additional species are described by :ref:`cross-product` records.

.. list-table:: Neutron transport categories
   :header-rows: 1
   :widths: 25 40 35

   * - Category
     - Category-specific fields
     - Meaning
   * - ``elastic-scattering``
     - ``angular-cosine-distribution``: Distribution
     - Elastic neutron scattering with two-body kinematics.
   * - ``capture``
     - None
     - No outgoing neutron; other products may be specified.
   * - ``inelastic-scattering``
     - Fields in the inelastic table below
     - Non-fission neutron production.
   * - ``fission``
     - Fields in the fission table below
     - Neutron production separated into prompt and delayed components.

Elastic recoil follows the reaction's two-body kinematics.
For elastic scattering, ``Q-value`` is zero.

Inelastic scattering
^^^^^^^^^^^^^^^^^^^^

.. list-table:: Category-specific fields
   :header-rows: 1
   :widths: 35 25 40

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``multiplicity``
     - integer
     - Number of outgoing neutrons per reaction.
   * - ``angular-cosine-distribution``
     - Distribution, conditional
     - Required when the spectra supply energy only; omitted when outgoing angle is supplied jointly with energy.
   * - ``energy-distribution``
     - Distribution
     - Outgoing-energy or joint energy-angle distribution; multiple spectra use a :ref:`mixture <distribution-mixture>`.

Multiplicity is positive for this category.
Each outgoing neutron independently samples the energy Distribution.
Independent angular and energy distributions sample their respective quantities independently at the specified incident energy.

Fission
^^^^^^^

.. list-table:: Category-specific fields
   :header-rows: 1
   :widths: 38 22 40

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``prompt-angular-cosine-distribution``
     - Distribution, conditional
     - Required for an energy-only prompt spectrum; omitted for a joint energy-angle spectrum.
   * - ``prompt-energy-distribution``
     - Distribution
     - Prompt outgoing-energy or joint energy-angle distribution; multiple spectra use a :ref:`mixture <distribution-mixture>`.
   * - ``prompt-multiplicity``
     - Data
     - Mean prompt neutrons per fission as a function of incident energy in eV.
   * - ``delayed-multiplicity``
     - Data
     - Mean delayed neutrons per fission as a function of incident energy in eV.
   * - ``delayed-neutrons/fractions``
     - float [D]
     - Relative delayed-neutron group fractions.
   * - ``delayed-neutrons/decay-rates``
     - float [D], 1/s
     - Positive exponential decay rates of the delayed groups.
   * - ``delayed-neutrons/energy-distribution-{i}``
     - Distribution, eV
     - Energy distribution of delayed group i.

For D > 0, fractions are nonnegative and sum to one.
Group i produces an average of ``delayed-multiplicity`` times ``fractions[i]`` neutrons per fission.
The fraction, decay rate, and named distribution at an index describe the same group.
For D = 0, both arrays are empty, no delayed-group distributions are present, and delayed multiplicity is zero.
The delayed-group distribution specifies energy only; it does not imply an unstated angular law.

Fission production data belongs to the reaction, even if the source shares it among multiple fission channels.
The optional string attribute ``fission-production-source`` can describe that provenance.
Total-fission and chance-fission cross sections must not be included together as independently additive channels for the same physical contribution.

Proton reactions
----------------

These reactions reside in ``/reactions`` within ``proton-{nuclide}.h5``.
All include the basic fields, with ``cross-products`` available for species other than protons.

.. list-table:: Proton transport categories
   :header-rows: 1
   :widths: 25 40 35

   * - Category
     - Category-specific fields
     - Meaning
   * - ``elastic-scattering``
     - ``angular-cosine-distribution``: Distribution
     - Elastic proton scattering with two-body kinematics.
   * - ``capture``
     - None
     - No outgoing proton; other products may be specified.
   * - ``inelastic-scattering``
     - Fields in the proton inelastic table below
     - Non-fission proton production.

Proton stopping power is a projectile-level field described in :doc:`file_layouts`.

Proton inelastic scattering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Category-specific fields
   :header-rows: 1
   :widths: 35 25 40

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``multiplicity``
     - integer or Data
     - Number or mean number of outgoing protons per reaction; use Data when it depends on incident energy.
   * - ``angular-cosine-distribution``
     - Distribution, conditional
     - Required for energy-only spectra; omitted when outgoing angle is supplied jointly with energy.
   * - ``energy-distribution``
     - Distribution
     - Outgoing proton energy or joint energy-angle distribution; multiple spectra use a :ref:`mixture <distribution-mixture>`.

Multiplicity is positive. A constant integer is used when the count is fixed;
an energy-dependent yield uses a nonnegative Data table or polynomial.
For energy-only spectra, the angular distribution is sampled independently at the incident energy.

Electron reactions
------------------

These reactions reside in ``/reactions`` within ``electron-{element}.h5``.
All include the basic fields, with ``cross-products`` available for additional production.

.. list-table:: Electron category-specific fields
   :header-rows: 1
   :widths: 23 42 35

   * - Category
     - Fields
     - Meaning
   * - ``elastic-scattering``
     - ``large-angle/xs``: Data, barn; ``large-angle/angular-cosine-distribution``: Distribution
     - Explicit large-angle elastic component; its cross section uses its own incident-energy grid.
   * - ``excitation``
     - ``energy-loss``: Data, eV
     - Energy transferred to atomic excitation as a function of incident energy.
   * - ``bremsstrahlung``
     - ``energy-loss``: Data, eV
     - Electron energy lost through photon emission as a function of incident energy.
   * - ``ionization``
     - ``subshells/{subshell}/`` records described below
     - Subshell-resolved ionization and secondary electron energy data.

Each ionization subshell record contains ``binding-energy`` (float, eV), ``xs`` (Data, barn), ``secondary-energy-distribution`` (Distribution, eV), and a ``cross-products`` group.
The subshell cross section has its own incident-energy grid when represented as a table.
The secondary energy distribution alone does not specify a secondary angular distribution.
Additional emission can be described at reaction level or conditional on a subshell, without duplicating the same production.

Photon reactions
----------------

These reactions reside in ``/reactions`` within ``photon-{element}.h5``.
All include the basic fields, with ``cross-products`` available for additional production.

.. list-table:: Photon category-specific fields
   :header-rows: 1
   :widths: 25 40 35

   * - Category
     - Fields
     - Meaning
   * - ``coherent-scattering``
     - ``form-factor``: Data
     - Dimensionless atomic form factor versus momentum-transfer coordinate.
   * - ``incoherent-scattering``
     - ``scattering-function``: Data, optional
     - Dimensionless incoherent scattering function versus momentum-transfer coordinate.
   * - ``photoelectric``
     - ``subshells/{subshell}/`` records described below
     - Subshell-resolved photoelectric absorption.
   * - ``pair-production``
     - ``field``: string, either ``electron`` or ``nuclear``
     - Identifies electron-field or nuclear-field pair production.

The momentum-transfer grids carry ``unit = 1/angstrom``.
The coordinate definition must match the source tables; the unit alone does not distinguish different momentum-transfer conventions.

Each photoelectric subshell record contains ``designator`` (integer), ``binding-energy`` (float, eV), ``xs`` (Data, barn), and a ``cross-products`` group.
The designator links the subshell to the element's atomic-relaxation data.
The subshell cross section has its own incident-energy grid when represented as a table.
Photoelectrons and relaxation products may be specified conditionally on the subshell rather than through a shell-averaged reaction product.

.. _cross-product:

CrossProduct
------------

Each ``cross-products/{i}`` group describes one additional emitted product channel.
The list is generic and is not grouped by particle name.

.. list-table:: Product fields
   :header-rows: 1
   :widths: 35 25 40

   * - Field
     - Type / unit
     - Meaning
   * - ``particle``
     - string
     - Particle identifier, such as ``photon``, ``electron``, or ``proton``.
   * - ``reference-frame``
     - string
     - ``LAB`` or ``COM``; frame of this product's distribution.
   * - ``yield``
     - Data
     - Mean emitted count per reaction as a function of incident energy in eV.
   * - ``decay-rate``
     - float, 1/s
     - Positive finite rate for an exponential delay, or positive infinity for prompt emission.
   * - ``angle-energy-distribution``
     - Distribution
     - Joint outgoing energy and direction representation, including independent angle-energy representations.

Yield is nonnegative and may be smaller or larger than one.
Separate product records do not specify correlations between different emitted particles.
An independent angle-energy representation stores ``type = independent-angle-energy`` with ``angular-cosine-distribution`` and ``energy-distribution`` child Distributions.
Every product declares its own reference frame, which may differ from the reaction's frame.
