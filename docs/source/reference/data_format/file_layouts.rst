============
File Layouts
============

All three file types carry the root attributes defined in :doc:`conventions`.
The target's identity follows its filename, and projectile data is stored under ``/projectiles/{projectile}``.
Projectile names include ``neutron``, ``photon``, ``electron``, and ``proton``; an ion uses its nuclide identifier, such as ``Si32``.
Only the projectile groups applicable to the file's target and data are present.

Element files
-------------

``{element}.h5`` stores properties shared by the element's isotopes and element-based projectile data.

.. list-table:: Root fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``source-libraries/``
     - group
     - Contributing library records; see :doc:`conventions`.
   * - ``atomic-number``
     - integer
     - Number of protons in the nucleus.
   * - ``atomic-relaxation/``
     - group, optional
     - Subshell identities and vacancy transition tables.
   * - ``projectiles/electron/``
     - group, optional
     - Electron data; see :doc:`reactions`.
   * - ``projectiles/photon/``
     - group, optional
     - Photon data; see :doc:`reactions`.

Atomic relaxation
^^^^^^^^^^^^^^^^^

Each record under ``/atomic-relaxation/subshells/{subshell}`` describes one subshell.
The subshell name is a record identifier; its ``designator`` supplies the shell identity used by transition tables.

.. list-table:: Subshell fields
   :header-rows: 1
   :widths: 35 25 40

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``designator``
     - integer
     - Unique subshell designator within the element.
   * - ``binding-energy``
     - float, eV
     - Energy required to remove a bound electron.
   * - ``electron-count``
     - float
     - Electron population of the subshell.
   * - ``transitions/energy``
     - float [T], eV
     - Emitted photon or electron energy for each transition.
   * - ``transitions/probability``
     - float [T]
     - Branching probability for each transition from this vacancy.
   * - ``transitions/origin-designator``
     - integer [T]
     - Subshell supplying the electron that fills the vacancy.
   * - ``transitions/secondary-designator``
     - integer [T]
     - Zero for a radiative transition, otherwise the subshell of the second vacancy.

The transition arrays have equal length and matching indices.
Terminal subshells have length-zero transition arrays.
Probabilities are nonnegative and sum to one for a nonempty transition table.
Nonzero shell references must resolve to subshell designators in the same element file.
Relaxation data is shared by all interactions that create a vacancy.
Reaction products and relaxation transitions must not count the same emitted contribution twice.

Nuclide files
-------------

``{nuclide}.h5`` stores nuclide identity and temperature-independent nuclear data.

.. list-table:: Root fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``source-libraries/``
     - group
     - Contributing library records; see :doc:`conventions`.
   * - ``mass-number``
     - integer
     - Total number of protons and neutrons.
   * - ``atomic-weight-ratio``
     - float
     - Target nuclide mass divided by neutron mass.
   * - ``excitation-level``
     - integer
     - Excitation-state index; zero denotes the ground state.
   * - ``fissionable``
     - boolean
     - Whether the nuclide has fission data.
   * - ``radiation-length``
     - float, g/cm2, optional
     - Radiation length expressed as areal mass.
   * - ``projectiles/proton/``
     - group, optional
     - Proton cross sections, reactions, and stopping power.

An ion projectile can likewise have a group under ``/projectiles``.
The parent element is obtained from the element symbol in the nuclide identifier.

Nuclide-temperature files
-------------------------

``{nuclide}-{temperature}K.h5`` stores temperature-dependent nuclide data.

.. list-table:: Root fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``source-libraries/``
     - group
     - Contributing library records, including source temperature and thermal energy.
   * - ``temperature``
     - float, K
     - Target temperature.
   * - ``kT``
     - float, eV
     - Boltzmann constant times target temperature.
   * - ``projectiles/neutron/``
     - group
     - Neutron data for this target and temperature.

``temperature`` and ``kT`` describe the same thermodynamic state and must agree to the precision of the data.
A spectrum's ``temperature`` parameter is a separate quantity in eV, defined in :doc:`representations`.

Projectile groups
-----------------

The following paths are relative to ``/projectiles/{projectile}`` in the appropriate file.

.. list-table:: Projectile fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``xs-energy-grid``
     - float [N], eV
     - Shared, nondecreasing incident-energy grid for reaction cross sections, with at least two distinct energies.
   * - ``reactions/``
     - group
     - Named reactions, each described in :doc:`reactions`.
   * - ``stopping-power``
     - Data, eV cm2/g, conditional
     - Proton mass stopping power as a function of incident energy.

``stopping-power`` is optional for analog proton transport and required when the data supplies condensed-history energy loss.
When represented as a table, it uses its own grid rather than ``xs-energy-grid``.
Every reaction ``CrossSection`` references the grid in its containing projectile group.
Repeated energies store the left and right limits of a discontinuity in that order.
Interpolation applies between distinct energies, not across the zero-width interval.
Reaction names identify individual channels, while category attributes identify their transport behavior.
