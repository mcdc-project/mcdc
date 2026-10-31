============
File Layouts
============

All files carry the root attributes and source-library records defined in :doc:`conventions`.
Projectile files contain the target properties and transport data for one incident particle.
Atomic evaluations are stored separately in ``atoms.h5`` and shared by interactions that require them.

Projectile files
----------------

Filenames follow ``{projectile}-{target}.h5``.
Projectile identifiers use established particle names, including ``neutron``, ``electron``, and ``photon``.
For light nuclear projectiles, use ``proton`` (H1), ``deuteron`` (H2), ``triton`` (H3), ``helion`` (He3), and ``alpha`` (He4), each denoting the bare nucleus.
Other ions use nuclide identifiers, for example ``Si32``, unless an established particle name is available.
The target is an element or nuclide identifier appropriate to the evaluation.
For temperature-dependent data, the target identifier includes the temperature in kelvin, such as ``Si32-293.6K``.

.. list-table:: Example filenames
   :header-rows: 1

   * - Filename
     - Contents
   * - ``neutron-Si32-293.6K.h5``
     - Neutron transport on Si32 at 293.6 K.
   * - ``photon-Si.h5``
     - Photon transport on elemental silicon.
   * - ``electron-Si.h5``
     - Electron transport on elemental silicon.
   * - ``proton-Si32.h5``
     - Proton transport on Si32.
   * - ``deuteron-Si32.h5``
     - Deuteron transport on Si32.
   * - ``triton-Si32.h5``
     - Triton transport on Si32.
   * - ``helion-Si32.h5``
     - Helion transport on Si32.
   * - ``alpha-Si32.h5``
     - Alpha-particle transport on Si32.

Target properties
^^^^^^^^^^^^^^^^^

A projectile file includes its own target properties, even when another projectile file contains the same properties.

.. list-table:: Root target fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``atomic-number``
     - integer
     - Target element's number of protons.
   * - ``mass-number``
     - integer, nuclide targets
     - Target nucleus's number of protons and neutrons.
   * - ``atomic-weight-ratio``
     - float, nuclide targets
     - Target mass divided by neutron mass.
   * - ``excitation-level``
     - integer, nuclide targets
     - Excitation-state index; zero denotes the ground state.
   * - ``fissionable``
     - boolean, neutron files
     - Whether the target has fission data.
   * - ``radiation-length``
     - float, g/cm2, optional
     - Radiation length expressed as areal mass.
   * - ``temperature``
     - float, K, temperature-dependent files
     - Target temperature.
   * - ``kT``
     - float, eV, temperature-dependent files
     - Boltzmann constant times target temperature.

The parent element follows from the nuclide identifier.
``temperature`` and ``kT`` describe the same thermodynamic state and must agree to the precision of the data.
A spectrum's ``temperature`` parameter is a separate quantity in eV, defined in :doc:`representations`.

During simulation compilation, loaded properties populate the existing shared Element and Nuclide objects.
If repeated properties differ, a warning identifies the values and source files, and the last loaded value is used.
Loading order is deterministic, and the resolved objects are packed after data loading finishes.
A property that contradicts the target's chemical or nuclide identity is an error.

Projectile transport fields
---------------------------

The following fields are stored directly at the projectile file root.

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
Every reaction ``CrossSection`` references the grid in its projectile file.
Repeated energies store the left and right limits of a discontinuity in that order.
Interpolation applies between distinct energies, not across the zero-width interval.
Reaction names identify individual channels, while category attributes identify their transport behavior.

Atomic library
--------------

``atoms.h5`` collects atomic structure and relaxation data, with one group per element at ``/{element}``.
Each element record contains ``atomic-number`` and ``atomic-relaxation/``.
Projectile generators can prepare the required records through the shared atomic-data preparation function.
Compilation reads only the element records required by the simulation.

Atomic relaxation
^^^^^^^^^^^^^^^^^

Each record under ``/{element}/atomic-relaxation/subshells/{subshell}`` describes one subshell.
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
Nonzero shell references must resolve to subshell designators in the same element record.
Relaxation data is shared by all interactions that create a vacancy.
Reaction products and relaxation transitions must not count the same emitted contribution twice.
