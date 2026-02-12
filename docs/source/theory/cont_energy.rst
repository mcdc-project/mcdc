.. _cont_energy:

=================
Continuous Energy
=================

MC/DC supports continuous energy (CE) neutron transport using pointwise nuclear data libraries.
In CE mode, cross sections are represented as energy-dependent tabulated data rather than multi-group averages, enabling higher-fidelity simulations.

Data Libraries
--------------

CE data is loaded from HDF5 files for each nuclide at a specified temperature.
The environment variable ``MCDC_LIB`` must point to the library directory.
Each nuclide file (e.g., ``U235-293.6K.h5``) contains:

- An energy grid (converted from MeV to eV on load),
- Pointwise total, elastic, capture, inelastic, and fission cross sections,
- Angular and energy distributions for secondary particles,
- Prompt and delayed fission neutron multiplicities and spectra,
- Delayed neutron precursor data (fractions, decay constants, and energy spectra).

Supported temperature points are 0.1, 233.15, 273.15, 293.6, 600.0, 900.0, 1200.0, and 2500.0 K.
MC/DC selects the nearest available temperature for each nuclide.

Cross Section Evaluation
------------------------

Microscopic cross sections are evaluated by binary search on the energy grid followed by linear interpolation between bounding points.
Macroscopic cross sections for a material are computed as the sum over constituent nuclides:

.. math::

   \Sigma(E) = \sum_i N_i \, \sigma_i(E)

where :math:`N_i` is the atom density and :math:`\sigma_i(E)` is the microscopic cross section of nuclide :math:`i` at energy :math:`E`.

Collision Physics
-----------------

CE collision processing implements full center-of-mass (COM) kinematics:

- **Elastic scattering** (MT-2): Thermal motion of the target nucleus is sampled from a Maxwellian distribution parameterized by :math:`\beta = \sqrt{A m / (2 k_B T)}`, where :math:`A` is the mass ratio. Rejection sampling is used for the relative speed.
- **Inelastic scattering**: Multiple MT channels with tabulated energy-angle distributions (Kalbach-Mann, evaporation, Maxwellian, N-body, level scattering).
- **Capture**: Particle is absorbed; implicit capture can be enabled as a variance reduction technique.
- **Fission**: Secondary particles are emitted using :math:`\nu(E)/k_\text{eff}` scaling, with prompt and delayed components sampled separately.

Relativistic particle speed is computed as:

.. math::

   v = c \, \frac{\sqrt{E(E + 2m_n c^2)}}{E + m_n c^2}

.. note::

   CE libraries are provided to CEMeNT members via an internal repository.
   Due to export controls, this data cannot be distributed publicly.
   See the :ref:`install` for configuration instructions.
