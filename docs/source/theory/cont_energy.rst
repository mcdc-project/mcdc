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


Generating a Data Library from ACE Files
-----------------------------------------

MC/DC ships with a conversion tool in ``tools/data_library_generator/`` that reads
standard ACE-format nuclear data files and writes them into MC/DC's per-nuclide
HDF5 format.  This is the primary path for creating CE libraries.

**Prerequisites:**

.. code-block:: sh

   pip install ACEtk h5py numpy tqdm

You also need a set of ACE files (e.g., from `NJOY <http://www.njoy21.io/>`_ or
an ENDF/B distribution).

**Environment variables:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Description
   * - ``MCDC_ACELIB``
     - Path to the directory containing your ACE files.
   * - ``MCDC_LIB``
     - Path to the output directory where MC/DC HDF5 files will be written.

**Running the generator:**

.. code-block:: sh

   export MCDC_ACELIB=/path/to/ace/files
   export MCDC_LIB=/path/to/mcdc/library

   cd tools/data_library_generator
   python generate.py

By default the tool only converts nuclides that do not already have a corresponding
HDF5 file in ``$MCDC_LIB``.  Use ``--rewrite`` to regenerate all files, or
``--verbose`` for detailed per-nuclide output:

.. code-block:: sh

   python generate.py --rewrite --verbose

The generator processes each ACE file as follows:

#. Reads the ACE header to determine nuclide identity (Z, A, isomeric state)
   and temperature.
#. Extracts the principal cross-section block (energy grid, elastic, capture,
   fission, inelastic channels) and writes them as HDF5 datasets grouped by
   reaction type (elastic scattering, capture, inelastic scattering, fission).
#. Extracts angular distributions (tabulated cosine PDFs) and energy
   distributions (level scattering, evaporation, Maxwellian, Kalbach-Mann,
   N-body phase space, tabulated outgoing energy) for each reaction channel.
#. For fissionable nuclides, extracts prompt/delayed :math:`\nu(E)` multiplicities,
   delayed neutron precursor fractions, decay constants, and energy spectra.

The resulting HDF5 file (e.g., ``U235-293.6K.h5``) is ready for use with
``mcdc.Material()``.


Using CE Materials in an Input Deck
------------------------------------

Once the library is generated, set the ``MCDC_LIB`` environment variable and
define materials with ``mcdc.Material()``:

.. code-block:: python3

   import mcdc

   # Define nuclides with atom densities (atoms/barn-cm)
   fuel = mcdc.Material(
       nuclides=["U235", "U238", "O16"],
       density=[5.58e-4, 2.24e-2, 4.583e-2],
       temperature=293.6,
   )

MC/DC will automatically look up the matching HDF5 file in ``$MCDC_LIB``
(e.g., ``U235-293.6K.h5``) and load the pointwise cross sections.

See the :ref:`example_pincell` for a complete continuous-energy input deck.


Note on External Data Sources
------------------------------

MC/DC's internal HDF5 format is independent of the original data source.
While the shipped tool converts from **ACE format**, users with data in other
formats (e.g., OpenMC HDF5 nuclear data) can write their own converter
following the same HDF5 schema used by ``generate.py``.

The key HDF5 structure expected by MC/DC is:

.. code-block:: text

   <Nuclide>-<Temperature>K.h5
   ├── nuclide_name              (string)
   ├── temperature               (float, K)
   ├── atomic_weight_ratio       (float)
   ├── fissionable               (bool)
   └── neutron_reactions/
       ├── xs_energy_grid         (1-D array, MeV)
       ├── elastic_scattering/
       │   └── MT-002/
       │       ├── xs             (1-D array, barns)
       │       ├── cosine/        (angular distribution)
       │       └── energy/        (energy distribution)
       ├── capture/
       │   └── MT-102/ ...
       ├── inelastic_scattering/
       │   └── MT-051/ ...
       └── fission/
           └── MT-018/
               ├── xs
               ├── cosine/
               ├── energy/
               ├── nu_total/
               ├── nu_prompt/
               ├── nu_delayed/
               └── delayed_neutron/ ...

A converter from OpenMC's ``IncidentNeutron`` HDF5 format to this schema is
a planned future addition.  Contributions are welcome — see
`Issue #333 <https://github.com/CEMeNT-PSAAP/MCDC/issues/333>`_.
