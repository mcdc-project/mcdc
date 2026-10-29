.. _proton-data-preparation:

======
Proton
======

The generator in ``tools/data_library_generator/proton/`` converts continuous-energy proton ACE evaluations into :doc:`MC/DC transport data <../../reference/data_format/index>`.
It writes one temperature-independent ``proton-{nuclide}.h5`` file per target nuclide, following the :doc:`proton file layout <../../reference/data_format/illustrations>`.
Optional NIST PSTAR tables provide proton mass stopping power.
H-1, H-2, He-3, and He-4 stopping-power files are also created when their PSTAR data exist but no ACE evaluation was selected.

Requirements and usage
----------------------

Use a Python environment with ACEtk, ``h5py``, ``numpy``, and ``tqdm``.
Install ACEtk following its `project instructions <https://github.com/njoy/ACEtk>`_.
Proton ACE evaluations are available from sources such as `TENDL-2021 <https://tendl.imperial.ac.uk/tendl_2021/tar.html>`_.
The generator expects extracted ``*.ace`` files in one directory.

PSTAR stopping-power input is optional per element, but the ``MCDC_PSTAR_LIB`` directory must exist.
Place one text file per element in that directory, named by chemical symbol (for example ``Al.txt``).
Each data row contains energy in MeV and total mass stopping power in MeV cm2/g.
Rows must have at least two points with positive, strictly increasing energies and nonnegative stopping powers; non-data lines are ignored.

From the repository root, set the source and output directories and run the generator:

.. code-block:: sh

   export MCDC_ACELIB_PROTON="/path/to/tendl/proton/ace"
   export MCDC_PSTAR_LIB="/path/to/pstar/tables"
   export MCDC_LIB_PROTON="/path/to/mcdc/library"

   python tools/data_library_generator/proton/generate.py

.. list-table:: Command-line options
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Behavior
   * - ``--only p-Al027.ace p-Si028.ace``
     - Convert only these filenames from ``MCDC_ACELIB_PROTON``.
   * - ``--rewrite``
     - Regenerate existing output files.
   * - ``--verbose``
     - Print each conversion, reaction summary, and warning instead of showing the progress bar.
   * - ``--keep-going``
     - Continue after conversion errors and summarize failed ACE and PSTAR inputs at the end.
   * - ``--source-library NAME``
     - Set the ACE provenance label described in :doc:`index`; PSTAR retains its own record.

Existing compatible outputs are reused unless ``--rewrite`` is supplied.
An input that cannot be represented is reported as an error and does not publish a partial replacement.
See :doc:`index` for shared batch error handling.
For simulations, make the output directory available through ``MCDC_LIB`` (or place the generated files in the configured library directory).

Prepared data
-------------

Each ACE-backed file contains the target atomic number, mass number, atomic-weight ratio, excitation level, fission flag, radiation length, provenance, the shared incident-energy grid, and proton reactions.
Energies and Q-values are converted from MeV to eV; cross sections remain in barns.
Reactions include elastic scattering, capture, or inelastic scattering, with their source MT, reference frame, cross section, Q-value, and applicable angular and outgoing-energy distributions.
Energy-dependent outgoing-proton multiplicity is stored as a tabulated ``Data`` record; constant multiplicity is stored as an integer.

Additional emitted particles are stored as indexed ``cross-products`` with a yield table, reference frame, and angle-energy distribution.
The generator handles the ACE secondary neutron, deuteron, triton, helion, alpha, and photon blocks, preserving product frames even when they differ from the reaction frame.
Unsupported particle identifiers, distribution types, or fission data are reported as conversion errors rather than silently omitted.

When a matching PSTAR file is present, mass stopping power is written as a tabulated ``stopping-power`` record on its own energy grid.
Its grid is in eV and its values are in eV cm2/g.
This field is optional for analog proton transport and required for condensed-history proton energy loss.

Reaction coverage
-----------------

Elastic scattering uses the principal ACE cross-section block (MT 2).
Reaction channels with zero outgoing-proton multiplicity are classified as capture; channels with positive or energy-dependent multiplicity are inelastic scattering.
Aggregate MT 1, 3, 10, and 101 are omitted.
An aggregate channel is also omitted when its corresponding partial channels are present; channels above MT 891 are not written.
Fission data is rejected.

Validation
----------

The output uses the :doc:`proton illustration <../../reference/data_format/illustrations>` and shared :doc:`format conventions <../../reference/data_format/conventions>`.
Unsupported source representations and inconsistent ACE block data are reported as errors.
