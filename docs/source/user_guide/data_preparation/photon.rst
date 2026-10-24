.. _photon-data-preparation:

======
Photon
======

The generator in ``tools/data_library_generator/photon/`` converts EPDL photon evaluations and EADL atomic-relaxation evaluations into :doc:`MC/DC transport data <../../reference/data_format/index>`.
It writes photon data under ``/projectiles/photon`` and shared atomic-relaxation data under ``/atomic-relaxation`` in each :doc:`element file <../../reference/data_format/file_layouts>`.

Requirements and usage
----------------------

Use a Python environment with ``h5py``, ``numpy``, and ``tqdm``.
The source files are per-element ENDF evaluations from `EPICS <https://nuclear.llnl.gov/EPICS/index.html>`_.
Download the `EPDL <https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL.ELEMENTS/getza.htm>`_ and `EADL <https://nuclear.llnl.gov/EPICS/ENDF2025/EADL.ELEMENTS/getza.htm>`_ files and save them directly in one directory as ``EPDL.ZA<ZZZ>000.endf`` and ``EADL.ZA<ZZZ>000.endf``, where ``ZZZ`` is the three-digit atomic number.
For example, magnesium uses ``EPDL.ZA012000.endf`` and ``EADL.ZA012000.endf``.

From the repository root, set the input and output directories and run the generator:

.. code-block:: sh

   export MCDC_LIB="/path/to/output/mcdc-library"
   export MCDC_EPICS_LIB="/path/to/input/epics2025"

   python tools/data_library_generator/photon/generate.py

.. list-table:: Command-line options
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Behavior
   * - ``--elements H Mg Si``
     - Convert selected chemical symbols; omit the option to process all EPDL element files in the input directory.
   * - ``--rewrite``
     - Regenerate photon data and any supplied atomic-relaxation data.
   * - ``--verbose``
     - Report filenames and reaction details during conversion.

Existing compatible photon data is reused unless ``--rewrite`` is supplied.
Other projectile data and source-library records in shared element files are retained.
Incompatible formats or conflicting atomic numbers are reported as errors.
A failed conversion preserves the previous element file.

EADL files are optional.
When a matching EADL file is supplied, its target must agree with the EPDL target.
Without EADL input, an existing atomic-relaxation group is retained; a new element file has no atomic-relaxation data.

Prepared data
-------------

Prepared data includes coherent scattering, incoherent scattering, photoelectric absorption, and electron-field and nuclear-field pair production.
Form factors, scattering functions, and subshell photoelectric cross sections retain their own source grids and interpolation rules.
Atomic relaxation includes subshell identities, binding energies, electron populations, and radiative and non-radiative transition tables.
Terminal subshells have empty transition arrays.
See the :doc:`photon reaction reference <../../reference/data_format/reactions>` and :doc:`atomic-relaxation layout <../../reference/data_format/file_layouts>` for fields and meanings.

The shared photon cross-section grid includes the source reaction knots.
Repeated energies retain the values on both sides of absorption edges rather than smoothing the discontinuity.
Energies are in eV, cross sections in barns, and momentum-transfer coordinates in inverse angstroms.
Source-library records identify EPDL and EADL releases independently and follow the :doc:`format conventions <../../reference/data_format/conventions>`.

Reaction coverage
-----------------

The generator retains MT 502 (coherent scattering), 504 (incoherent scattering), 522 (photoelectric absorption), 515 (electron-field pair production), and 517 (nuclear-field pair production).
Aggregate MT 501 and 516 are not written as additional transport reactions, avoiding double counting alongside their component channels.
Photoelectric subshell data comes from MF 23/MT 534–599, scattering tables from MF 27, and atomic relaxation from MF 28/MT 533.
Anomalous scattering factors and cross-species product distributions are not included yet.

Validation
----------

The generator has been tested using ENDF-formatted EPDL and EADL data from `LLNL EPICS2025 <https://nuclear.llnl.gov/EPICS/index.html>`_.
