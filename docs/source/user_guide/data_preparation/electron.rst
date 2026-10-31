.. _electron-data-preparation:

========
Electron
========

The generator in ``tools/data_library_generator/electron/`` converts `LANL EPRDATA14 <https://nucleardata.lanl.gov/ace/eprdata14>`_ ACE tables into :doc:`MC/DC transport data <../../reference/data_format/index>`.
It writes ``electron-{element}.h5`` and fills missing element records in the shared ``atoms.h5``.

Requirements and usage
----------------------

Use a Python environment with `ACEtk <https://github.com/njoy/ACEtk>`_, ``h5py``, ``numpy``, and ``tqdm``.
Download EPRDATA14 and set ``MCDC_ACE_EPRDATA_LIB`` to the directory containing its concatenated ACE file, rather than the archive's documentation directory.
The generator selects elemental tables with the ``.14p`` suffix.

From the repository root:

.. code-block:: sh

   export MCDC_LIB="/path/to/output/mcdc-library"
   export MCDC_ACE_EPRDATA_LIB="/path/to/input/EPRDATA14/eprdata14"

   python tools/data_library_generator/electron/generate.py --ace-files eprdata14

.. list-table:: Command-line options
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Behavior
   * - ``--elements H Mg Si``
     - Convert selected chemical symbols; omit to process all EPRDATA14 elements.
   * - ``--ace-files eprdata14``
     - Select input filenames; omit to process every regular file in the input directory.
   * - ``--rewrite``
     - Regenerate electron projectile files.
   * - ``--source-library NAME``
     - Override the ``EPRDATA14`` provenance label while retaining the original header metadata.
   * - ``--verbose``
     - Report filenames and reactions during conversion.
   * - ``--keep-going``
     - Continue after conversion errors and summarize failed inputs at the end.

Existing compatible electron files are reused unless ``--rewrite`` is supplied.
Existing atomic records are retained even when electron files are rewritten, so preparing electrons does not replace the atomic evaluation used by photon transport.
Only atomic records contributed by this generator add EPRDATA14 provenance to ``atoms.h5``.
See :doc:`index` for shared batch error handling and :doc:`../../reference/data_format/conventions` for provenance conventions.

Prepared data
-------------

The generator retains total elastic scattering (MT 526), excitation (MT 528), bremsstrahlung (MT 527), and total ionization (MT 522).
Ionization is the sum of subshell channels, with each subshell retaining its cross section, binding energy, designator, and knock-on spectrum.
Elastic data includes the large-angle component (MT 525), angular distributions, and the transport cross section used to scale deflections between angular tables.
Excitation and bremsstrahlung retain their mean energy losses, and bremsstrahlung also retains its emitted-photon spectrum.

Source cross-section and parameter grids are retained without resampling.
Energies are converted from MeV to eV, and cross sections are in barns.
Angular and outgoing-energy CDFs use the packed :ref:`MultiTable representation <distribution-multi-table>`.
Electron sampling uses the EPR CDF knots: elastic angles retain their deflection scaling, bremsstrahlung interpolates sampled energy fractions in log incident energy, and ionization retains its binding-energy threshold and knock-on energy interpolation.
See the :doc:`electron reaction fields <../../reference/data_format/reactions>` for the stored quantities.

Atomic preparation retains subshell identities, electron populations, binding energies, and radiative and non-radiative transitions.
EPRDATA14 cumulative transition probabilities are converted to the absolute branching probabilities in the :doc:`shared atomic layout <../../reference/data_format/file_layouts>`.

Validation
----------

The generator has been tested against all 100 elements in the `LANL-prepared EPRDATA14 library <https://nucleardata.lanl.gov/ace/eprdata14>`_.
