.. _neutron-data-preparation:

=======
Neutron
=======

The generator in ``tools/data_library_generator/neutron/`` converts continuous-energy neutron ACE files into :doc:`MC/DC transport data <../../reference/data_format/index>`.
It writes one ``neutron-{nuclide}-{temperature}K.h5`` file per evaluation, including target properties and provenance, using the :doc:`specified file layouts <../../reference/data_format/file_layouts>`.

Requirements and usage
----------------------

Use a Python environment with `ACEtk <https://github.com/njoy/ACEtk>`_, ``h5py``, ``numpy``, and ``tqdm``.

From the repository root, set the input and output directories and run the generator:

.. code-block:: sh

   export MCDC_LIB="/path/to/output/mcdc-library"
   export MCDC_ACE_LIB="/path/to/input/ace-library"

   python tools/data_library_generator/neutron/generate.py

.. list-table:: Command-line options
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Behavior
   * - ``--ace-files 1002.10c 88223.10c 92240.10c``
     - Convert selected filenames from ``MCDC_ACE_LIB``; omit the option to process all files in that directory.
   * - ``--rewrite``
     - Regenerate existing neutron files.
   * - ``--verbose``
     - Report filenames and reaction details during conversion.
   * - ``--keep-going``
     - Continue after conversion errors and summarize failed files at the end.
   * - ``--source-library NAME``
     - Set the ACE provenance label described in :doc:`index`.

Existing compatible data is reused unless ``--rewrite`` is supplied.
Incompatible formats or conflicting identity fields are reported as errors.
A failed rewrite preserves the previous neutron file.
See :doc:`index` for shared batch error handling.
Distribution errors identify the reaction, distribution path, ACE law, and table coordinates when available.

Temperature
-----------

Output filenames include the target temperature in kelvin.
Temperature is derived from the ACE header's ``kT``, except where a recognized library supplies a nominal temperature for the ZAID suffix.

For Lib81 and Lib81x, filename suffixes map to nominal temperatures as follows:

.. list-table:: Lib81 and Lib81x temperatures
   :header-rows: 1

   * - Suffix
     - Temperature (K)
   * - ``.10c``
     - 293.6
   * - ``.11c``
     - 600.0
   * - ``.12c``
     - 900.0
   * - ``.13c``
     - 1200.0
   * - ``.14c``
     - 2500.0
   * - ``.15c``
     - 0.1
   * - ``.16c``
     - 233.15
   * - ``.17c``
     - 273.15

Prepared data
-------------

Prepared data includes neutron reaction cross sections, angular and energy distributions, and prompt and delayed fission-neutron data.
See the :doc:`reaction reference <../../reference/data_format/reactions>` for categories and fields, and the :doc:`distribution reference <../../reference/data_format/representations>` for distribution layouts.
The :doc:`ACE distribution support reference <../../reference/ace_distributions>` lists supported laws, their representations, and conversion restrictions.

Additional particle production, damage data, unresolved-resonance probability tables, and thermal-scattering laws are not included yet.
Fission tables with delayed-neutron yields must also provide explicit prompt-neutron yields.
Unsupported ACE representations are reported as errors.

Units and source-library metadata follow the :doc:`format conventions <../../reference/data_format/conventions>`.
ACE source-library labels follow :doc:`index`.

Reaction coverage
-----------------

Reaction data covers MT 1–117 and MT 152–200, with the aggregate handling described below.
MT 152–200 include multi-neutron emission, such as ``(n,5n)`` through ``(n,8n)``, and reactions emitting both neutrons and charged particles.
Their neutron reaction contributions are retained even when additional particle production is not included.
Non-fission channels with no outgoing neutron use the ``capture`` category; within MT 152–200, these are MT 155, 182, 191–193, and 197.

Aggregate MT 1 (total), 3 (nonelastic), 4 (inelastic), and 10 (continuum) are omitted to avoid counting their contributions alongside individual channels.
Fission is represented by either total MT 18 or the available chance-fission channels MT 19–21 and 38.
Inputs containing both total and chance-fission cross sections are rejected.
Inclusive particle-production totals, such as MT 201–207, are not included as transport reactions.

Aggregate channels
^^^^^^^^^^^^^^^^^^

The following resolved channels are represented by aggregate reactions supplied in the source ACE data.
The individual branches are not retained separately.

.. list-table:: Aggregate representations
   :header-rows: 1
   :widths: 30 25 45

   * - Resolved MTs
     - Aggregate MT
     - Reaction
   * - 600–649
     - 103
     - ``(n,p)``
   * - 650–699
     - 104
     - ``(n,d)``
   * - 700–749
     - 105
     - ``(n,t)``
   * - 750–799
     - 106
     - ``(n,³He)``
   * - 800–849
     - 107
     - ``(n,α)``
   * - 875–891
     - 16
     - ``(n,2n)``

Inputs must provide complete aggregate cross sections and any required neutron distributions; missing aggregates are not reconstructed from the resolved channels.
For MT 16, the aggregate neutron distributions are retained, while residual-state branch identities are not.
The aggregate relationships are defined in `Appendix B of the ENDF-6 specification <https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf>`_.

Validation
----------

The generator has been tested using ACE data from the `LANL-prepared Lib81 neutron library <https://nucleardata.lanl.gov/ace/lib81/>`_, including evaluations labeled ``Lib81x``.
