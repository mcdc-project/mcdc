.. _data-preparation:

================
Data Preparation
================

The generators in ``tools/data_library_generator/`` convert source libraries into the :doc:`MC/DC data format <../../reference/data_format/index>`.
Each projectile guide describes its source libraries, requirements, commands, and conversion rules.
The :doc:`format conventions <../../reference/data_format/conventions>` and :doc:`YAML illustrations <../../reference/data_format/illustrations>` describe the shared output structure.

.. list-table:: Source libraries
   :header-rows: 1
   :widths: 20 60 20

   * - Projectile
     - Source libraries
     - Source format
   * - :doc:`Neutron <neutron>`
     - Processed continuous-energy neutron libraries, such as `LANL Lib81 <https://nucleardata.lanl.gov/ace/lib81/>`_.
     - ACE
   * - :doc:`Photon <photon>`
     - `EPICS2025 <https://nuclear.llnl.gov/EPICS/index.html>`_: EPDL for photon interactions and optional EADL for shared atomic-relaxation data.
     - ENDF
   * - :doc:`Electron <electron>`
     - `LANL EPRDATA14 <https://nucleardata.lanl.gov/ace/eprdata14>`_: electron interactions and atomic-relaxation data based on EPICS2014.
     - ACE
   * - :doc:`Proton <proton>`
     - `TENDL2021 <https://tendl.imperial.ac.uk/tendl_2021/tendl2021.html>`: TENDL2021 ACE data
     - `NIST PSTAR <https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html>`: stopping power tables

Input and output locations
--------------------------

Each ACE generator has an input variable that can remain configured alongside the others.
Release names belong in directory paths and recorded provenance.

.. list-table:: ACE input directories
   :header-rows: 1

   * - Variable
     - Input
   * - ``MCDC_ACE_NEUTRON_LIB``
     - Neutron ACE files, such as Lib81 or processed TENDL neutron data.
   * - ``MCDC_ACE_PROTON_LIB``
     - Proton ACE files, such as processed TENDL proton data.
   * - ``MCDC_ACE_EPRDATA_LIB``
     - Concatenated electron/photon/relaxation ACE files.

``MCDC_ACE_LIB``, when set, overrides the selected generator's input variable.
Use it for a particular invocation, for example:

.. code-block:: sh

   MCDC_ACE_LIB="/path/to/another/neutron-library" python tools/data_library_generator/neutron/generate.py

All generators write to ``MCDC_LIB``.
Photon EPICS inputs use ``MCDC_EPICS_LIB``, and proton stopping-power inputs use ``MCDC_PSTAR_LIB``.

Batch conversion
----------------

The neutron, photon, electron, and proton generators stop at the first conversion error by default.
Use ``--keep-going`` to process the remaining inputs and summarize all failed conversions at the end.
Successful conversions are retained, and the command returns a nonzero exit status if any conversion fails.
Error messages identify the source input and include reaction or distribution context when available.

ACE source-library labels
-------------------------

The neutron and proton generators preserve the original ACE header metadata for every converted evaluation.
Use ``--source-library NAME`` to set the provenance subgroup at ``source-libraries/NAME``.
Without this option, the generator recognizes header labels matching ``Lib<number>[x]`` (such as ``Lib81`` or ``Lib81x``) and ``TENDL-YYYY`` or ``TENDL YYYY`` (such as ``TENDL-2021``).
LANL labels are preserved, and TENDL labels are stored as ``TENDL-YYYY``; other inputs use ``unknown``.
Library recognition supplies metadata defaults and library-specific conventions; supported ACE representations can be converted from other libraries as well.

.. toctree::
   :maxdepth: 1

   neutron
   photon
   electron
   proton
