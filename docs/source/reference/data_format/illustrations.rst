====================
Layout Illustrations
====================

These annotated YAML illustrations show the :doc:`index`.
They describe HDF5 layouts; they are not YAML input files for a converter.
The illustrations show a selection of reactions and optional fields, not an exhaustive list of channels or a requirement that every file contain every illustrated block.
See :doc:`conventions` for general format rules, :doc:`representations` for reusable types, and :doc:`reactions` for field requirements.

Notation
--------

- Names ending in ``/`` denote HDF5 groups; the slash is not part of the stored name.
- Names beginning with ``@`` denote attributes on the containing object; the ``@`` is not stored.
- Other fields denote datasets unless they reference a reusable group representation such as ``Data`` or ``CrossSection``.
- Braced names such as ``{library}``, ``{i}``, and ``{subshell}`` are placeholders.

Values such as ``float [N], eV`` describe a dataset's type, shape, and unit rather than literal stored values.
Inline comments beginning with ``#`` describe individual fields.

Neutron file
------------

``neutron-{nuclide}-{temperature}K.h5`` contains target properties and temperature-dependent neutron data.

:download:`Download layout-neutron.yaml <illustrations/layout-neutron.yaml>`.

.. literalinclude:: illustrations/layout-neutron.yaml
   :language: yaml

Photon file
-----------

``photon-{element}.h5`` contains element properties and photon data.

:download:`Download layout-photon.yaml <illustrations/layout-photon.yaml>`.

.. literalinclude:: illustrations/layout-photon.yaml
   :language: yaml

Electron file
-------------

``electron-{element}.h5`` contains element properties and electron data.

:download:`Download layout-electron.yaml <illustrations/layout-electron.yaml>`.

.. literalinclude:: illustrations/layout-electron.yaml
   :language: yaml

Proton file
-----------

``proton-{nuclide}.h5`` contains target properties and proton data; other nuclear projectiles use the same naming pattern.

:download:`Download layout-proton.yaml <illustrations/layout-proton.yaml>`.

.. literalinclude:: illustrations/layout-proton.yaml
   :language: yaml

Atomic library
--------------

``atoms.h5`` contains element-indexed atomic structure, relaxation data, and evaluation provenance.

:download:`Download layout-atoms.yaml <illustrations/layout-atoms.yaml>`.

.. literalinclude:: illustrations/layout-atoms.yaml
   :language: yaml

Reusable data
-------------

The following illustrations show table, polynomial, and constant scalar functions.
Their fields and units are defined in :doc:`representations`.

:download:`Download layout-data.yaml <illustrations/layout-data.yaml>`.

.. literalinclude:: illustrations/layout-data.yaml
   :language: yaml

Reusable distributions
----------------------

The following illustrations show probability tables, a multi-table distribution, joint energy-angle distributions, and a mixture of spectra.
Their fields and offset conventions are defined in :doc:`representations`.
For angular distributions, sampled coordinates and densities are dimensionless and omit unit attributes.

:download:`Download layout-distributions.yaml <illustrations/layout-distributions.yaml>`.

.. literalinclude:: illustrations/layout-distributions.yaml
   :language: yaml
