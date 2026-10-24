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

Element file
------------

``{element}.h5`` contains element identity, atomic relaxation, and electron and photon data.

:download:`Download layout-element.yaml <illustrations/layout-element.yaml>`.

.. literalinclude:: illustrations/layout-element.yaml
   :language: yaml

Nuclide file
------------

``{nuclide}.h5`` contains nuclide identity and temperature-independent projectile data, illustrated with protons.

:download:`Download layout-nuclide.yaml <illustrations/layout-nuclide.yaml>`.

.. literalinclude:: illustrations/layout-nuclide.yaml
   :language: yaml

Nuclide-temperature file
------------------------

``{nuclide}-{temperature}K.h5`` contains temperature-dependent data, illustrated with neutrons.

:download:`Download layout-nuclide-temperature.yaml <illustrations/layout-nuclide-temperature.yaml>`.

.. literalinclude:: illustrations/layout-nuclide-temperature.yaml
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

The following illustrations show continuous, discrete, and mixed tables, a multi-table distribution, and a joint energy-angle distribution.
Their fields and offset conventions are defined in :doc:`representations`.
For angular distributions, sampled coordinates and densities are dimensionless and omit unit attributes.

:download:`Download layout-distributions.yaml <illustrations/layout-distributions.yaml>`.

.. literalinclude:: illustrations/layout-distributions.yaml
   :language: yaml
