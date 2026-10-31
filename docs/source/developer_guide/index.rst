.. _developer_guide:

===============
Developer Guide
===============

The Developer Guide explains how MC/DC works internally and how to extend or maintain its implementation.

.. _development_areas:

Development Areas
-----------------

MC/DC development spans two complementary areas.

**Methods development** covers the implementation and evaluation of transport methods, numerical algorithms, and their model-facing data.
This work primarily uses ``mcdc/object_`` and ``mcdc/transport``.

**Framework development** covers the machinery that compiles models, generates runtime layers and accessors, configures execution, and supports CPU and GPU backends.
This work primarily uses ``mcdc/code_factory``, ``mcdc/main.py``, and related execution infrastructure.

Where to Go Next
----------------

- Read :doc:`architecture/index` to understand MC/DC's Python-first design and follow a transport model through simulation compilation, runtime preparation, shared transport algorithm, and the available execution modes.
- Read :doc:`extending/index` when adding model fields, registered objects, polymorphic subtypes, or Numba-compatible transport behavior.
- Read :doc:`documentation/index` when writing or reviewing project documentation.
- Use the :doc:`MC/DC data format reference <../reference/data_format/index>` when preparing transport data or working with library files.
- Use :doc:`contributing/index` for repository setup, development workflow, testing, and pull-request requirements.
- Use :doc:`release_policy` when preparing, validating, and publishing a release.

.. toctree::
   :maxdepth: 1

   architecture/index
   extending/index
   documentation/index
   release_policy
   contributing/index
