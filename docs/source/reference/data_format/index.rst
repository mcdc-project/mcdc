.. _mcdc-data-format:

=================
MC/DC Data Format
=================

``mcdc-data`` defines the HDF5 files used to exchange MC/DC transport data.
This reference describes format version **0.1.0**, including file organization, physical units, reusable representations, and reaction fields.
It is the specification for the format; the :doc:`annotated YAML illustrations <illustrations>` illustrate its structure.

Data is divided by target and temperature:

.. list-table:: File organization
   :header-rows: 1
   :widths: 30 40 30

   * - Filename
     - Contents
     - Example
   * - ``{element}.h5``
     - Element identity, atomic relaxation, and element-based projectile data
     - ``Fe.h5``
   * - ``{nuclide}.h5``
     - Nuclide identity and temperature-independent projectile data
     - ``Fe56.h5``
   * - ``{nuclide}-{temperature}K.h5``
     - Temperature and temperature-dependent projectile data
     - ``Fe56-293.6K.h5``

Each file identifies its format and version through root attributes and its contributing libraries through the ``source-libraries`` group.
Each projectile has a named group under ``/projectiles`` within the appropriate target file.
For example, neutron cross sections reside under ``/projectiles/neutron`` in a nuclide-temperature file.

HDF5 terminology follows the `HDF Group's data model <https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html>`_: groups organize objects, datasets store typed arrays, and attributes attach metadata to an object.
The scientific meaning and relationships of those objects are defined here by MC/DC.

Use `myHDF5 <https://myhdf5.hdfgroup.org>`_, hosted by The HDF Group, to browse file groups, datasets, and attributes and visualize data directly in a web browser.

.. toctree::
   :maxdepth: 1

   conventions
   file_layouts
   reactions
   representations
   illustrations
