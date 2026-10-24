===========
Conventions
===========

Identification and versioning
-----------------------------

Every file has the following required attributes on its root group ``/``.

.. list-table:: Root attributes
   :header-rows: 1

   * - Attribute
     - Type / shape
     - Value
   * - ``format``
     - string
     - ``mcdc-data``
   * - ``format-version``
     - string
     - ``0.1.0``

The version uses ``major.minor.patch`` notation.
A major version introduces incompatible changes, a minor version adds backward-compatible features, and a patch version corrects the specification without changing compatibility.
The format version is independent of MC/DC software versions and source-library releases.
Readers must check both attributes and report unsupported formats or versions explicitly.
A shared major version alone does not guarantee that a reader supports every representation introduced by a later minor version.

Source libraries
----------------

Every file contains a ``/source-libraries`` group with one subgroup per contributing library at ``/source-libraries/{library}``.
Subgroup names identify distinct library releases using identifiers from the source files or accompanying library metadata.
Shared files retain the records of their contributing libraries.

Each library subgroup contains the following scalar string datasets.
Missing or inapplicable information is an empty string.

.. list-table:: Source-library fields
   :header-rows: 1
   :widths: 30 70

   * - Dataset
     - Meaning
   * - ``title``
     - Original source title, when applicable to the file's target.
   * - ``version``
     - Library release version.
   * - ``release-date``
     - Library release date.
   * - ``reference``
     - Source-library reference.
   * - ``comments``
     - Additional source information.
   * - ``format``
     - Source data format, such as ``ACE``.
   * - ``format-version``
     - Source data format version.
   * - ``processing-date``
     - Date recorded for source data processing.
   * - ``conversion-time``
     - UTC timestamp of this library's latest contribution to the file, in ISO 8601 form ``YYYY-MM-DDTHH:MM:SSZ``.

``conversion-time`` is recorded when data is written and remains unchanged when conversion is skipped.
Nuclide-temperature files additionally include scalar ``temperature`` (K) and ``kT`` (eV) datasets in each applicable source-library subgroup.
These preserve the source's nominal temperature and thermal energy, with a ``unit`` attribute on each dataset.

Names and paths
---------------

Paths beginning with ``/`` are absolute HDF5 paths.
Other paths are relative to the object being described.

Names use hyphens between words.
Element symbols retain their standard capitalization, such as ``Fe``.
Nuclide identifiers append the mass number, such as ``Fe56``.
An excited-state identifier appends ``m`` and its nonzero state index, such as ``Co58m1``.
The temperature token is a decimal value in kelvin and must agree with the file's ``temperature`` dataset.

Indexed records use contiguous zero-based decimal string names: ``0``, ``1``, and so on.
An index embedded in a name follows the same rule, for example ``energy-spectrum-0``.
Record identity follows the numeric index, not HDF5 iteration order or lexicographic sorting.
Indices shared by related arrays and groups identify the same record.

Datatypes and shapes
--------------------

The tables specify HDF5 logical datatypes.
``float`` denotes a 64-bit floating-point value, ``integer`` an integer value, ``boolean`` a true/false value, and ``string`` text.
Integer readers must handle the stored signedness and width without assuming a native machine integer.
Strings may use fixed-length or variable-length HDF5 string storage; readers must handle either and decode text using the declared encoding.
Booleans use an HDF5 boolean enumeration.
Byte order, compression, chunking, and object creation order do not change a field's meaning.

An unqualified dataset is scalar, with shape ``()``.
Brackets denote array dimensions: ``[K]`` is a one-dimensional array, and ``[B, S]`` is a two-dimensional array with bins along the first axis and spectra along the second.
Dimension symbols are local to the table or representation where they appear.
A reusable type expands to a group at the named field.

Fields listed for a selected representation are required unless marked optional or conditional.
Category-specific fields are required only for their category.
An omitted optional field means unspecified data; it does not imply zero or a default distribution.
An empty collection is represented by an empty group or a length-zero dataset where explicitly allowed.

Units
-----

Every dimensional dataset carries a string attribute ``unit``.
Dimensionless datasets omit the ``unit`` attribute.
Angles are represented by their dimensionless cosines.
Multiplicities, yields, probabilities, and mass ratios are dimensionless.

.. list-table:: Unit spellings
   :header-rows: 1

   * - Quantity
     - ``unit``
   * - Energy, Q-value, thermal energy, spectral temperature
     - ``eV``
   * - Temperature
     - ``K``
   * - Microscopic cross section
     - ``barn``
   * - Energy probability density
     - ``1/eV``
   * - Decay rate
     - ``1/s``
   * - Radiation length expressed as areal mass
     - ``g/cm2``
   * - Mass stopping power
     - ``eV cm2/g``
   * - Momentum-transfer coordinate used by the photon tables
     - ``1/angstrom``

A polynomial coefficient array can contain quantities with different units.
Its ``unit`` attribute is then a string array with one entry per coefficient, using ``1`` for a dimensionless entry and, for example, ``1/eV`` or ``1/eV^2`` for higher-order coefficients of a dimensionless function of energy.
An entirely dimensionless coefficient array omits the attribute.
Units belong to each dataset, including the grids and PDFs nested inside reusable representations.
