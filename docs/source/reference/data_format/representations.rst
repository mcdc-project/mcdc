========================
Reusable Representations
========================

``CrossSection``, ``Data``, ``Distribution``, and ``Interpolation`` describe reusable group layouts.
These type names identify the layouts expanded at individual fields.
The reusable representations are designed to be as consistent as possible with their counterparts in MC/DC's ``mcdc.object_`` and ``mcdc.transport`` modules.
``Data`` and ``Distribution`` select a representation through a required scalar string attribute ``type`` on the containing group.
See :doc:`../ace_distributions` for the mapping from ACE laws to these representations and their object and transport implementations.

CrossSection
------------

A reaction's ``xs`` group stores only the values needed on its projectile's shared ``xs-energy-grid``.
Let N be the shared grid length and o the offset.

.. list-table:: CrossSection fields
   :header-rows: 1

   * - Field
     - Type / shape / unit
     - Meaning
   * - ``offset``
     - integer
     - Zero-based index o of the first stored cross section on the shared grid.
   * - ``value``
     - float [N - o], barn
     - Cross-section values corresponding to ``xs-energy-grid[o:]``.
   * - ``interpolation``
     - Interpolation
     - Interpolation over the stored values.

The offset is nonnegative and leaves at least two stored grid points.
Cross sections are nonnegative and are zero at shared-grid points below the offset.
Interpolation boundaries index the local ``value`` array, starting at zero, rather than the full shared grid.
CrossSection does not carry its own grid or a ``type`` attribute.

Data
----

Data represents a scalar-valued function of an input coordinate.
The containing field defines the input and output quantities, such as incident energy in eV and neutron multiplicity.
Each representation below lists all of its datasets.

.. list-table:: Data representations
   :header-rows: 1
   :widths: 20 40 40

   * - ``type`` attribute
     - Fields
     - Meaning
   * - ``table``
     - ``grid``: float [K]; ``value``: float [K]; ``interpolation``: Interpolation
     - Values on a nondecreasing input grid with K >= 2 and at least two distinct coordinates.
   * - ``polynomial``
     - ``coefficients``: float [L]
     - Coefficients in ascending powers of the input, with L >= 1.

A polynomial represents

.. math::

   f(x) = \sum_{l=0}^{L-1} c_l x^l.

The coefficient at index zero is the constant term.
A constant uses ``type = polynomial`` with a single coefficient.
Coefficient units are output units divided by input units raised to the coefficient's index.
For example, a dimensionless multiplicity polynomial of incident energy in eV has coefficient units ``1``, ``1/eV``, ``1/eV^2``, and so on.

Polynomial Data does not have interpolation fields.
Data tables require an explicit ``interpolation`` group.
No generic extrapolation rule is implied outside a table's grid range.
Repeated coordinates store the left and right limits of a discontinuity in that order.
Interpolation applies between distinct coordinates, not across the zero-width interval.

Interpolation
-------------

An ``interpolation`` group accompanies the table representation whose intervals it describes.
Let K be that representation's number of table points and R its number of interpolation regions.

.. list-table:: Interpolation fields
   :header-rows: 1

   * - Field
     - Type / shape
     - Meaning
   * - ``laws``
     - string [R]
     - One interpolation law per region.
   * - ``boundaries``
     - integer [R]
     - Exclusive upper-point index of each region.

The arrays have equal nonzero length.
Boundaries strictly increase, the first boundary is at least two, and the last equals K.
The first region begins at point zero; subsequent regions begin at the preceding boundary minus one.
Thus every interval belongs to exactly one region.
For example, boundaries ``[3, 6]`` apply the first law to intervals 0–1 and 1–2, then the second law to intervals 2–3, 3–4, and 4–5.

For neighboring points :math:`(x_0,y_0)` and :math:`(x_1,y_1)`, define :math:`t=(x-x_0)/(x_1-x_0)` and :math:`s=\ln(x/x_0)/\ln(x_1/x_0)`.

.. list-table:: Interpolation laws
   :header-rows: 1
   :widths: 20 45 35

   * - Law
     - Interpolated value
     - Domain condition
   * - ``histogram``
     - :math:`y=y_0` on the interval below :math:`x_1`
     - None beyond an increasing grid.
   * - ``linear``
     - :math:`y=(1-t)y_0+t y_1`
     - None beyond an increasing grid.
   * - ``semilog-x``
     - :math:`y=(1-s)y_0+s y_1`
     - Positive x coordinates.
   * - ``semilog-y``
     - :math:`y=y_0(y_1/y_0)^t`
     - Positive y values.
   * - ``log``
     - :math:`y=y_0(y_1/y_0)^s`
     - Positive x coordinates and y values.

Logarithms operate on dimensionless ratios.
Each table representation specifies its interpolation laws and any restrictions on them.

Distribution
------------

A Distribution describes sampled quantities in the reaction's declared reference frame.
Energy distributions supply outgoing energy in eV; angular distributions supply a dimensionless cosine relative to the incident direction; joint distributions supply both.
The layouts below define stored fields and constraints; :doc:`../ace_distributions` describes their physical models and sampling.

.. _distribution-table:

Probability table
^^^^^^^^^^^^^^^^^

``type = table`` stores a continuous PDF, discrete probability masses, or both.

.. list-table:: Table distribution fields
   :header-rows: 1

   * - Field
     - Type / shape
     - Meaning
   * - ``value``
     - float [K]
     - Continuous coordinates, required with ``pdf``; nondecreasing with positive support width and K >= 2.
   * - ``pdf``
     - float [K]
     - Continuous density, required with ``value``.
   * - ``interpolation``
     - Interpolation
     - Required for the continuous component only.
   * - ``discrete-value``
     - float [D]
     - Discrete outcomes, required with ``pmf``, D >= 1.
   * - ``pmf``
     - float [D]
     - Probability masses, required with ``discrete-value``.

At least one coordinate/probability pair is required.
Coordinates and probabilities are finite, and probabilities are nonnegative.
The PMF sum plus the PDF integral, evaluated under its interpolation laws, differs from one by at most :math:`10^{-6}`.
The components retain their combined probabilities rather than each being normalized to one.
Repeated continuous coordinates describe a density jump and carry no probability mass across their zero-width interval.
Discrete outcomes may lie inside or outside the continuous support.

``value`` and ``discrete-value`` carry the sampled coordinate's unit; ``pdf`` carries its inverse; ``pmf`` is dimensionless.
Cosine coordinates lie in [-1, 1], and both coordinates and densities are dimensionless.
Unit attributes follow :doc:`conventions`.
CDFs and cumulative probability masses are derived, not stored.

.. _packed-pdf-tables:
.. _packed-probability-tables:

Packed probability tables
^^^^^^^^^^^^^^^^^^^^^^^^^

This shared storage layout has no ``type`` attribute.
It packs the components of multiple :ref:`probability tables <distribution-table>` into concatenated arrays.
The containing distribution defines how the tables are indexed and which components are allowed.
Let T be the number of tables, P the total continuous points, D the total discrete entries, and R the total PDF interpolation regions.

.. list-table:: Packed probability fields
   :header-rows: 1

   * - Field
     - Type / shape
     - Meaning
   * - ``offsets``
     - integer [T + 1]
     - Start of each table's continuous points, followed by P.
   * - ``value``
     - float [P]
     - Concatenated continuous coordinates.
   * - ``pdf``
     - float [P]
     - Concatenated continuous densities.
   * - ``interpolation/offsets``
     - integer [T + 1]
     - Start of each table's interpolation regions, followed by R.
   * - ``interpolation/laws``
     - string [R]
     - Concatenated PDF interpolation laws.
   * - ``interpolation/boundaries``
     - integer [R]
     - Exclusive upper-point indices local to each table.
   * - ``discrete-offsets``
     - integer [T + 1]
     - Start of each table's discrete entries, followed by D.
   * - ``discrete-value``
     - float [D]
     - Concatenated discrete outcomes.
   * - ``pmf``
     - float [D]
     - Concatenated probability masses.

Supply all continuous fields when any table has a PDF, and all discrete fields when any table has a PMF; omit the fields of a component absent from every table.
Offset arrays start at zero, are nondecreasing, and end at their stated total.
Table i occupies ``offsets[i]:offsets[i + 1]`` in the continuous arrays and ``discrete-offsets[i]:discrete-offsets[i + 1]`` in the discrete arrays.
Repeated offsets represent an absent component; an absent PDF also has no interpolation regions.
Every nonempty PDF slice follows the standalone table's coordinate and interpolation constraints.
Interpolation boundaries are relative to the individual PDF slice, not the concatenated array.
Coordinates may restart at each table boundary.
Probability normalization and units follow the standalone table rules separately for each table.

Multi-table distribution
^^^^^^^^^^^^^^^^^^^^^^^^

``type = multi-table`` indexes packed probability tables by a shared grid.

.. list-table:: Multi-table fields
   :header-rows: 1

   * - Field
     - Type / shape
     - Meaning
   * - ``grid``
     - float [I]
     - Finite, nondecreasing indexing coordinates, I >= 2; incident energy uses eV.
   * - ``interpolation``
     - Interpolation
     - Interpolation between neighboring tables; histogram or linear only.
   * - ``tables``
     - Packed probability tables
     - One continuous, discrete, or mixed table per grid point, in grid order.

All tables describe the same sampled quantity and use the :ref:`packed layout <packed-probability-tables>` above.
Repeated grid coordinates describe the distributions on either side of a discontinuity.
Across each linear grid interval, discrete outcomes correspond by index and have equal counts.
See :ref:`ace-law-4` for line correspondence and :ref:`ace-incident-energy-sampling` for incident-table selection and support scaling.

Isotropic distribution
^^^^^^^^^^^^^^^^^^^^^^

``type = isotropic`` describes an angular cosine uniformly distributed on [-1, 1], with uniform azimuth.
It has no datasets.

Analytic energy spectra
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Analytic distribution fields
   :header-rows: 1
   :widths: 22 48 30

   * - ``type`` attribute
     - Fields
     - Model and sampling
   * - ``level-scattering``
     - ``C1``: float, eV; ``C2``: float
     - :ref:`ace-law-3`
   * - ``maxwellian``
     - ``temperature``: Data, eV; ``restriction-energy``: float, eV
     - :ref:`ace-law-7`
   * - ``evaporation``
     - ``temperature``: Data, eV; ``restriction-energy``: float, eV
     - :ref:`ace-law-9`

``temperature`` takes incident energy in eV as its input and returns a positive spectral temperature in eV.

Correlated energy-angle distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``type = kalbach-mann`` and ``type = energy-angle-table`` share these fields.

.. list-table:: Common correlated fields
   :header-rows: 1

   * - Field
     - Type / shape
     - Meaning
   * - ``energy``
     - float [I], eV
     - Strictly increasing incident energies.
   * - ``tables/outgoing-energy``
     - Packed probability tables, continuous-only
     - One outgoing-energy PDF per incident-energy point.

These representations have inherent linear incident-energy dependence and no outer interpolation group.
Their packed collections use the :ref:`shared layout <packed-probability-tables>` with every table containing a PDF and no PMF fields.

.. list-table:: Additional correlated fields
   :header-rows: 1

   * - ``type`` attribute
     - Field
     - Type
     - Meaning
   * - ``kalbach-mann``
     - ``tables/precompound-factor/{i}``
     - Data
     - Dimensionless precompound fraction in [0, 1] at incident-energy index i.
   * - ``kalbach-mann``
     - ``tables/angular-slope/{i}``
     - Data
     - Nonnegative dimensionless angular slope at incident-energy index i.
   * - ``energy-angle-table``
     - ``tables/cosine-distributions``
     - Packed probability tables, continuous-only
     - One dimensionless cosine PDF per point in ``tables/outgoing-energy/value``, in the same order.

Kalbach–Mann parameter Data takes outgoing energy in eV as its input; parameter tables may use independent grids and interpolation regions.
See :ref:`ace-law-44` and :ref:`ace-law-61` for the joint models and conditional sampling.

N-body phase-space distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``type = n-body`` stores a dimensionless energy-fraction PDF and final-state mass parameters.

.. list-table:: N-body fields
   :header-rows: 1

   * - Field
     - Type / shape
     - Meaning
   * - ``number-particles``
     - integer
     - Number of final-state bodies, at least three.
   * - ``total-mass-ratio``
     - float
     - Total final-state mass divided by the emitted particle mass, greater than one.
   * - ``value``
     - float [K]
     - Energy fractions in [0, 1].
   * - ``pdf``
     - float [K]
     - Dimensionless density of the energy fraction.
   * - ``interpolation``
     - Interpolation
     - Interpolation of the energy-fraction density.

The continuous-only :ref:`probability-table constraints <distribution-table>` apply.
The reaction requires ``reference-frame = COM`` and omits a separate angular-cosine distribution.
See :ref:`ace-law-66` for the phase-space model, physical energy scale, and angular sampling.
