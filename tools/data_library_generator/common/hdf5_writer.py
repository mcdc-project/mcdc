"""Write MC/DC HDF5 representations and coordinate output-file handling.

Source-format converters supply already interpreted values in the output
units. This module handles datasets, source-library provenance, interpolation
metadata, format identification, and staged publication
of output files. It does not interpret source records or convert physical units.

``FORMAT`` and ``VERSION`` identify the output specification independently of
the MC/DC software version and source-library release. Compatibility checks
require exact matches. Writers operate on caller-provided files and groups,
including root source records within the shared atomic library.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile

import h5py
import numpy as np

# ======================================================================================
# Format identifiers and source-library metadata schema
# ======================================================================================


FORMAT = "mcdc-data"
VERSION = "0.1.0"
INTERPOLATION_LAWS = {"histogram", "linear", "semilog-x", "semilog-y", "log"}
PROBABILITY_TOLERANCE = 1.0e-6
SOURCE_FIELDS = (
    "title",
    "version",
    "release-date",
    "reference",
    "comments",
    "format",
    "format-version",
    "processing-date",
)
SOURCE_UNITS = {"temperature": "K", "kT": "eV"}


class ProbabilityTableError(ValueError):
    """Invalid packed table, with its index and path available to converters.

    The writer identifies the failed table without interpreting its physical
    coordinate. A source converter can use this information to report the
    incident energy or other source coordinates associated with that table.
    """

    def __init__(self, index, reason, path):
        self.index = index
        self.reason = str(reason)
        self.path = path
        super().__init__(f"Packed PDF table {index}: {reason}")


# ======================================================================================
# File identity and source-library provenance
# ======================================================================================


def stamp(file, sources):
    """Write format identifiers and merge the supplied source-library records.

    Parameters
    ----------
    file : h5py.File
        Writable destination. When merging an existing nonempty file, the
        caller checks its format first; this function writes the identifiers
        but does not check the previous version.
    sources : dict of str to dict
        Library subgroup names mapped to scalar metadata. ``SOURCE_FIELDS``
        identifies the string fields; missing strings become empty. Optional
        numeric ``temperature`` and ``kT`` values are already in K and eV.
        The caller includes those fields only for temperature-specific files.

    Raises
    ------
    ValueError
        No sources are supplied; a library name is empty, ``.`` or ``..``,
        or contains a slash; a string field has a non-string value; or
        incoming nonempty metadata conflicts with an existing source record.

    Notes
    -----
    Root attributes ``format`` and ``format-version`` use the module constants.
    Each supplied library is stored under ``source-libraries/{library}`` as
    scalar datasets. Existing nonempty strings survive missing incoming
    information. Conflicting nonempty strings raise an error; numeric source
    temperatures are compared with NumPy's default ``isclose`` tolerances.
    Other source libraries and unrelated file contents remain intact.

    A UTC ISO 8601 ``conversion-time`` is generated for this write and applied
    only to the supplied libraries. Calling this function updates that time
    even if all other values already match, so skipped conversions must avoid
    calling it. Metadata conflicts are checked before writing; HDF5 I/O
    failures during publication can still leave partial changes in an open
    file. Existing named metadata datasets are recreated when written.
    """
    # Validate and merge source records before modifying the HDF5 file
    if not sources:
        raise ValueError("At least one source library is required")
    updates = {}
    for library, source in sources.items():
        if not library or library in (".", "..") or "/" in library:
            raise ValueError(f"Invalid source-library identifier: {library}")
        group = file.get(f"source-libraries/{library}")
        values = {}
        for name in SOURCE_FIELDS:
            value = source.get(name, "")
            if not isinstance(value, str):
                raise ValueError(f"{library}/{name}: source metadata must be a string")
            previous = (
                group[name].asstr()[()] if group is not None and name in group else ""
            )
            if previous and value and previous != value:
                raise ValueError(
                    f"{file.file.filename}: conflicting {name} for {library}"
                )
            values[name] = previous or value
        # Compare temperature-specific quantities in their declared output units
        for name in SOURCE_UNITS:
            if name in source:
                value = source[name]
                if (
                    group is not None
                    and name in group
                    and not np.isclose(group[name][()], value)
                ):
                    raise ValueError(
                        f"{file.file.filename}: conflicting {name} for {library}"
                    )
                values[name] = value
        updates[library] = values
    # Publish format identifiers and stamp the supplied library contributions
    file.attrs["format"] = FORMAT
    file.attrs["format-version"] = VERSION
    conversion_time = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    for library, values in updates.items():
        group = file.require_group(f"source-libraries/{library}")
        values["conversion-time"] = conversion_time
        for name, value in values.items():
            if name in group:
                del group[name]
            quantity(group, name, value, SOURCE_UNITS.get(name))


# ======================================================================================
# Datasets and reusable table representations
# ======================================================================================


def quantity(group, name, data, unit=None):
    """Create a scalar or array dataset and attach its supplied unit metadata.

    Parameters
    ----------
    group : h5py.Group or h5py.File
        Writable parent object.
    name : str
        Dataset name or relative path accepted by ``create_dataset``.
    data : object
        Scalar or array-like data accepted by h5py. Shape and datatype are
        inferred from this object.
    unit : str or sequence of str, optional
        Unit attribute to attach. A sequence is used for polynomial
        coefficients with different units. ``None`` omits the attribute,
        indicating dimensionless data.

    Returns
    -------
    h5py.Dataset
        Newly created dataset, available for further metadata assignment.

    Notes
    -----
    Values and unit strings are stored as supplied; no unit conversion or
    dimensional validation is performed. An existing object at ``name`` is
    not overwritten, and h5py creation errors propagate to the caller.
    """
    dataset = group.create_dataset(name, data=data)
    if unit is not None:
        dataset.attrs["unit"] = unit
    return dataset


def interpolation(group, count, laws=None, boundaries=None):
    """Create interpolation metadata for a table of known length.

    Parameters
    ----------
    group : h5py.Group
        Parent representation in which to create an ``interpolation`` group.
    count : int
        Number of local table points, at least two.
    laws : sequence of str, optional
        One name from ``INTERPOLATION_LAWS`` per region. The default is one
        ``linear`` region.
    boundaries : array_like of int, optional
        Exclusive upper-point index of each region. The default is a single
        endpoint, ``count``. The first region starts at point zero;
        subsequent regions start at the preceding boundary minus one.

    Raises
    ------
    ValueError
        There are fewer than two points; boundaries are empty, noninteger,
        or not a one-dimensional strictly increasing sequence starting at
        two or greater and ending at ``count``; counts of laws and boundaries
        differ; or a law name is unsupported.

    Notes
    -----
    For six points, boundaries ``[3, 6]`` describe intervals 0--1 and 1--2
    in the first region, followed by 2--3, 3--4, and 4--5 in the second.
    Indices are local to the described table, including a cross-section
    tail stored with a nonzero shared-grid offset.

    The child datasets are ``laws`` (fixed-width ``S16`` strings) and
    ``boundaries`` (integers). The function validates this metadata before
    creating the group, but does not inspect coordinates or values, evaluate
    interpolation, or check positivity domains for logarithmic laws.
    """
    laws, boundaries = _interpolation_arrays(count, laws, boundaries)
    rule = group.create_group("interpolation")
    quantity(rule, "laws", laws)
    quantity(rule, "boundaries", boundaries)


def _interpolation_arrays(count, laws=None, boundaries=None):
    """Validate local interpolation regions and return their storage arrays.

    Boundaries are exclusive upper-point indices within one table, ending
    at the table's point count. This validation is shared by standalone and
    packed tables so packing does not change interpolation-region semantics.
    """
    if count < 2:
        raise ValueError("Table data requires at least two points")
    laws = np.asarray(["linear"] if laws is None else laws, dtype="S16")
    boundaries = np.asarray([count] if boundaries is None else boundaries)
    if (
        laws.ndim != 1
        or boundaries.ndim != 1
        or len(boundaries) == 0
        or not np.issubdtype(boundaries.dtype, np.integer)
        or len(laws) != len(boundaries)
        or boundaries[0] < 2
        or boundaries[-1] != count
        or np.any(np.diff(np.r_[0, boundaries]) <= 0)
        or not set(laws.astype(str)) <= INTERPOLATION_LAWS
    ):
        raise ValueError("Invalid interpolation regions")
    return laws, boundaries.astype(np.int64)


def _log_ratio(lower, upper):
    """Compute logarithmic endpoint ratios, including nearly equal endpoints.

    All endpoints are positive. ``log1p`` avoids cancellation for nearby
    values; log differences handle ratios that exceed the floating-point range
    or round their relative difference to minus one.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = np.log1p((upper - lower) / lower)
    replace = ~np.isfinite(result)
    result[replace] = np.log(upper[replace]) - np.log(lower[replace])
    return result


def _pdf_arrays(value, pdf, laws=None, boundaries=None):
    """Validate one continuous density and integrate its interpolation regions.

    Return the coordinate, density, law, boundary arrays and their total mass.
    Coordinates must be finite and nondecreasing with positive support width;
    densities must be finite and nonnegative. Repeated coordinates carry no
    mass. Positive-width intervals must meet their logarithmic law's positivity
    requirements. Analytic segment integrals retain the supplied weights;
    normalization is checked separately because a mixed table may allocate
    some or all probability to its discrete component.
    """
    value, pdf = np.asarray(value, dtype=float), np.asarray(pdf, dtype=float)
    if value.ndim != 1 or value.shape != pdf.shape or len(value) < 2:
        raise ValueError(
            "PDF coordinates and densities must be matching vectors with at least two points"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("PDF coordinates must be finite")
    if not np.all(np.isfinite(pdf)) or np.any(pdf < 0):
        raise ValueError("PDF densities must be finite and nonnegative")
    if np.any(np.diff(value) < 0):
        raise ValueError("PDF coordinates must be nondecreasing")
    if value[-1] <= value[0]:
        raise ValueError("PDF coordinates must have positive support width")
    laws, boundaries = _interpolation_arrays(len(value), laws, boundaries)
    mass, start = 0.0, 0
    for law, boundary in zip(laws.astype(str), boundaries):
        stop = boundary - 1
        x0, x1 = value[start:stop], value[start + 1 : stop + 1]
        y0, y1 = pdf[start:stop], pdf[start + 1 : stop + 1]
        active = x1 > x0
        x0, x1, y0, y1 = x0[active], x1[active], y0[active], y1[active]
        width = x1 - x0
        if law in ("semilog-x", "log") and np.any(x0 <= 0):
            raise ValueError("Logarithmic PDF coordinates must be positive")
        if law in ("semilog-y", "log") and (np.any(y0 <= 0) or np.any(y1 <= 0)):
            raise ValueError("Logarithmic PDF densities must be positive")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if law == "histogram":
                areas = width * y0
            elif law == "linear":
                areas = width * (0.5 * y0 + 0.5 * y1)
            elif law == "semilog-x":
                ratio = _log_ratio(x0, x1)
                weight = np.empty_like(ratio)
                small = ratio < 1.0e-3
                r = ratio[small]
                weight[small] = 0.5 + r / 12.0 - r**3 / 720.0 + r**5 / 30240.0
                r = ratio[~small]
                weight[~small] = 1.0 / (-np.expm1(-r)) - 1.0 / r
                areas = width * ((1.0 - weight) * y0 + weight * y1)
            elif law == "semilog-y":
                ratio = _log_ratio(y0, y1)
                mean = y0.copy()
                varying = ratio != 0
                mean[varying] = (y1[varying] - y0[varying]) / ratio[varying]
                areas = width * mean
            else:
                ratio = _log_ratio(x0, x1)
                exponent = ratio + _log_ratio(y0, y1)
                areas = np.empty_like(ratio)
                small = np.abs(exponent) < 1.0e-3
                factor = np.ones(np.count_nonzero(small))
                nonzero = exponent[small] != 0
                factor[nonzero] = (
                    np.expm1(exponent[small][nonzero]) / exponent[small][nonzero]
                )
                areas[small] = x0[small] * y0[small] * ratio[small] * factor
                areas[~small] = (
                    (x1[~small] * y1[~small] - x0[~small] * y0[~small])
                    * ratio[~small]
                    / exponent[~small]
                )
            if not np.all(np.isfinite(areas)) or np.any(areas < 0):
                raise ValueError("PDF interval masses must be finite and nonnegative")
            mass += float(np.sum(areas))
        start = stop
    return value, pdf, laws, boundaries, mass


def _check_probability_mass(mass):
    """Require unit total mass within the absolute source-precision tolerance.

    Validation never rescales the source values. The same tolerance applies
    to a standalone PDF, a discrete PMF, their combined mixed probability,
    and every individual PDF in a packed collection.
    """
    if not np.isfinite(mass) or abs(mass - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(
            f"Total probability {mass:.16g} must equal one within "
            f"{PROBABILITY_TOLERANCE:g}"
        )


def pdf_table(group, values, pdf, law="linear", unit="eV"):
    """Write a one-dimensional probability-density table.

    Parameters
    ----------
    group : h5py.Group
        Empty destination Distribution group.
    values : array_like
        Sample-coordinate grid, already expressed in the output unit.
    pdf : array_like
        Probability-density values aligned with the coordinate grid, already
        expressed in the inverse output unit.
    law : str, optional
        Interpolation law for all intervals, defaulting to ``linear``.
    unit : str, optional
        Coordinate unit, defaulting to ``eV``; the density unit becomes
        ``1/{unit}``. Pass ``None`` for dimensionless cosine coordinates
        and densities, so both datasets omit the unit attribute.

    Raises
    ------
    ValueError
        Coordinates, densities, interpolation domains, or normalization are
        invalid under the common probability-table rules.

    Notes
    -----
    Writes ``type = table``, ``value``, ``pdf``, and a single-region
    ``interpolation`` group. The caller supplies correctly converted,
    normalized data. The shared validator checks the density's integral
    under its interpolation law against ``PROBABILITY_TOLERANCE`` before
    writing. Source values are preserved; no CDF is written.
    """
    distribution_table(group, values, pdf, laws=[law], unit=unit)


def distribution_table(
    group,
    value=None,
    pdf=None,
    *,
    discrete_value=None,
    pmf=None,
    laws=None,
    boundaries=None,
    unit="eV",
):
    """Write a continuous, discrete, or mixed Distribution table.

    Coordinate/density and discrete-coordinate/mass pairs are optional, but
    at least one complete pair is required. The caller supplies probabilities
    whose combined PMF sum and PDF integral equal one. This writer preserves
    those weights without independently normalizing either component.
    Interpolation describes only a continuous PDF. CDFs and CMFs are derived
    by the reader. Discrete coordinates carry the same unit as PDF coordinates;
    probability masses are dimensionless. Coordinates, probabilities,
    interpolation domains, and combined normalization are validated before
    writing. ``PROBABILITY_TOLERANCE`` is an absolute tolerance about one;
    invalid input raises ValueError without creating datasets or attributes.
    """
    continuous, discrete = value is not None, discrete_value is not None
    if (
        continuous != (pdf is not None)
        or discrete != (pmf is not None)
        or not (continuous or discrete)
    ):
        raise ValueError(
            "Supply a PDF coordinate/density pair, a discrete coordinate/mass pair, or both"
        )
    mass = 0.0
    if continuous:
        value, pdf, laws, boundaries, mass = _pdf_arrays(value, pdf, laws, boundaries)
    elif laws is not None or boundaries is not None:
        raise ValueError("Discrete probability masses do not use interpolation")
    if discrete:
        discrete_value, pmf = _pmf_arrays(discrete_value, pmf)
        with np.errstate(over="ignore"):
            mass += float(np.sum(pmf))
    _check_probability_mass(mass)
    group.attrs["type"] = "table"
    if continuous:
        quantity(group, "value", value, unit)
        quantity(group, "pdf", pdf, f"1/{unit}" if unit else None)
        interpolation(group, len(value), laws, boundaries)
    if discrete:
        quantity(group, "discrete-value", discrete_value, unit)
        quantity(group, "pmf", pmf)


# ======================================================================================
# Packed probability tables
# ======================================================================================


def pdf_tables(group, tables, unit="eV"):
    """Write a collection of PDF tables as concatenated arrays and offsets.

    Parameters
    ----------
    group : h5py.Group
        Empty destination collection. The containing distribution determines
        how these tables are indexed; this collection has no type attribute.
    tables : iterable of tuple
        Each item is ``(value, pdf, laws, boundaries)`` for one table. Values
        and densities are one-dimensional arrays of equal length. Laws and
        boundaries follow :func:`interpolation`, including its defaults and
        exclusive, table-local boundary convention. All tables describe the
        same physical coordinate and have at least two points.
    unit : str, optional
        Coordinate unit, with the inverse attached to ``pdf``. ``None`` omits
        both attributes for dimensionless coordinates such as angular cosine.

    Notes
    -----
    For N tables, ``offsets`` has N + 1 entries: it starts at zero, and its
    terminal entry is the combined point count. Table i occupies the half-open
    slice ``offsets[i]:offsets[i + 1]`` in ``value`` and ``pdf``. The child
    ``interpolation`` group similarly packs regions with its own N + 1 offsets.
    Region boundaries remain local to their individual table, not the combined
    array. No normalization, resampling, or CDF construction is performed.

    Each PDF is validated under the same coordinate, density, interpolation
    domain, and normalization rules as a standalone probability table.
    Invalid input raises ValueError before any datasets are created.
    The parent distribution and file publication remain the caller's
    responsibility.
    """
    _write_pdf_arrays(group, _packed_pdf_arrays(tables, path=group.name), unit)


def _pmf_arrays(value, pmf):
    """Validate finite discrete coordinates and nonnegative probability masses."""
    value, pmf = np.asarray(value, dtype=float), np.asarray(pmf, dtype=float)
    if (
        value.ndim != 1
        or value.shape != pmf.shape
        or not len(pmf)
        or not np.all(np.isfinite(value))
        or not np.all(np.isfinite(pmf))
        or np.any(pmf < 0)
    ):
        raise ValueError("Invalid discrete probability table")
    return value, pmf


def _packed_pdf_arrays(tables, discrete_value=None, pmf=None, *, path=None):
    """Validate and pack probability components, preserving their combined mass.

    Continuous-only callers supply ordinary PDF tuples. MultiTable may also
    supply per-grid discrete vectors and use None for an absent continuous
    part. Point and region offsets repeat for absent PDFs. Discrete offsets
    independently delimit corresponding point masses in grid order.
    """
    tables = list(tables)
    if not tables:
        raise ValueError("PDF table collection must not be empty")
    discrete = discrete_value is not None
    if discrete != (pmf is not None):
        raise ValueError("Discrete values require probability masses")
    ds = [None] * len(tables) if not discrete else list(discrete_value)
    ps = [None] * len(tables) if not discrete else list(pmf)
    if len(ds) != len(tables) or len(ps) != len(tables):
        raise ValueError("Grid and discrete table counts must match")
    values, densities, rules, endpoints, discrete_values, masses = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    offsets, rule_offsets, mass_offsets = [0], [0], [0]
    for index, (table, d, m) in enumerate(zip(tables, ds, ps)):
        try:
            mass = 0.0
            value, pdf = np.array([]), np.array([])
            laws, boundaries = np.array([], dtype="S16"), np.array([], dtype=np.int64)
            if table is not None:
                value, pdf, laws, boundaries, mass = _pdf_arrays(*table)
            if d is not None and np.asarray(d).size == 0:
                d = None
            if m is not None and np.asarray(m).size == 0:
                m = None
            if (d is None) != (m is None) or (table is None and d is None):
                raise ValueError(
                    "Each table requires a PDF or a discrete coordinate/mass pair"
                )
            if d is None:
                d, m = np.array([]), np.array([])
            else:
                d, m = _pmf_arrays(d, m)
                with np.errstate(over="ignore"):
                    mass += float(np.sum(m))
            _check_probability_mass(mass)
        except ValueError as error:
            raise ProbabilityTableError(index, error, path) from error
        values.append(value)
        densities.append(pdf)
        rules.append(laws)
        endpoints.append(boundaries)
        discrete_values.append(d)
        masses.append(m)
        offsets.append(offsets[-1] + len(value))
        rule_offsets.append(rule_offsets[-1] + len(laws))
        mass_offsets.append(mass_offsets[-1] + len(d))
    arrays = {}
    if offsets[-1]:
        arrays.update(
            {
                "offsets": np.asarray(offsets, dtype=np.int64),
                "value": np.concatenate(values),
                "pdf": np.concatenate(densities),
                "interpolation/offsets": np.asarray(rule_offsets, dtype=np.int64),
                "interpolation/laws": np.concatenate(rules),
                "interpolation/boundaries": np.concatenate(endpoints),
            }
        )
    if discrete:
        arrays.update(
            {
                "discrete-offsets": np.asarray(mass_offsets, dtype=np.int64),
                "discrete-value": np.concatenate(discrete_values),
                "pmf": np.concatenate(masses),
            }
        )
    return arrays


def _write_pdf_arrays(group, arrays, unit):
    """Write a validated packed collection with coordinate units."""
    units = {
        "value": unit,
        "discrete-value": unit,
        "pdf": f"1/{unit}" if unit else None,
    }
    for name, array in arrays.items():
        quantity(group, name, array, units.get(name))


def _incident_interpolation_arrays(grid, laws=None, boundaries=None):
    """Validate histogram/linear regions on a finite nondecreasing grid."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) < 0):
        raise ValueError("Distribution grid must be finite and nondecreasing")
    laws, boundaries = _interpolation_arrays(len(grid), laws, boundaries)
    if not set(laws.astype(str)) <= {"histogram", "linear"}:
        raise ValueError("Distribution grid interpolation must be histogram or linear")
    return laws, boundaries


def incident_interpolation(group, grid, laws=None, boundaries=None):
    """Write a conditional distribution's incident-energy Interpolation group.

    The same region boundaries and domain checks apply to MultiTable and joint
    energy-angle laws. The grid is stored by the containing representation.
    """
    laws, boundaries = _incident_interpolation_arrays(grid, laws, boundaries)
    interpolation(group, len(grid), laws, boundaries)


def multi_table(
    group,
    grid,
    tables=None,
    *,
    discrete_value=None,
    pmf=None,
    grid_unit="eV",
    unit="eV",
    laws=None,
    boundaries=None,
):
    """Write grid-indexed continuous, discrete, or mixed packed distributions.

    Optional PDF tuples follow pdf_tables; use None for a missing continuous
    component. discrete_value and pmf contain one optional vector per grid
    point. Each PDF integral plus its PMF sum must equal one within source
    precision. Neither component is independently normalized.

    Linear intervals require equal counts of corresponding discrete outcomes,
    with zero probability retaining a line that vanishes at an endpoint.
    Histogram intervals may have different discrete counts. All validation
    completes before any HDF5 object is written.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if tables is None:
        tables = [None] * len(grid) if grid.ndim == 1 else []
    tables = list(tables)
    if (
        grid.ndim != 1
        or len(grid) != len(tables)
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) < 0)
    ):
        raise ValueError(f"{group.name}: invalid multi-table grid or table count")
    laws, boundaries = _incident_interpolation_arrays(grid, laws, boundaries)
    arrays = _packed_pdf_arrays(tables, discrete_value, pmf, path=group.name)
    if "discrete-offsets" in arrays:
        counts = np.diff(arrays["discrete-offsets"])
        region = 0
        for i in range(len(grid) - 1):
            while i + 1 >= boundaries[region]:
                region += 1
            if (
                grid[i + 1] > grid[i]
                and laws[region] == b"linear"
                and counts[i] != counts[i + 1]
            ):
                raise ValueError(
                    "Linear multi-table intervals require corresponding discrete lines"
                )
    group.attrs["type"] = "multi-table"
    quantity(group, "grid", grid, grid_unit)
    interpolation(group, len(grid), laws, boundaries)
    _write_pdf_arrays(group.create_group("tables"), arrays, unit)


# ======================================================================================
# Format compatibility and prepared-file reuse
# ======================================================================================


def check_format(file):
    """Require exact supported format identifiers on an open HDF5 file.

    Parameters
    ----------
    file : h5py.File
        Open file whose root attributes are checked without modification.

    Raises
    ------
    ValueError
        Either ``format`` or ``format-version`` is missing or differs from
        ``FORMAT`` or ``VERSION`` respectively.

    Notes
    -----
    This is an identification check, not a schema or physical-data validator.
    It does not accept a version merely because its major number matches,
    and does not migrate existing files.
    """
    if (
        file.attrs.get("format") != FORMAT
        or file.attrs.get("format-version") != VERSION
    ):
        raise ValueError(f"{file.filename}: incompatible format or format version")


def prepared_file_exists(path, fields, sources):
    """Check required paths and source provenance without modifying an output.

    Missing files, paths, metadata, or valid UTC conversion times return False.
    Nonempty expected strings and supplied source temperatures must match.
    Required paths and source records are checked relative to the file root.
    Additional libraries and unrelated records are allowed. An existing
    file with incompatible format identifiers raises ValueError. This checks
    reuse readiness, not every reaction's numerical or structural validity.
    """
    path = Path(path)
    if not path.is_file():
        return False
    with h5py.File(path, "r") as file:
        check_format(file)
        complete = all(field in file for field in fields)
        for library, source in sources.items():
            group = file.get(f"source-libraries/{library}")
            if group is None:
                complete = False
                continue
            for name in (*SOURCE_FIELDS, "conversion-time"):
                if (
                    name not in group
                    or group[name].shape != ()
                    or h5py.check_string_dtype(group[name].dtype) is None
                ):
                    complete = False
                    continue
                value = group[name].asstr()[()]
                if name == "conversion-time":
                    try:
                        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        complete = (
                            complete
                            and value.endswith("Z")
                            and timestamp.tzinfo is not None
                        )
                    except ValueError:
                        complete = False
                elif source.get(name) and value != source[name]:
                    complete = False
            for name, unit in SOURCE_UNITS.items():
                if name in source:
                    complete = (
                        complete
                        and name in group
                        and group[name].shape == ()
                        and np.isclose(group[name][()], source[name])
                        and group[name].attrs.get("unit") == unit
                    )
    return complete


# ======================================================================================
# Staged file publication
# ======================================================================================


@contextmanager
def atomic_file(path, sources, *, preserve=False):
    """Stage one HDF5 output and replace its destination on successful exit.

    Parameters
    ----------
    path : str or pathlib.Path
        Final output path. The parent directory must already exist.
    sources : dict of str to dict
        Source-library records written by ``stamp`` before control is yielded.
    preserve : bool, optional
        Copy an existing compatible destination into the staged file before
        updating it. This retains other element records in atoms.h5.
        The default creates a fresh projectile output. A missing destination
        always starts a new file.

    Yields
    ------
    h5py.File
        Writable file at a temporary path in the destination directory,
        Already carries format identifiers and any supplied source records.

    Raises
    ------
    ValueError
        Source metadata is invalid or conversion inside the context raises
        a ValueError; exceptions from the caller's body propagate unchanged.
    OSError
        Creating, closing, or replacing the HDF5 file fails.

    Notes
    -----
    The destination is untouched while the caller writes the staged file.
    After a normal context exit, the HDF5 handle is closed and ``os.replace``
    publishes the file. Staging in the same directory keeps replacement on
    the same filesystem. If conversion fails before replacement, an existing
    destination remains intact. The staged path is removed in either case.

    Publication covers this file only: other files modified within the
    context are outside this transaction.
    """
    # Stage beside the destination so replacement stays on the same filesystem
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".h5", delete=False
    ) as temp:
        staged = Path(temp.name)
    try:
        # Preserve shared-file contents before allowing a projectile update.
        updating = preserve and path.is_file()
        if updating:
            shutil.copyfile(path, staged)
        with h5py.File(staged, "r+" if updating else "w") as file:
            if updating:
                check_format(file)
            stamp(file, sources)
            yield file
        # Publish only after writing and closing the staged file successfully
        os.replace(staged, path)
    finally:
        # Clean up the staged path on both success and failure
        staged.unlink(missing_ok=True)
