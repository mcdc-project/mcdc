import re
from typing import Annotated, Any, Union, get_args, get_origin

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from mcdc.constant import INF
from mcdc.print_ import print_error

# ======================================================================================
# Runtime annotation checking
# ======================================================================================

_DIMENSION_EXPRESSION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:([+-])\s*(\d+))?\s*$")
_ANNOTATED_RE = re.compile(r"^\s*(?:typing\.)?Annotated\[(.*)\]\s*$")


def parse_dimension_expression(expression: str) -> tuple[str, int]:
    """Parse a dimension name with an optional integer offset.

    Expressions such as ``"G"``, ``"G+1"``, and ``"N - 2"`` resolve to an
    object attribute name and a signed offset without evaluating arbitrary
    Python code.
    """
    match = _DIMENSION_EXPRESSION_RE.fullmatch(expression)
    if match is None:
        raise ValueError(f"Invalid dimension expression: {expression!r}")

    attribute, operator, magnitude = match.groups()
    offset = 0
    if magnitude is not None:
        offset = int(magnitude)
        if operator == "-":
            offset *= -1

    return attribute, offset


def normalize_ndarray_hint(hint):
    """Expand NumPy's named NDArray alias to the ndarray shape/dtype form."""
    if get_origin(hint) is NDArray:
        # Named aliases expose only the scalar argument, not shape and dtype.
        (scalar_type,) = get_args(hint)
        return np.ndarray[tuple[Any, ...], np.dtype[scalar_type]]
    return hint


def check_type(value, hint, cls, obj=None) -> bool:
    """Check a value against an MC/DC runtime type annotation.

    The checker supports structured and stringified annotations, shaped NumPy
    arrays, containers, unions, and unresolved class names. Annotation strings
    are parsed without evaluating arbitrary Python code.
    """
    # Check stringified annotations without resolving forward references
    if isinstance(hint, str):
        hint = hint.strip()

        if _is_ndarray_string(hint):
            if not isinstance(value, np.ndarray):
                return False
            return _dtype_matches(value, _ndarray_dtype_key_from_string(hint))

        annotated = _parse_annotated_string(hint)
        if annotated is not None:
            base, metadata = annotated
            if _is_ndarray_string(base) and metadata:
                if not isinstance(value, np.ndarray):
                    return False
                shape = _shape_from_string(metadata[0])
                expected_shape = _resolve_shape(shape, obj)
                if expected_shape is None or not _shape_matches(value, expected_shape):
                    return False
                return _dtype_matches(value, _ndarray_dtype_key_from_string(base))
            return _mro_name_matches(value, base)

        if "|" in hint:
            return any(
                check_type(value, item.strip(), cls)
                for item in _split_top_level(hint, separator="|")
            )

        if hint.startswith("list[") and hint.endswith("]"):
            item_type = _type_name_from_string(hint[5:-1])
            if not isinstance(value, list):
                return False
            if item_type == "str":
                return all(isinstance(item, str) for item in value)
            if item_type in ("float", "float32", "float64"):
                return all(isinstance(item, (float, int)) for item in value)
            return True

        return _mro_name_matches(value, hint)

    # Check structured typing annotations
    hint = normalize_ndarray_hint(hint)
    origin = get_origin(hint)

    if origin is Annotated:
        base, *metadata = get_args(hint)
        if (
            isinstance(value, np.ndarray)
            and metadata
            and isinstance(metadata[0], tuple)
        ):
            expected_shape = _resolve_shape(metadata[0], obj)
            if expected_shape is None:
                return False
            return _shape_matches(value, expected_shape) and _dtype_matches(
                value, _ndarray_dtype_key(base)
            )
        return check_type(value, base, cls)

    if origin is np.ndarray:
        return isinstance(value, np.ndarray)

    if origin is None:
        try:
            return isinstance(value, hint)
        except TypeError:
            return True

    if origin is list:
        (item_type,) = get_args(hint)
        return isinstance(value, list) and all(
            check_type(item, item_type, cls) for item in value
        )

    if origin is set:
        (item_type,) = get_args(hint)
        return isinstance(value, set) and all(
            check_type(item, item_type, cls) for item in value
        )

    if origin is dict:
        key_type, value_type = get_args(hint)
        return isinstance(value, dict) and all(
            check_type(key, key_type, cls) and check_type(item, value_type, cls)
            for key, item in value.items()
        )

    if origin is tuple:
        item_types = get_args(hint)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return isinstance(value, tuple) and all(
                check_type(item, item_types[0], cls) for item in value
            )
        return (
            isinstance(value, tuple)
            and len(value) == len(item_types)
            and all(
                check_type(item, item_type, cls)
                for item, item_type in zip(value, item_types)
            )
        )

    if origin is Union:
        return any(check_type(value, item_type, cls) for item_type in get_args(hint))

    try:
        return isinstance(value, origin)
    except TypeError:
        return True


# --------------------------------------------------------------------------------------
# Stringified annotation helpers
# --------------------------------------------------------------------------------------


def _parse_annotated_string(hint: str):
    """Split a stringified Annotated type into its base and metadata."""
    match = _ANNOTATED_RE.fullmatch(_strip_annotation_prefixes(hint))
    if match is None:
        return None

    parts = _split_top_level(match.group(1))
    if not parts:
        return None
    return parts[0].strip(), [part.strip() for part in parts[1:]]


def _shape_from_string(shape: str):
    """Parse shape metadata such as ``(3,)`` or ``("G+1",)``."""
    shape = shape.strip()
    if not (shape.startswith("(") and shape.endswith(")")):
        return None

    body = shape[1:-1].strip()
    if not body:
        return ()

    dimensions = []
    for item in _split_top_level(body):
        item = item.strip()
        if not item:
            continue
        if item == "None":
            dimensions.append(None)
            continue
        try:
            dimensions.append(int(item))
            continue
        except ValueError:
            pass

        if len(item) >= 2 and item[0] == item[-1] and item[0] in ("'", '"'):
            item = item[1:-1]
        try:
            parse_dimension_expression(item)
        except ValueError:
            return None
        dimensions.append(item)

    return tuple(dimensions)


def _split_top_level(
    value: str,
    separator: str = ",",
    brackets: str = "[]()",
) -> list[str]:
    """Split text at separators that are not nested inside brackets."""
    parts = []
    buffer = []
    depth = 0
    opening = set(brackets[::2])
    closing = set(brackets[1::2])

    for character in value:
        if character in opening:
            depth += 1
        elif character in closing:
            depth -= 1

        if character == separator and depth == 0:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(character)

    if buffer:
        parts.append("".join(buffer).strip())
    return parts


def _strip_annotation_prefixes(value: str) -> str:
    """Remove module prefixes accepted in stringified annotations."""
    return (
        value.replace("typing.", "")
        .replace("numpy.typing.", "")
        .replace("numpy.", "")
        .replace("np.", "")
    )


def _type_name_from_string(value: str) -> str:
    """Return the unqualified base name of a stringified type."""
    value = _strip_annotation_prefixes(value)
    value = value.split("[", 1)[0]
    return value.split(".")[-1].strip()


def _mro_name_matches(value, expected: str) -> bool:
    """Match an unresolved class name against the value's inheritance tree."""
    expected_name = _type_name_from_string(expected)
    return any(base.__name__ == expected_name for base in value.__class__.mro())


# --------------------------------------------------------------------------------------
# NumPy array annotation helpers
# --------------------------------------------------------------------------------------


def _is_ndarray_string(value: str) -> bool:
    value = _strip_annotation_prefixes(value)
    return value.startswith("NDArray[") or value.startswith("ndarray[")


def _ndarray_dtype_key_from_string(value: str) -> str | None:
    value = _strip_annotation_prefixes(value)
    if "[" not in value or "]" not in value:
        return None
    dtype = value[value.find("[") + 1 : value.rfind("]")].strip()
    return _strip_annotation_prefixes(dtype)


def _ndarray_dtype_key(hint) -> str | None:
    """Return the logical dtype from a structured NDArray annotation."""
    arguments = get_args(hint)
    if not arguments:
        return None

    dtype = arguments[-1]
    dtype_arguments = get_args(dtype)
    if dtype_arguments:
        dtype = dtype_arguments[-1]

    if dtype is float:
        return "float"
    if hasattr(dtype, "name"):
        return dtype.name
    try:
        return np.dtype(dtype).name
    except TypeError:
        return None


def _resolve_shape(shape, obj):
    """Resolve dimension expressions against the object being validated."""
    if shape is None:
        return None

    resolved = []
    for dimension in shape:
        if not isinstance(dimension, str):
            resolved.append(dimension)
            continue

        if obj is None:
            return None
        try:
            attribute, offset = parse_dimension_expression(dimension)
            resolved.append(getattr(obj, attribute) + offset)
        except (AttributeError, TypeError, ValueError):
            return None

    return tuple(resolved)


def _dtype_matches(array: np.ndarray, dtype_key: str | None) -> bool:
    if dtype_key is None:
        return True

    dtype_key = dtype_key.lower()
    if dtype_key == "float":
        return np.issubdtype(array.dtype, np.floating)
    if dtype_key == "int":
        return np.issubdtype(array.dtype, np.integer)
    try:
        return array.dtype == np.dtype(dtype_key)
    except TypeError:
        return True


def _shape_matches(array: np.ndarray, shape: tuple[int | None, ...]) -> bool:
    if array.ndim != len(shape):
        return False
    return all(
        expected is None or actual == expected
        for actual, expected in zip(array.shape, shape)
    )


# ======================================================================================
# Probability distribution conversions
# ======================================================================================


def cmf_from_pmf(
    pmf: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Normalize a probability mass function and construct its cumulative mass
    function.

    Parameters
    ----------
    pmf : ndarray of float64
        Probability masses.

    Returns
    -------
    pmf : ndarray of float64
        Normalized PMF.
    cmf : ndarray of float64
        Corresponding normalized CMF.
    """

    pmf = pmf.copy()

    cmf = np.zeros(len(pmf) + 1)

    # Build CMF incrementally.
    total = 0.0
    for i in range(len(pmf)):
        total += pmf[i]
        cmf[i + 1] = total

    norm = cmf[-1]

    if norm <= 0.0:
        print_error("PMF sum must be positive.")

    pmf /= norm
    cmf /= norm

    return pmf, cmf


def cdf_from_pdf(
    value: NDArray[float64],
    pdf: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Normalize a piecewise-linear PDF and construct its CDF.

    Parameters
    ----------
    value : ndarray of float64
        Sample values.
    pdf : ndarray of float64
        PDF values at the sample values.

    Returns
    -------
    pdf : ndarray of float64
        Normalized PDF.
    cdf : ndarray of float64
        Corresponding normalized CDF.
    """

    pdf = pdf.copy()
    cdf = np.zeros_like(pdf)

    # Trapezoidal integration of a piecewise-linear PDF.
    for i in range(len(pdf) - 1):
        cdf[i + 1] = cdf[i] + 0.5 * (pdf[i] + pdf[i + 1]) * (value[i + 1] - value[i])

    norm = cdf[-1]

    if norm <= 0.0:
        print_error("PDF integral must be positive.")

    pdf /= norm
    cdf /= norm

    return pdf, cdf


def pdf_from_cdf(
    value: NDArray[float64],
    cdf: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Normalize a piecewise-linear CDF and construct a histogram PDF.

    Parameters
    ----------
    value : ndarray of float64
        Sample values.
    cdf : ndarray of float64
        CDF values at the sample values.

    Returns
    -------
    pdf : ndarray of float64
        Histogram PDF.
    cdf : ndarray of float64
        Normalized CDF.
    """

    cdf = cdf.copy()

    if cdf[-1] <= 0.0:
        print_error("Final CDF value must be positive.")

    # Normalize CDF.
    cdf /= cdf[-1]

    pdf = np.zeros_like(cdf)

    # Piecewise-linear CDF => histogram PDF.
    for i in range(len(cdf) - 1):
        dv = value[i + 1] - value[i]

        if dv <= 0.0:
            print_error("Values must be strictly increasing.")

        pdf[i] = (cdf[i + 1] - cdf[i]) / dv

    # Histogram convention:
    # last PDF value is unused but kept for alignment.
    pdf[-1] = pdf[-2]

    return pdf, cdf


def multi_cdf_from_pdf(offset, value, pdf):
    """Normalize multiple flattened PDFs and construct their CDFs.

    Parameters
    ----------
    offset : array_like of int
        Start index of each table.
    value, pdf : ndarray
        Flattened sample values and probability densities.

    Returns
    -------
    pdf, cdf : ndarray
        Per-table normalized densities and cumulative distributions.
    """

    cdf = np.zeros_like(pdf)

    for i in range(len(offset)):
        start = offset[i]
        end = offset[i + 1] if i < len(offset) - 1 else len(pdf)

        # Build CDF incrementally with trapezoidal integration
        for idx in range(start, end - 1):
            cdf[idx + 1] = (
                cdf[idx]
                + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
            )

        # Normalize this segment so CDF ends at 1
        norm = cdf[end - 1]
        pdf[start:end] /= norm
        cdf[start:end] /= norm

    return pdf, cdf


# ======================================================================================
# Array helpers
# ======================================================================================


def is_sorted(a):
    """Return whether an array is monotonically nondecreasing."""
    return np.all(a[:-1] <= a[1:])


# ======================================================================================
# Natural isotopic abundance
# ======================================================================================

# Natural isotopic abundance data from
# https://www.nndc.bnl.gov/walletcards/search.html
ISOTOPIC_ABUNDANCE = {
    "H": {
        "H1": (99.972 + 99.999) / 2,
        "H2": (0.001 + 0.028) / 2,
    },
    "He": {
        "He3": 0.0002,
        "He4": 99.9998,
    },
    "Li": {
        "Li6": (1.9 + 7.8) / 2,
        "Li7": (92.2 + 98.1) / 2,
    },
    "Be": {
        "Be9": 100.0,
    },
    "B": {
        "B10": (18.9 + 20.4) / 2,
        "B11": (79.6 + 81.1) / 2,
    },
    "C": {
        "C12": (98.84 + 99.04) / 2,
        "C13": (0.96 + 1.16) / 2,
    },
    "N": {
        "N14": (99.578 + 99.663) / 2,
        "N15": (0.337 + 0.422) / 2,
    },
    "O": {
        "O16": (99.738 + 99.776) / 2,
        "O17": (0.0367 + 0.04) / 2,
        "O18": (0.187 + 0.222) / 2,
    },
    "F": {
        "F19": 100.0,
    },
    "Ne": {
        "Ne20": 90.48,
        "Ne21": 0.27,
        "Ne22": 9.25,
    },
    "Na": {
        "Na23": 100.0,
    },
    "Mg": {
        "Mg24": (78.88 + 79.05) / 2,
        "Mg25": (9.988 + 10.034) / 2,
        "Mg26": (10.96 + 11.09) / 2,
    },
    "Al": {
        "Al27": 100.0,
    },
    "Si": {
        "Si28": (92.191 + 92.318) / 2,
        "Si29": (4.645 + 4.699) / 2,
        "Si30": (3.037 + 3.11) / 2,
    },
    "P": {
        "P31": 100.0,
    },
    "S": {
        "S32": (94.41 + 95.29) / 2,
        "S33": (0.729 + 0.797) / 2,
        "S34": (3.96 + 4.77) / 2,
        "S36": (0.0129 + 0.0187) / 2,
    },
    "Cl": {
        "Cl35": (75.5 + 76.1) / 2,
        "Cl37": (23.9 + 24.5) / 2,
    },
    "Ar": {
        "Ar36": 0.3336,
        "Ar38": 0.0629,
        "Ar40": 99.6035,
    },
    "K": {
        "K39": 93.2581,
        "K40": 0.0117,
        "K41": 6.7302,
    },
    "Ca": {
        "Ca40": 96.941,
        "Ca42": 0.647,
        "Ca43": 0.135,
        "Ca44": 2.086,
        "Ca46": 0.004,
        "Ca48": 0.187,
    },
    "Sc": {
        "Sc45": 100.0,
    },
    "Ti": {
        "Ti46": 8.25,
        "Ti47": 7.44,
        "Ti48": 73.72,
        "Ti49": 5.41,
        "Ti50": 5.18,
    },
    "V": {
        "V50": 0.25,
        "V51": 99.75,
    },
    "Cr": {
        "Cr50": 4.345,
        "Cr52": 83.789,
        "Cr53": 9.501,
        "Cr54": 2.365,
    },
    "Mn": {
        "Mn55": 100.0,
    },
    "Fe": {
        "Fe54": 5.845,
        "Fe56": 91.754,
        "Fe57": 2.119,
        "Fe58": 0.282,
    },
    "Co": {
        "Co59": 100.0,
    },
    "Ni": {
        "Ni58": 68.0769,
        "Ni60": 26.2231,
        "Ni61": 1.1399,
        "Ni62": 3.6345,
        "Ni64": 0.9256,
    },
    "Cu": {
        "Cu63": 69.15,
        "Cu65": 30.85,
    },
    "Zn": {
        "Zn64": 49.17,
        "Zn66": 27.73,
        "Zn67": 4.04,
        "Zn68": 18.45,
        "Zn70": 0.61,
    },
    "Ga": {
        "Ga69": 60.108,
        "Ga71": 39.892,
    },
    "Ge": {
        "Ge70": 20.52,
        "Ge72": 27.45,
        "Ge73": 7.76,
        "Ge74": 36.52,
        "Ge76": 7.75,
    },
    "As": {
        "As75": 100.0,
    },
    "Se": {
        "Se74": 0.86,
        "Se76": 9.23,
        "Se77": 7.6,
        "Se78": 23.69,
        "Se80": 49.8,
        "Se82": 8.82,
    },
    "Br": {
        "Br79": (50.5 + 50.8) / 2,
        "Br81": (49.2 + 49.5) / 2,
    },
    "Kr": {
        "Kr78": 0.355,
        "Kr80": 2.286,
        "Kr82": 11.593,
        "Kr83": 11.5,
        "Kr84": 56.987,
        "Kr86": 17.279,
    },
    "Rb": {
        "Rb85": 72.17,
        "Rb87": 27.83,
    },
    "Sr": {
        "Sr84": 0.56,
        "Sr86": 9.86,
        "Sr87": 7.0,
        "Sr88": 82.58,
    },
    "Y": {
        "Y89": 100.0,
    },
    "Zr": {
        "Zr90": 51.45,
        "Zr91": 11.22,
        "Zr92": 17.15,
        "Zr94": 17.38,
        "Zr96": 2.8,
    },
    "Nb": {
        "Nb93": 100.0,
    },
    "Mo": {
        "Mo92": 14.649,
        "Mo94": 9.187,
        "Mo95": 15.873,
        "Mo96": 16.673,
        "Mo97": 9.582,
        "Mo98": 24.292,
        "Mo100": 9.744,
    },
    "Ru": {
        "Ru96": 5.54,
        "Ru98": 1.87,
        "Ru99": 12.76,
        "Ru100": 12.6,
        "Ru101": 17.06,
        "Ru102": 31.55,
        "Ru104": 18.62,
    },
    "Rh": {
        "Rh103": 100.0,
    },
    "Pd": {
        "Pd102": 1.02,
        "Pd104": 11.14,
        "Pd105": 22.33,
        "Pd106": 27.33,
        "Pd108": 26.46,
        "Pd110": 11.72,
    },
    "Ag": {
        "Ag107": 51.839,
        "Ag109": 48.161,
    },
    "Cd": {
        "Cd106": 1.245,
        "Cd108": 0.888,
        "Cd110": 12.47,
        "Cd111": 12.795,
        "Cd112": 24.109,
        "Cd113": 12.227,
        "Cd114": 28.754,
        "Cd116": 7.512,
    },
    "In": {
        "In113": 4.281,
        "In115": 95.719,
    },
    "Sn": {
        "Sn112": 0.97,
        "Sn114": 0.66,
        "Sn115": 0.34,
        "Sn116": 14.54,
        "Sn117": 7.68,
        "Sn118": 24.22,
        "Sn119": 8.59,
        "Sn120": 32.58,
        "Sn122": 4.63,
        "Sn124": 5.79,
    },
    "Sb": {
        "Sb121": 57.21,
        "Sb123": 42.79,
    },
    "Te": {
        "Te120": 0.09,
        "Te122": 2.55,
        "Te123": 0.89,
        "Te124": 4.74,
        "Te125": 7.07,
        "Te126": 8.84,
        "Te128": 31.74,
        "Te130": 34.08,
    },
    "I": {
        "I127": 100.0,
    },
    "Xe": {
        "Xe124": 0.095,
        "Xe126": 0.089,
        "Xe128": 1.91,
        "Xe129": 26.4,
        "Xe130": 4.071,
        "Xe131": 21.232,
        "Xe132": 26.909,
        "Xe134": 10.436,
        "Xe136": 8.857,
    },
    "Cs": {
        "Cs133": 100.0,
    },
    "Ba": {
        "Ba130": 0.11,
        "Ba132": 0.1,
        "Ba134": 2.42,
        "Ba135": 6.59,
        "Ba136": 7.85,
        "Ba137": 11.23,
        "Ba138": 71.7,
    },
    "La": {
        "La138": 0.08881,
        "La139": 99.91119,
    },
    "Ce": {
        "Ce136": 0.186,
        "Ce138": 0.251,
        "Ce140": 88.449,
        "Ce142": 11.114,
    },
    "Pr": {
        "Pr141": 100.0,
    },
    "Nd": {
        "Nd142": 27.153,
        "Nd143": 12.173,
        "Nd144": 23.798,
        "Nd145": 8.293,
        "Nd146": 17.189,
        "Nd148": 5.756,
        "Nd150": 5.638,
    },
    "Sm": {
        "Sm144": 3.08,
        "Sm147": 15.0,
        "Sm149": 13.82,
        "Sm150": 7.37,
        "Sm152": 26.74,
        "Sm154": 22.74,
    },
    "Eu": {
        "Eu151": 47.81,
        "Eu153": 52.19,
    },
    "Gd": {
        "Gd152": 0.2,
        "Gd154": 2.18,
        "Gd155": 14.8,
        "Gd156": 20.47,
        "Gd157": 15.65,
        "Gd158": 24.84,
        "Gd160": 21.86,
    },
    "Tb": {
        "Tb159": 100.0,
    },
    "Dy": {
        "Dy156": 0.056,
        "Dy158": 0.095,
        "Dy160": 2.329,
        "Dy161": 18.889,
        "Dy162": 25.475,
        "Dy163": 24.896,
        "Dy164": 28.26,
    },
    "Ho": {
        "Ho165": 100.0,
    },
    "Er": {
        "Er162": 0.139,
        "Er164": 1.601,
        "Er166": 33.503,
        "Er167": 22.869,
        "Er168": 26.978,
        "Er170": 14.91,
    },
    "Tm": {
        "Tm169": 100.0,
    },
    "Yb": {
        "Yb168": 0.123,
        "Yb170": 2.982,
        "Yb171": 14.086,
        "Yb172": 21.686,
        "Yb173": 16.103,
        "Yb174": 32.025,
        "Yb176": 12.995,
    },
    "Lu": {
        "Lu175": 97.401,
        "Lu176": 2.599,
    },
    "Hf": {
        "Hf174": 0.16,
        "Hf176": 5.26,
        "Hf177": 18.6,
        "Hf178": 27.28,
        "Hf179": 13.62,
        "Hf180": 35.08,
    },
    "Ta": {
        "Ta180": 0.01201,
        "Ta181": 99.98799,
    },
    "W": {
        "W180": 0.12,
        "W182": 26.5,
        "W183": 14.31,
        "W184": 30.64,
        "W186": 28.43,
    },
    "Re": {
        "Re185": 37.4,
        "Re187": 62.6,
    },
    "Os": {
        "Os184": 0.02,
        "Os186": 1.59,
        "Os187": 1.97,
        "Os188": 13.24,
        "Os189": 16.15,
        "Os190": 26.26,
        "Os192": 40.78,
    },
    "Ir": {
        "Ir191": 37.3,
        "Ir193": 62.7,
    },
    "Pt": {
        "Pt190": 0.012,
        "Pt192": 0.782,
        "Pt194": 32.864,
        "Pt195": 33.77,
        "Pt196": 25.21,
        "Pt198": 7.356,
    },
    "Au": {
        "Au197": 100.0,
    },
    "Hg": {
        "Hg196": 0.15,
        "Hg198": 10.04,
        "Hg199": 16.94,
        "Hg200": 23.14,
        "Hg201": 13.17,
        "Hg202": 29.74,
        "Hg204": 6.82,
    },
    "Tl": {
        "Tl203": (29.44 + 29.59) / 2,
        "Tl205": (70.41 + 70.56) / 2,
    },
    "Pb": {
        "Pb204": 1.4,
        "Pb206": 24.1,
        "Pb207": 22.1,
        "Pb208": 52.4,
    },
    "Bi": {
        "Bi209": 100.0,
    },
    "Th": {
        "Th230": 0.02,
        "Th232": 99.98,
    },
    "Pa": {
        "Pa231": 100.0,
    },
    "U": {
        "U234": 0.0054,
        "U235": 0.7204,
        "U238": 99.2742,
    },
}


# ======================================================================================
# Object helpers
# ======================================================================================


def move_object(object_, velocities, durations):
    """Populate the shared piecewise-constant motion representation.

    A zero-velocity segment of infinite duration is appended after the supplied
    segments. The input lists are therefore mutated; array inputs are converted
    to lists first.
    """

    object_.moving = True
    object_.N_move = len(durations) + 1
    object_.N_move_grid = len(durations) + 2

    if isinstance(velocities, np.ndarray):
        velocities = velocities.tolist()
        durations = durations.tolist()

    # Add the statics for the rest of the simulation
    move_velocities = velocities
    move_velocities.append([0.0, 0.0, 0.0])
    object_.move_velocities = np.array(move_velocities)
    #
    move_durations = durations
    move_durations.append(INF)
    object_.move_durations = np.array(move_durations)

    # Set time grid and translations
    object_.move_time_grid = np.zeros(object_.N_move_grid)
    object_.move_translations = np.zeros((object_.N_move_grid, 3))
    for n in range(object_.N_move):
        t_start = object_.move_time_grid[n]
        object_.move_time_grid[n + 1] = t_start + object_.move_durations[n]

        trans_start = object_.move_translations[n]
        object_.move_translations[n + 1] = (
            trans_start + object_.move_velocities[n] * object_.move_durations[n]
        )


def subtype_size(main_list, subtype: str):
    """Count objects whose framework label matches ``subtype``."""
    return len([x for x in main_list if x.label == subtype])
