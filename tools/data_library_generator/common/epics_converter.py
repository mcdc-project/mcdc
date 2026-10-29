"""Interpret EPICS ENDF evaluations and write MC/DC transport representations.

The EPDL and EADL inputs contain elemental photon and atomic-relaxation data.
ENDF energies are already in eV and cross sections in barns. This module parses
source records, preserves interpolation and discontinuities, extracts library
provenance, and writes interpreted quantities through ``hdf5_writer``.
Shared EADL preparation updates element records in atoms.h5. Projectile scripts
select inputs and coordinate publication of projectile files.
"""

from datetime import datetime
from pathlib import Path
import re

import h5py
import numpy as np

from .hdf5_writer import (
    atomic_file,
    check_format,
    interpolation,
    prepared_file_exists,
    quantity,
)

# ======================================================================================
# EPICS identifiers and ENDF interpolation laws
# ======================================================================================

Z_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
}

SYMBOL_TO_Z = {symbol: z for z, symbol in Z_TO_SYMBOL.items()}
ENDF_INTERPOLATION = {
    1: "histogram",
    2: "linear",
    3: "semilog-x",
    4: "semilog-y",
    5: "log",
}
PHOTON_REACTIONS = {
    502: ("coherent-scattering", "coherent-scattering"),
    504: ("incoherent-scattering", "incoherent-scattering"),
    522: ("photoelectric", "photoelectric"),
    515: ("pair-production-electron-field", "pair-production"),
    517: ("pair-production-nuclear-field", "pair-production"),
}
SUBSHELL_NAMES = ("K",) + tuple(
    f"{letter}{index}"
    for n, letter in enumerate("LMNOPQ", start=2)
    for index in range(1, 2 * n)
)
_ENDF_EXPONENT = re.compile(r"^([+-]?[0-9]*\.?[0-9]*)([+-][0-9]+)$")

# ======================================================================================
# ENDF-6 records and source-library provenance
# ======================================================================================


def endf_float(field):
    """Decode an ENDF numeric field, accepting implicit exponents and E or D.

    Blank fields return zero, as padding in numeric records. Text records are
    handled separately and must not be passed to this function.
    """
    field = field.strip().replace("D", "E").replace("d", "e")
    if not field:
        return 0.0
    match = _ENDF_EXPONENT.fullmatch(field)
    if match:
        field = match.group(1) + "e" + match.group(2)
    return float(field)


def read_sections(path):
    """Collect the sections of a single-element ENDF evaluation by (MF, MT).

    Both 75-column records and records with optional sequence numbers are
    accepted. Tape headers and section/material terminators are excluded.
    Every returned section begins with its HEAD record. Files containing more
    than one material are rejected rather than combining unrelated sections.
    """
    sections, materials = {}, set()
    with Path(path).open() as source:
        for number, line in enumerate(source, start=1):
            if len(line.rstrip("\r\n")) < 75:
                if line.strip():
                    raise ValueError(f"{path}:{number}: incomplete ENDF record")
                continue
            try:
                mat, mf, mt = int(line[66:70]), int(line[70:72]), int(line[72:75])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{number}: invalid ENDF identifiers"
                ) from error
            if mat > 0 and mf > 0 and mt > 0:
                materials.add(mat)
                sections.setdefault((mf, mt), []).append(line.rstrip("\r\n"))
    if len(materials) != 1:
        raise ValueError(f"{path}: expected one ENDF material")
    return sections


def read_atomic_number(sections):
    """Read the elemental ZA identity and verify it across relevant sections.

    EPICS targets are elements, with ZA = 1000 Z and no isotope mass number.
    The returned Z selects the element symbol and companion input filename.
    """
    identities = {
        endf_float(block[0][:11])
        for (mf, _), block in sections.items()
        if mf in (1, 23, 27, 28)
    }
    if len(identities) != 1:
        raise ValueError("ENDF sections have inconsistent target identities")
    za = identities.pop()
    z = int(za / 1000)
    if za != 1000 * z or z not in Z_TO_SYMBOL:
        raise ValueError(f"Unsupported elemental ENDF identity: {za}")
    return z


def read_source_libraries(path, sections, library):
    """Extract a source-library record from the tape header and MF 1/MT 451.

    ``library`` identifies EPDL or EADL. The release year in the tape title
    supplies the version and subgroup suffix, e.g. ``EPDL2025``. The original
    distribution date and reference come from the descriptive TEXT records;
    unavailable metadata remains empty. The evaluation date is retained in
    comments and is not relabeled as a processing date.
    """
    if library not in ("EPDL", "EADL"):
        raise ValueError(f"Unsupported EPICS library: {library}")
    with Path(path).open() as source:
        title = source.readline()[:66].strip()
    if f"({library})" not in title:
        raise ValueError(f"{path}: tape title does not identify {library}")
    match = re.search(r"\b(\d{4})\s+Evaluated\b", title)
    version = match.group(1) if match else ""
    if (1, 451) not in sections or len(sections[1, 451]) < 4:
        raise ValueError(f"{path}: missing ENDF descriptive section MF 1/MT 451")
    header = sections[1, 451]
    count = int(endf_float(header[3][44:55]))
    if len(header) < 4 + count:
        raise ValueError(f"{path}: truncated ENDF descriptive section")
    comments = "\n".join(line[:66].rstrip() for line in header[4 : 4 + count])
    distribution = re.search(r"\bDist-([A-Za-z]{3}\d{2,4})\b", comments)
    release = ""
    if distribution:
        token = distribution.group(1)
        date = datetime.strptime(token, "%b%y" if len(token) == 5 else "%b%Y")
        release = date.strftime("%Y-%m")
    reference = re.search(r"\bNDS-IAEA-\d+\b", comments)
    return {
        library
        + version: {
            "title": title,
            "version": version,
            "release-date": release,
            "reference": reference.group(0) if reference else "",
            "comments": comments,
            "format": "ENDF",
            "format-version": str(int(endf_float(header[1][55:66]))),
            "processing-date": "",
        }
    }


def _number_fields(records, count):
    """Decode exactly ``count`` numeric fields, excluding final-record padding."""
    values = [
        endf_float(line[start : start + 11])
        for line in records
        for start in range(0, 66, 11)
    ]
    if len(values) < count:
        raise ValueError("Truncated ENDF numeric record")
    return np.asarray(values[:count], dtype=np.float64)


def read_tab1(block):
    """Read a HEAD/TAB1 section as grid, values, NBT boundaries, and INT laws.

    NBT remains the ENDF one-based upper-point count until writing. Repeated
    coordinates retain the values on both sides of a discontinuity. Explicit
    interpolation regions must cover all points; NR=0 defaults to linear.
    """
    if len(block) < 2:
        raise ValueError("Truncated ENDF TAB1 section")
    regions = int(endf_float(block[1][44:55]))
    count = int(endf_float(block[1][55:66]))
    if regions < 0 or count < 2:
        raise ValueError("Invalid ENDF TAB1 point or region count")
    region_records = (2 * regions + 5) // 6
    data_records = (2 * count + 5) // 6
    if len(block) != 2 + region_records + data_records:
        raise ValueError("ENDF TAB1 section length disagrees with its point count")
    pairs = _number_fields(block[2 : 2 + region_records], 2 * regions)
    if np.any(pairs != np.floor(pairs)):
        raise ValueError("ENDF interpolation metadata must contain integers")
    boundaries, laws = pairs[::2].astype(np.int64), pairs[1::2].astype(np.int64)
    if not regions:
        boundaries, laws = np.array([count]), np.array([2])
    values = _number_fields(block[2 + region_records :], 2 * count)
    grid, ordinates = values[::2], values[1::2]
    if (
        not np.all(np.isfinite(values))
        or np.any(np.diff(grid) < 0)
        or grid[-1] <= grid[0]
        or boundaries[0] < 2
        or boundaries[-1] != count
        or np.any(np.diff(boundaries) <= 0)
        or not set(laws) <= ENDF_INTERPOLATION.keys()
    ):
        raise ValueError("Invalid ENDF table or interpolation regions")
    codes = _interval_laws(np.arange(count - 1), boundaries, laws)
    intervals = np.diff(grid) > 0
    if np.any(intervals & np.isin(codes, (3, 5)) & (grid[:-1] <= 0)):
        raise ValueError("Logarithmic ENDF coordinates must be positive")
    if np.any(
        intervals
        & np.isin(codes, (4, 5))
        & ((ordinates[:-1] <= 0) | (ordinates[1:] <= 0))
    ):
        raise ValueError("Logarithmic ENDF ordinates must be positive")
    return grid, ordinates, boundaries, laws


# ======================================================================================
# Table data, interpolation, and shared-grid cross sections
# ======================================================================================


def write_table(group, table, grid_unit=None, value_unit=None):
    """Write an ENDF TAB1 table as Data without resampling its own grid.

    Coordinates and values already use the requested output units. ENDF NBT
    point counts are retained as exclusive interpolation boundaries.
    Repeated coordinates are retained, including photoelectric opening edges.
    """
    grid, values, boundaries, laws = table
    group.attrs["type"] = "table"
    quantity(group, "grid", grid, grid_unit)
    quantity(group, "value", values, value_unit)
    interpolation(
        group, len(grid), [ENDF_INTERPOLATION[law] for law in laws], boundaries
    )


def _interval_laws(indices, boundaries, laws):
    """Find the ENDF interpolation law governing each local interval index."""
    regions = np.searchsorted(boundaries - 1, indices + 1, side="left")
    return laws[np.minimum(regions, len(laws) - 1)]


def project_onto_grid(grid, table):
    """Project a cross section onto a shared grid and preserve both edge limits.

    The first copy of a repeated target energy receives the source's left
    limit; the second receives its right limit. Values below the source grid
    are zero and values above it use the last ordinate. The returned interval
    laws allow evaluation between shared knots without replacing nonlinear
    interpolation by linear interpolation.
    """
    source_grid, values, boundaries, laws = table
    if np.any(values < 0):
        raise ValueError("Photon cross sections must be nonnegative")
    first = np.r_[grid[:-1] == grid[1:], False]
    indices = np.searchsorted(source_grid, grid, side="right") - 1
    indices = np.where(
        first, np.searchsorted(source_grid, grid, side="left") - 1, indices
    )
    local = np.clip(indices, 0, len(source_grid) - 2)
    x0, x1 = source_grid[local], source_grid[local + 1]
    y0, y1 = values[local], values[local + 1]
    codes = _interval_laws(local, boundaries, laws)
    width = x1 - x0
    fraction = np.divide(grid - x0, width, out=np.zeros_like(grid), where=width > 0)
    result = y0 + fraction * (y1 - y0)
    result[codes == 1] = y0[codes == 1]
    active = (indices >= 0) & (indices < len(source_grid) - 1) & (width > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logx = active & np.isin(codes, (3, 5))
        if np.any(logx & ((x0 <= 0) | (x1 <= 0) | (grid <= 0))):
            raise ValueError("Logarithmic ENDF coordinates must be positive")
        fraction[logx] = np.log(grid[logx] / x0[logx]) / np.log(x1[logx] / x0[logx])
        result[codes == 3] = (y0 + fraction * (y1 - y0))[codes == 3]
        logy = active & np.isin(codes, (4, 5))
        if np.any(logy & ((y0 <= 0) | (y1 <= 0))):
            raise ValueError("Logarithmic ENDF ordinates must be positive")
        result[logy] = y0[logy] * np.exp(fraction[logy] * np.log(y1[logy] / y0[logy]))
    result[indices < 0] = 0.0
    result[indices >= len(source_grid) - 1] = values[-1]
    # The interval after a shared point starts on the right side of its edge.
    following = np.clip(
        np.searchsorted(source_grid, grid[:-1], side="right") - 1,
        0,
        len(source_grid) - 2,
    )
    projected_laws = _interval_laws(following, boundaries, laws)
    outside = (grid[:-1] < source_grid[0]) | (grid[:-1] >= source_grid[-1])
    projected_laws[outside | (np.diff(grid) == 0)] = 2
    return result, projected_laws


def write_cross_section(group, grid, table):
    """Write a photon CrossSection and compress adjacent equal-law regions.

    The shared grid includes every source knot. Leading zero-valued points
    are retained so threshold discontinuities keep their left and right limits.
    """
    values, laws = project_onto_grid(grid, table)
    quantity(group, "offset", 0)
    quantity(group, "value", values, "barn")
    ends = np.r_[np.flatnonzero(laws[1:] != laws[:-1]) + 1, len(laws)]
    interpolation(
        group, len(grid), [ENDF_INTERPOLATION[law] for law in laws[ends - 1]], ends + 1
    )


def photon_energy_grid(sections):
    """Union photon reaction knots, preserving the maximum edge multiplicity.

    MT 501 normally already contains this union. Including individual-channel
    knots also preserves source interpolation if a supplied total uses a
    different grid. Subshell tables keep their own grids and are not unionized.
    """
    multiplicities = {}
    for mt in (501, *PHOTON_REACTIONS):
        if (23, mt) not in sections:
            raise ValueError(f"Missing photon cross section MF 23/MT {mt}")
        grid, _, _, _ = read_tab1(sections[23, mt])
        coordinates, counts = np.unique(grid, return_counts=True)
        for coordinate, count in zip(coordinates, counts):
            multiplicities[coordinate] = max(multiplicities.get(coordinate, 0), count)
    return np.asarray(
        [x for x, count in sorted(multiplicities.items()) for _ in range(count)]
    )


# ======================================================================================
# Atomic relaxation
# ======================================================================================


def subshell_name(designator):
    """Return the spectroscopic shell name for a positive ENDF designator."""
    if designator < 1 or designator > len(SUBSHELL_NAMES):
        raise ValueError(f"Unsupported subshell designator: {designator}")
    return SUBSHELL_NAMES[designator - 1]


def read_relaxation(block):
    """Read EADL MF 28/MT 533 into shell identities and transition arrays.

    Each LIST record contains binding energy and electron population followed
    by six values per transition. Only the four defined transition quantities
    are retained. Empty transition arrays describe terminal shells. Nonempty
    branching probabilities must sum to one within source rounding precision;
    radiative probabilities alone sum to the fluorescence yield.
    """
    if not block:
        raise ValueError("Missing EADL relaxation HEAD record")
    count = int(endf_float(block[0][44:55]))
    shells, position = {}, 1
    for _ in range(count):
        if position >= len(block):
            raise ValueError("Truncated EADL subshell record")
        header = block[position]
        designator = int(endf_float(header[:11]))
        size = int(endf_float(header[44:55]))
        transitions = int(endf_float(header[55:66]))
        if size != 6 * (transitions + 1) or transitions < 0:
            raise ValueError("Invalid EADL LIST size or transition count")
        records = (size + 5) // 6
        data = _number_fields(block[position + 1 : position + 1 + records], size)
        position += 1 + records
        subshell_name(designator)
        if (
            designator in shells
            or not np.all(np.isfinite(data))
            or np.any(data[:2] < 0)
        ):
            raise ValueError("Invalid EADL subshell identity or binding data")
        rows = data[6:].reshape(transitions, 6)
        if np.any(rows[:, :4] < 0) or np.any(rows[:, :2] != np.floor(rows[:, :2])):
            raise ValueError("Invalid EADL transition values or designators")
        if transitions and not np.isclose(
            np.sum(rows[:, 3]), 1.0, rtol=1e-5, atol=1e-8
        ):
            raise ValueError("EADL transition probabilities must sum to one")
        shells[designator] = {
            "binding-energy": data[0],
            "electron-count": data[1],
            "origin-designator": rows[:, 0].astype(np.int64),
            "secondary-designator": rows[:, 1].astype(np.int64),
            "energy": rows[:, 2],
            "probability": rows[:, 3],
        }
    if position != len(block):
        raise ValueError("EADL section length disagrees with its subshell count")
    for shell in shells.values():
        if (
            not set(shell["origin-designator"]) <= shells.keys()
            or not (set(shell["secondary-designator"]) - {0}) <= shells.keys()
        ):
            raise ValueError("EADL transition references an unknown subshell")
    return shells


def write_relaxation(group, shells):
    """Write source shell identities and aligned transitions at element scope."""
    parent = group.create_group("subshells")
    for designator, shell in sorted(shells.items()):
        child = parent.create_group(subshell_name(designator))
        quantity(child, "designator", designator)
        quantity(child, "binding-energy", shell["binding-energy"], "eV")
        quantity(child, "electron-count", shell["electron-count"])
        transitions = child.create_group("transitions")
        for name in (
            "energy",
            "probability",
            "origin-designator",
            "secondary-designator",
        ):
            quantity(transitions, name, shell[name], "eV" if name == "energy" else None)


# ======================================================================================
# Atomic library preparation
# ======================================================================================


def prepare_atomic_data(
    path, output_dir, *, atomic_number=None, rewrite=False, verbose=False
):
    """Prepare one EADL element in atoms.h5 and return its parsed subshells.

    The optional atomic_number checks a projectile's target before any output
    is changed. Compatible records are reused without updating their provenance.
    Rewrites replace only the requested element; other element records survive.
    Root provenance describes library releases; target-specific metadata is
    omitted because the file can contain evaluations for multiple elements.
    Invalid inputs or publication failures leave the previous atomic file intact.
    The output directory must already exist, and paths are pathlib Paths.
    """
    sections = read_sections(path)
    z = read_atomic_number(sections)
    if atomic_number is not None and z != atomic_number:
        raise ValueError(f"{path}: EADL target differs from projectile target")
    if (28, 533) not in sections:
        raise ValueError(f"{path}: missing atomic relaxation MF 28/MT 533")
    shells = read_relaxation(sections[28, 533])
    sources = read_source_libraries(path, sections, "EADL")
    shared_fields = ("version", "release-date", "reference", "format", "format-version")
    sources = {
        library: {field: source.get(field, "") for field in shared_fields}
        for library, source in sources.items()
    }
    symbol = Z_TO_SYMBOL[z]
    destination = output_dir / "atoms.h5"
    if destination.is_file():
        with h5py.File(destination, "r") as file:
            check_format(file)
            identity = file.get(f"{symbol}/atomic-number")
            if identity is not None and int(identity[()]) != z:
                raise ValueError(f"{destination}/{symbol}: inconsistent atomic-number")
    required = [f"{symbol}/atomic-number", f"{symbol}/atomic-relaxation/subshells"]
    if not rewrite and prepared_file_exists(destination, required, sources):
        if verbose:
            print(f"Skip atoms.h5/{symbol}: atomic record already exists")
        return shells
    if verbose:
        print(f"Create atoms.h5/{symbol} from {path.name}")
    with atomic_file(destination, sources, preserve=True) as file:
        if symbol in file:
            if int(file[f"{symbol}/atomic-number"][()]) != z:
                raise ValueError(f"{destination}/{symbol}: inconsistent atomic-number")
            del file[symbol]
        record = file.create_group(symbol)
        quantity(record, "atomic-number", z)
        write_relaxation(record.create_group("atomic-relaxation"), shells)
    return shells


# ======================================================================================
# Photon reactions
# ======================================================================================


def write_photon(file, sections, shells=None, verbose=False):
    """Write EPDL photon reactions with shared-grid xs and native-grid Data.

    Aggregate MT 501 contributes shared-grid knots; neither it nor MT 516 is
    written as an additional transport reaction. Atomic relaxation is stored
    at element scope. No secondary-particle production is inferred from totals.
    """
    grid = photon_energy_grid(sections)
    photon = file
    quantity(photon, "xs-energy-grid", grid, "eV")
    reactions = photon.create_group("reactions", track_order=True)
    for mt, (name, category) in PHOTON_REACTIONS.items():
        reaction = reactions.create_group(name, track_order=True)
        reaction.attrs["ENDF-MT"] = mt
        reaction.attrs["category"] = category
        write_cross_section(
            reaction.create_group("xs"), grid, read_tab1(sections[23, mt])
        )
        quantity(reaction, "reference-frame", "LAB")
        reaction.create_group("cross-products")
        if mt in (502, 504):
            key = (27, mt)
            if key not in sections:
                if mt == 502:
                    raise ValueError(
                        "Coherent scattering requires MF 27/MT 502 form factors"
                    )
                continue
            field = "form-factor" if mt == 502 else "scattering-function"
            table = read_tab1(sections[key])
            if np.any(table[1] < 0):
                raise ValueError(f"{field} values must be nonnegative")
            write_table(reaction.create_group(field), table, "1/angstrom")
        elif mt in (515, 517):
            quantity(reaction, "field", "electron" if mt == 515 else "nuclear")
        elif mt == 522:
            parent = reaction.create_group("subshells")
            for mf, shell_mt in sorted(sections):
                if mf != 23 or not 534 <= shell_mt <= 599:
                    continue
                designator = shell_mt - 533
                if shells is not None and designator not in shells:
                    raise ValueError(
                        f"Photoelectric subshell {designator} has no EADL identity"
                    )
                table = read_tab1(sections[mf, shell_mt])
                if np.any(table[1] < 0):
                    raise ValueError(
                        "Photoelectric subshell cross sections must be nonnegative"
                    )
                child = parent.create_group(subshell_name(designator))
                quantity(child, "designator", designator)
                quantity(child, "binding-energy", table[0][0], "eV")
                write_table(child.create_group("xs"), table, "eV", "barn")
                child.create_group("cross-products")
        if verbose:
            print(f"  MT {mt}: {name} ({category})")
