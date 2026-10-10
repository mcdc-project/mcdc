"""Helpers for converting EPDL/EADL photon data into MC/DC's HDF5 format.

The ENDF-6 parsing here is deliberately minimal: EPDL and EADL use only the
TAB1 and LIST record types in the sections photon transport needs, and reading
them directly avoids a dependency on a general ENDF library.
"""

import re

import numpy as np

from mcdc.print_ import print_error, print_msg

# ======================================================================================
# Element symbols
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

SYMBOL_TO_Z = {symbol: Z for Z, symbol in Z_TO_SYMBOL.items()}

# ======================================================================================
# Reaction MT numbers (ENDF MF=23, photon interaction cross sections)
# ======================================================================================

#: Shared energy grid. EPDL tabulates the total on the unionized grid that every
#: other photon channel is a subset of, so MT-501's abscissa *is* the library's
#: ``xs_energy_grid``. The total itself is derived in the loader by summing the
#: channels, so it is not written as its own group.
MT_TOTAL = 501

#: ``(group name, [MT, ...])`` for each channel group, in the layout
#: ``Element.set_photon_data`` reads.
REACTION_GROUPS = (
    ("coherent_scattering", [502]),
    ("incoherent_scattering", [504]),
    ("photoelectric", [522]),
    # Pair production is split by field: MT-515 in the electron field and
    # MT-517 in the nuclear field. MT-516 is their sum and is written as an
    # unnumbered ``total`` subgroup rather than a third MT group.
    ("pair_production", [515, 517]),
)

#: Pair-production total, written as ``pair_production/total/xs``.
MT_PAIR_TOTAL = 516

#: Photoelectric subshells, MT-534 upward, one per ENDF subshell designator.
#: Electron ionization uses the same block for the same subshells.
MT_SUBSHELL_FIRST = 534
MT_SUBSHELL_LAST = 599

#: MF=27 form factors and scattering functions, tabulated against momentum
#: transfer in inverse angstroms.
MF27_FORM_FACTOR = 502
MF27_SCATTERING_FUNCTION = 504
MF27_ANOMALOUS_REAL = 505
MF27_ANOMALOUS_IMAGINARY = 506


def subshell_mt(designator):
    """Return the MT number for an ENDF subshell designator."""
    return MT_SUBSHELL_FIRST + int(designator) - 1


def subshell_designator(mt):
    """Return the ENDF subshell designator for an MT number."""
    return int(mt) - MT_SUBSHELL_FIRST + 1


# ======================================================================================
# ENDF-6 record parsing
# ======================================================================================

_ENDF_EXPONENT = re.compile(r"^([+-]?[0-9]*\.?[0-9]*)([+-][0-9]+)$")


def endf_float(field):
    """Parse one 11-column ENDF number field.

    ENDF writes exponents without the ``E``, as in ``1.234567+6``. Returns
    ``None`` for a blank field, which is how a short final record is padded.
    """
    field = field.strip()
    if not field:
        return None

    match = _ENDF_EXPONENT.match(field)
    if match:
        return float(match.group(1) + "e" + match.group(2))

    return float(field)


def read_sections(file_path):
    """Return ``{(MF, MT): [line, ...]}`` for one ENDF file.

    Section-end records carry MT=0 and are dropped, as are the tape and
    material-end records, so every returned block starts at its own HEAD record.
    """
    sections = {}
    with open(file_path) as file:
        for line in file:
            if len(line) < 75:
                continue

            mf_field = line[70:72].strip()
            mt_field = line[72:75].strip()
            if not mf_field.isdigit() or not mt_field.isdigit():
                continue

            MF = int(mf_field)
            MT = int(mt_field)
            if MT == 0 or MF == 0:
                continue

            sections.setdefault((MF, MT), []).append(line)

    return sections


def _read_number_fields(lines):
    """Read every 11-column number field from a list of ENDF records."""
    values = []
    for line in lines:
        for start in range(0, 66, 11):
            value = endf_float(line[start : start + 11])
            if value is not None:
                values.append(value)
    return values


def read_tab1(block):
    """Read a TAB1 record as ``(x, y)``.

    Layout: HEAD, then a CONT carrying ``NR`` and ``NP``, then ``NR``
    interpolation-range pairs, then ``NP`` ``(x, y)`` pairs.
    """
    if len(block) < 3:
        print_error("Malformed TAB1 record: fewer than three ENDF records")

    N_range = int(block[1][44:55])
    N_point = int(block[1][55:66])

    # Each interpolation range is two integers, six to a record.
    N_range_record = max(1, -(-2 * N_range // 6))
    range_values = _read_number_fields(block[2 : 2 + N_range_record])
    boundaries = [int(v) for v in range_values[0::2]][:N_range]
    laws = [int(v) for v in range_values[1::2]][:N_range]

    data_lines = block[2 + N_range_record :]
    values = _read_number_fields(data_lines)
    x = np.asarray(values[0::2], dtype=np.float64)[:N_point]
    y = np.asarray(values[1::2], dtype=np.float64)[:N_point]

    if len(x) != N_point or len(y) != N_point:
        print_error(
            f"TAB1 record declares NP={N_point} but yielded "
            f"{len(x)} abscissae and {len(y)} ordinates"
        )

    return x, y, boundaries, laws


def read_head_za_awr(block):
    """Read ``(ZA, AWR)`` from the HEAD record of a section."""
    ZA = endf_float(block[0][0:11])
    AWR = endf_float(block[0][11:22])
    return ZA, AWR


#: EADL keeps every subshell's relaxation data in one section, MF=28/MT=533,
#: rather than one section per subshell.
MF28_RELAXATION = 533


def read_relaxation(block):
    """Read EADL MF=28/MT=533 as ``{designator: subshell}``.

    Record layout, per ENDF-6: a HEAD carrying the subshell count ``NSS``, then
    for each subshell a CONT carrying the subshell designator ``SUBI`` and the
    transition count ``NTR``, a record carrying the binding energy ``EBI`` and
    electron occupancy ``ELN``, and finally ``NTR`` transition records of
    ``(SUBJ, SUBK, ETR, FTR)`` -- origin designator, secondary designator,
    transition energy in eV, and absolute probability.

    ``SUBK`` is what distinguishes the two de-excitation channels: zero is
    radiative, so a characteristic X-ray is emitted and transported; non-zero is
    non-radiative (Auger), whose electron is not transported.

    The probabilities are absolute and need not sum to one -- the radiative ones
    sum to the shell's fluorescence yield.
    """
    N_subshell = int(block[0][44:55])

    subshells = {}
    line = 1
    for _ in range(N_subshell):
        if line + 1 >= len(block):
            break

        designator = int(endf_float(block[line][0:11]))
        N_transition = int(block[line][55:66])
        line += 1

        binding_energy = endf_float(block[line][0:11])
        n_electrons = endf_float(block[line][11:22])
        line += 1

        origin = np.zeros(N_transition, dtype=np.int64)
        secondary = np.zeros(N_transition, dtype=np.int64)
        energy = np.zeros(N_transition, dtype=np.float64)
        probability = np.zeros(N_transition, dtype=np.float64)

        for index in range(N_transition):
            record = block[line]
            origin[index] = int(endf_float(record[0:11]))
            secondary[index] = int(endf_float(record[11:22]) or 0)
            energy[index] = endf_float(record[22:33])
            probability[index] = endf_float(record[33:44])
            line += 1

        subshells[designator] = {
            "binding_energy": binding_energy,
            "n_electrons": n_electrons,
            "origin_designator": origin,
            "secondary_designator": secondary,
            "energy": energy,
            "probability": probability,
        }

    return subshells


# ======================================================================================
# Interpolation onto the shared grid
# ======================================================================================


#: ENDF interpolation laws.
INTERPOLATION_HISTOGRAM = 1
INTERPOLATION_LINEAR = 2
INTERPOLATION_SEMILOGX = 3
INTERPOLATION_SEMILOGY = 4
INTERPOLATION_LOGLOG = 5


def interpolate(x, x_table, y_table, laws=None, boundaries=None):
    """Interpolate ``y_table`` onto ``x`` using the law the file declares.

    Honouring the declared law matters: every EPDL photon section is **INT=2
    (lin-lin)**, not the log-log an earlier revision of the porting plan
    assumed, and interpolating log-log instead changes the coherent cross
    section by over 20% in places. The grid is dense enough (6,682 points from
    1 eV to 100 GeV) that linear interpolation between stored points is
    accurate; the law is a property of the evaluation, not a choice.

    Mixed-law tables are handled per range. Zero-width bins -- which EPDL writes
    at every absorption edge by repeating the edge energy -- resolve to the
    upper value.
    """
    x = np.asarray(x, dtype=np.float64)

    if laws is None or len(laws) == 0:
        laws = [INTERPOLATION_LINEAR]
        boundaries = [len(x_table)]
    if boundaries is None or len(boundaries) != len(laws):
        boundaries = [len(x_table)] * len(laws)

    # A repeated abscissa is how ENDF writes a discontinuity: the first copy
    # carries the limit from below the edge and the second the limit from above.
    # Resolving both to the same side is wrong at exactly the five absorption
    # edges aluminium has, so the first copy of a repeated target energy is
    # looked up in the bin *below* it and the second in the bin above.
    index = np.searchsorted(x_table, x, side="right") - 1
    index_left = np.searchsorted(x_table, x, side="left") - 1

    first_of_pair = np.zeros(x.shape, dtype=bool)
    if x.size > 1:
        first_of_pair[:-1] = x[:-1] == x[1:]
    index = np.where(first_of_pair, index_left, index)

    index = np.clip(index, 0, len(x_table) - 2)

    # The law that governs each interpolated point, from the range its bin is in.
    law = np.full(x.shape, laws[-1], dtype=np.int64)
    start = 0
    for boundary, range_law in zip(boundaries, laws):
        in_range = (index >= start) & (index < boundary)
        law[in_range] = range_law
        start = boundary

    x0 = x_table[index]
    x1 = x_table[index + 1]
    y0 = y_table[index]
    y1 = y_table[index + 1]

    degenerate = x1 <= x0
    width = np.where(degenerate, 1.0, x1 - x0)
    fraction = np.where(degenerate, 1.0, (x - x0) / width)

    result = y0 + fraction * (y1 - y0)

    histogram = law == INTERPOLATION_HISTOGRAM
    if histogram.any():
        result = np.where(histogram, y0, result)

    positive = (y0 > 0.0) & (y1 > 0.0) & ~degenerate
    with np.errstate(divide="ignore", invalid="ignore"):
        semilogy = (law == INTERPOLATION_SEMILOGY) & positive
        if semilogy.any():
            value = y0 * (y1 / np.where(positive, y0, 1.0)) ** fraction
            result = np.where(semilogy, value, result)

        logx = (x0 > 0.0) & (x1 > x0) & (x > 0.0)
        semilogx = (law == INTERPOLATION_SEMILOGX) & logx
        if semilogx.any():
            ratio = np.log(np.where(logx, x / x0, 1.0)) / np.log(
                np.where(logx, x1 / x0, 2.0)
            )
            result = np.where(semilogx, y0 + ratio * (y1 - y0), result)

        loglog = (law == INTERPOLATION_LOGLOG) & positive & logx
        if loglog.any():
            exponent = np.log(np.where(positive, y1 / y0, 1.0)) / np.log(
                np.where(logx, x1 / x0, 2.0)
            )
            value = y0 * np.where(logx, x / x0, 1.0) ** exponent
            result = np.where(loglog, value, result)

    result = np.where(degenerate, y1, result)

    # Below the table the channel is closed. At a threshold written as a
    # repeated abscissa, the first copy is still below the opening, so it is
    # zero and the jump happens at the second copy.
    closed = np.where(first_of_pair, x <= x_table[0], x < x_table[0])
    return np.where(closed, 0.0, result)


def log_log_interpolate(x, x_table, y_table):
    """Deprecated alias retained for callers that assumed log-log."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)

    index = np.searchsorted(x_table, x, side="right") - 1
    index = np.clip(index, 0, len(x_table) - 2)

    x0 = x_table[index]
    x1 = x_table[index + 1]
    y0 = y_table[index]
    y1 = y_table[index + 1]

    linear = (x1 <= x0) | (y0 <= 0.0) | (y1 <= 0.0)

    # Log-log branch
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.log(np.where(linear, 1.0, y1 / y0)) / np.log(
            np.where(linear, 2.0, x1 / x0)
        )
        result = np.where(linear, 0.0, y0 * (x / np.where(linear, 1.0, x0)) ** exponent)

    # Linear branch, including the zero-width bins where the upper value wins
    width = np.where(x1 > x0, x1 - x0, 1.0)
    linear_value = np.where(x1 > x0, y0 + (x - x0) / width * (y1 - y0), y1)
    result = np.where(linear, linear_value, result)

    # Below the table, the channel is closed.
    result = np.where(x < x_table[0], 0.0, result)

    return result


def grid_offset(grid, threshold):
    """Return the index in ``grid`` where a channel beginning at ``threshold`` starts.

    The returned index is 0-based, which is the convention
    ``Element.set_photon_data`` applies when it does
    ``xs_container[offset:] += xs``.
    """
    offset = int(np.searchsorted(grid, threshold, side="left"))
    return min(offset, len(grid) - 1)


def project_onto_grid(grid, x_table, y_table, laws=None, boundaries=None):
    """Project a channel onto the shared grid, returning ``(offset, xs)``.

    The offset is 0 and the returned array spans the whole grid, matching the
    libraries already in circulation: they zero-pad every channel to the full
    grid rather than storing a trimmed array with a non-zero offset. Both load
    correctly under ``xs_container[offset:] += xs``, so this is a compatibility
    choice, not a correctness one.
    """
    xs = interpolate(grid, x_table, y_table, laws, boundaries)
    return 0, xs


# ======================================================================================
# HDF5 writing
# ======================================================================================


def create_mt_group(parent, MT, hyphenated=False):
    """Create the group for one MT and tag it with its MT attribute.

    Reaction groups are spelled ``MTnnn`` and photoelectric subshells
    ``MT-nnn``. The inconsistency is upstream's; it is matched rather than
    corrected so that a library written here loads with the same code that reads
    the libraries already in circulation.
    """
    name = f"MT-{MT}" if hyphenated else f"MT{MT}"
    group = parent.create_group(name)
    group.attrs["MT"] = np.int64(MT)
    return group


def write_xs(group, xs, offset):
    """Write a cross-section dataset with the offset and unit attributes."""
    dataset = group.create_dataset("xs", data=np.asarray(xs, dtype=np.float64))
    dataset.attrs["offset"] = np.int64(offset)
    dataset.attrs["unit"] = "barns"
    return dataset


def write_energy(group, name, values):
    """Write an energy dataset in eV, declaring the unit."""
    dataset = group.create_dataset(name, data=np.asarray(values, dtype=np.float64))
    dataset.attrs["unit"] = "eV"
    return dataset


def write_momentum_transfer(group, values):
    """Write a momentum-transfer grid in inverse angstroms."""
    dataset = group.create_dataset(
        "momentum_transfer", data=np.asarray(values, dtype=np.float64)
    )
    dataset.attrs["unit"] = "1/angstrom"
    return dataset


def write_reference_frame(group, frame="LAB"):
    """Write the reference-frame tag read by ``set_basic_properties``."""
    return group.create_dataset("reference_frame", data=frame.encode())


__all__ = [
    "MF27_ANOMALOUS_IMAGINARY",
    "MF27_ANOMALOUS_REAL",
    "MF27_FORM_FACTOR",
    "MF27_SCATTERING_FUNCTION",
    "MF28_RELAXATION",
    "MT_PAIR_TOTAL",
    "MT_SUBSHELL_FIRST",
    "MT_SUBSHELL_LAST",
    "MT_TOTAL",
    "REACTION_GROUPS",
    "SYMBOL_TO_Z",
    "Z_TO_SYMBOL",
    "create_mt_group",
    "endf_float",
    "grid_offset",
    "interpolate",
    "log_log_interpolate",
    "print_error",
    "print_msg",
    "project_onto_grid",
    "read_head_za_awr",
    "read_relaxation",
    "read_sections",
    "read_tab1",
    "subshell_designator",
    "subshell_mt",
    "write_energy",
    "write_momentum_transfer",
    "write_reference_frame",
    "write_xs",
]
