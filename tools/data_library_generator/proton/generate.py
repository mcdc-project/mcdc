# This file was written by Anthropic's Claude.
# Claude was used to understand & parse the ACE format,
# and generate the desired hdf5 structure.
"""
MC/DC proton data library generator: TENDL-2021 proton ACE files -> MC/DC HDF5.

Environment variables
    MCDC_ACELIB_PROTON  directory with the TENDL-2021 proton ACE files (*.ace)
    MCDC_PSTAR_LIB      directory with NIST PSTAR stopping-power tables (<Symbol>.txt)
    MCDC_LIB_PROTON     output directory for the MC/DC HDF5 files

Usage
    python generate.py                 # convert only nuclides with no output yet
    python generate.py --rewrite       # regenerate everything
    python generate.py --verbose       # detailed per-reaction output
    python generate.py --only p-Al027.ace [more.ace ...]

Design notes (see also the README in this directory)
  * ACEtk accessors are used by their exact names.  Two lower-case aliases in some
    ACEtk builds are mis-wired (secondary_particle_production_cross_section_block
    returns the TYRH block; photon_production_cross_section_block returns the main
    SIG block), so this script uses the upper-case ACE-name accessors
    (PTYPE, NTRO, MTRH, TYRH, SIGH, DLWH, ANDH, MTRP, SIGP, DLWP, ANDP).
  * Secondary-particle multiplicities are NOT in TYRH (it is always +-1 in TENDL
    proton files).  They are tabulated yields in SIGH (MFTYPE 12) / SIGP (MFTYPE 16):
    "mean number of products per occurrence of that reaction, versus incident energy".
    They are written as a `yield` table, never as a scalar.
  * Anything unexpected raises ConversionError.  A failed file is reported at the end
    and no partial output is left behind.
"""

import argparse
import os
import sys
import traceback
from dataclasses import dataclass

import ACEtk
import h5py
import numpy as np
from tqdm import tqdm

# ==============================================================================
# Constants
# ==============================================================================

# Temperature written to all files (TENDL proton ACE files report 0 K)
T_KELVIN = 0.0

ELEMENT_SYMBOLS = (  # index 0 -> Z=1
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

# Radiation length [g/cm2] for Z = 1..118 (same order as ELEMENT_SYMBOLS)
RADIATION_LENGTH = (
    63.04,
    94.32,
    82.77,
    65.19,
    52.68,
    42.70,
    37.99,
    34.24,
    32.93,
    28.93,
    27.74,
    25.03,
    24.01,
    21.82,
    21.21,
    19.50,
    19.28,
    19.55,
    17.32,
    16.14,
    16.55,
    16.16,
    15.84,
    14.94,
    14.64,
    13.84,
    13.62,
    12.68,
    12.86,
    12.43,
    12.47,
    12.25,
    11.94,
    11.91,
    11.42,
    11.37,
    11.03,
    10.76,
    10.41,
    10.20,
    9.92,
    9.80,
    9.58,
    9.48,
    9.27,
    9.20,
    8.97,
    9.00,
    8.85,
    8.82,
    8.73,
    8.83,
    8.48,
    8.48,
    8.31,
    8.31,
    8.14,
    7.96,
    7.76,
    7.71,
    7.51,
    7.57,
    7.44,
    7.48,
    7.36,
    7.32,
    7.23,
    7.14,
    7.03,
    7.02,
    6.92,
    6.89,
    6.82,
    6.76,
    6.69,
    6.68,
    6.59,
    6.54,
    6.46,
    6.44,
    6.42,
    6.37,
    6.29,
    6.16,
    6.07,
    6.28,
    6.19,
    6.15,
    6.06,
    6.07,
    5.93,
    6.00,
    5.87,
    5.93,
    5.80,
    5.79,
    5.69,
    5.68,
    5.61,
    5.62,
    5.55,
    5.48,
    5.45,
    5.47,
    5.40,
    5.34,
    5.27,
    5.17,
    5.26,
    5.24,
    5.18,
    5.16,
    5.10,
    5.08,
    5.01,
    5.00,
    4.95,
    4.88,
)

SYMBOL_TO_Z = {symbol: z for z, symbol in enumerate(ELEMENT_SYMBOLS, start=1)}

DEUTERIUM_RADIATION_LENGTH = 125.98  # g/cm2, for (Z, A) == (1, 2)

# Isotopes to generate stopping-power-only HDF5 files for when no ACE file exists
# (TENDL excludes H and He because TALYS does not apply).
# Format: (symbol, A, atomic_weight_ratio); AWR = atomic mass / neutron mass.
NEUTRON_MASS_U = 1.008664916
PSTAR_ONLY_ISOTOPES = (
    ("H", 1, 1.00794 / NEUTRON_MASS_U),  # natural H ~ H-1
    ("H", 2, 2.01410 / NEUTRON_MASS_U),  # deuterium
    ("He", 3, 3.01603 / NEUTRON_MASS_U),
    ("He", 4, 4.00260 / NEUTRON_MASS_U),
)

# --- Reaction bookkeeping -------------------------------------------------------
# MTs that are never written (total, non-elastic, "(z,anything)", disappearance).
ALWAYS_REDUNDANT_MTS = frozenset({1, 3, 10, 101})

# A "sum" MT is redundant ONLY if (some of) its partial reactions are present in the
# same file; otherwise the sum is the only record of that channel and must be kept.
# (TENDL-2021 proton files contain no partials at all, so every sum is kept there.)
SUM_MT_PARTIALS = {
    4: range(50, 92),  # (z,n) <- levels
    16: range(875, 892),  # (z,2n) <- levels
    103: range(600, 650),  # (z,p)
    104: range(650, 700),  # (z,d)
    105: range(700, 750),  # (z,t)
    106: range(750, 800),  # (z,He3)
    107: range(800, 850),  # (z,alpha)
}
MAX_MT = 891  # reactions above this (e.g. 9xx partial capture) are not written

# Upper edge [MeV] of the single-bin spectrum-probability grid written for reactions
# with one outgoing-energy distribution.  Kept from the original script; NOTE it is
# below the 200 MeV top of the TENDL data (see summary notes).
SPECTRUM_GRID_MAX_MEV = 30.0

# --- Secondary particles --------------------------------------------------------
# ACE particle-type id (PTYPE block) -> (product name, ZAP = 1000*Z + A)
ACE_PARTICLE_TO_PRODUCT = {
    1: ("neutron", 1),
    9: ("proton", 1001),
    31: ("deuteron", 1002),
    32: ("triton", 1003),
    33: ("He3", 2003),
    34: ("alpha", 2004),
}

# PLACEHOLDER: photons are not in PTYPE (they come from the separate photon-production
# blocks) and have no Z/A.  ZAP = 0 is used for them for now.  It is NOT a real ZA
# value; the MC/DC photon support has to decide what identifier it wants here.
PHOTON_NAME = "photon"
PHOTON_ZAP = 0
PHOTON_ACE_ID = 2  # ACE's own id for photons

# ACE photon-production reactions are numbered MTRP = 1000*MT + photon_index.
PHOTON_MT_FACTOR = 1000


class ConversionError(Exception):
    """The ACE data are not what this converter supports."""


def fail(message):
    raise ConversionError(message)


def note(message):
    print(f"  [note] {message}")


# ==============================================================================
# Small helpers
# ==============================================================================


def decode_ace_zaid(zaid):
    """Return (Z, A, S) from an ACE ZAID string such as '13027.00h'."""
    za = int(zaid.strip().split(".")[0])
    S = 0
    if za >= 600000:  # metastable convention: A + 300 + 100*S
        S = (za % 1000) // 400
        za = za - S * 400
    return za // 1000, za % 1000, S


def radiation_length(Z, A):
    if (Z, A) == (1, 2):
        return DEUTERIUM_RADIATION_LENGTH
    return RADIATION_LENGTH[Z - 1] if 1 <= Z <= len(RADIATION_LENGTH) else None


def frame_name(frame):
    if frame == ACEtk.ReferenceFrame.Laboratory:
        return "LAB"
    if frame == ACEtk.ReferenceFrame.CentreOfMass:
        return "COM"
    fail(f"Unknown reference frame {frame!r}")


def write_array(group, name, data, unit=None, dtype=np.float64):
    ds = group.create_dataset(name, data=np.array(data, dtype=dtype))
    if unit is not None:
        ds.attrs["unit"] = unit
    return ds


def load_pstar_file(filepath):
    """Two-column PSTAR file (Energy MeV, Stopping power MeV cm2/g)."""
    energies, stopping = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                energies.append(float(parts[0]))
                stopping.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(energies), np.array(stopping)


def write_stopping_power(h5, pstar_dir, symbol, verbose=False):
    """Write the stopping_power group if a PSTAR file exists. Returns True if written."""
    path = os.path.join(pstar_dir, f"{symbol}.txt")
    if not os.path.exists(path):
        if verbose:
            print(f"  [warn] No PSTAR file for {symbol}")
        return False
    energy, stopping = load_pstar_file(path)
    group = h5.create_group("stopping_power")
    write_array(group, "energy", energy, unit="MeV")
    write_array(group, "total_stopping_power", stopping, unit="MeV cm2/g")
    return True


# ==============================================================================
# Tabulated yields (multiplicity as a function of incident energy)
# ==============================================================================


def read_tabulated_yield(data, what):
    """
    Read an ACEtk TabulatedSecondaryParticleMultiplicity (SIGH/SIGP entries) or
    TabulatedMultiplicity (primary DLW) into (energy [MeV], value) arrays.
    Only linear-linear interpolation is supported (all TENDL-2021 files use it).
    """
    if not data.interpolation_data.is_linear_linear:
        fail(
            f"{what}: yield table is not linear-linear "
            f"(NBT={list(data.NBT)}, INT={list(data.INT)})"
        )
    energy = np.array(data.energies, dtype=np.float64)
    value = np.array(data.multiplicities, dtype=np.float64)
    if len(energy) != len(value) or len(energy) < 2:
        fail(
            f"{what}: malformed yield table ({len(energy)} energies, {len(value)} values)"
        )
    if np.any(np.diff(energy) < 0.0):
        fail(f"{what}: yield energies are not non-decreasing")
    if np.any(value < 0.0):
        fail(f"{what}: negative yield")
    return energy, value


def write_yield_table(group, name, energy, value):
    """
    Write a yield table as group `name` with datasets `energy` [MeV] and `value`
    (mean products per occurrence of the reaction), attrs type='tabulated' and
    interpolation='lin-lin' (the layout MC/DC uses for tabulated data).

    ACE tables may repeat an energy point (zero-width interval, e.g. 30.0, 30.0 in
    the MT-5 yields).  If the repeated points have the same value they are redundant
    and one is dropped.  If the values differ the table has a real discontinuity:
    both points are kept and attribute `has_discontinuity` is set.
    """
    energy = np.asarray(energy, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    same_as_previous = np.zeros(len(energy), dtype=bool)
    same_as_previous[1:] = (energy[1:] == energy[:-1]) & (value[1:] == value[:-1])
    energy, value = energy[~same_as_previous], value[~same_as_previous]

    table = group.create_group(name)
    table.attrs["type"] = "tabulated"
    table.attrs["interpolation"] = "lin-lin"
    table.attrs["has_discontinuity"] = bool(np.any(energy[1:] == energy[:-1]))
    write_array(table, "energy", energy, unit="MeV")
    write_array(table, "value", value)
    return table


# ==============================================================================
# Angular and energy distributions
# ==============================================================================


def write_cosine_distribution(data, group):
    """
    Write an angular distribution into `group` (attribute `type` is set here).
    Returns False if the angle is given inside the energy distribution (Kalbach-Mann).
    """
    C = ACEtk.continuous
    if isinstance(data, C.DistributionGivenElsewhere):
        group.attrs["type"] = "given_in_energy_distribution"
        return False
    if isinstance(data, C.FullyIsotropicDistribution):
        group.attrs["type"] = "isotropic"
        return True
    if not isinstance(data, C.AngularDistributionData):
        fail(f"Unsupported angular distribution {type(data).__name__}")

    group.attrs["type"] = "tabulated"
    group.attrs["unit"] = "MeV"
    write_array(group, "incident_energies", data.incident_energies)
    for i, sub in enumerate(data.distributions):
        eg = group.create_group(f"E_in_{i + 1}")
        if isinstance(sub, C.TabulatedAngularDistribution):
            eg.attrs["type"] = "tabulated"
            write_array(eg, "cosines", sub.cosines)
            write_array(eg, "pdf", sub.pdf)
            write_array(eg, "cdf", sub.cdf)
        elif isinstance(sub, C.IsotropicAngularDistribution):
            eg.attrs["type"] = "isotropic"
        else:
            fail(f"Unsupported angular sub-distribution {type(sub).__name__}")
    return True


def write_kalbach_mann(data, group):
    """KalbachMannDistributionData -> flat arrays; offset[i] = start of table i.
    Kalbach-Mann Distribution has LAW = 44"""
    group.attrs["type"] = "kalbach-mann"
    group.attrs["LAW"] = int(data.LAW)
    write_array(group, "energy", data.incident_energies, unit="MeV")

    offset, energy_out, pdf, cdf, precompound, slope = [], [], [], [], [], []
    for i in range(1, data.number_incident_energies + 1):
        dist = data.distribution(i)
        offset.append(len(energy_out))
        energy_out.extend(dist.outgoing_energies)
        pdf.extend(dist.pdf)
        cdf.extend(dist.cdf)
        precompound.extend(dist.precompound_fraction_values)
        slope.extend(dist.angular_distribution_slope_values)

    write_array(group, "offset", offset, dtype=np.int32)
    write_array(group, "energy_out", energy_out, unit="MeV")
    write_array(group, "pdf", pdf)
    write_array(group, "cdf", cdf)
    write_array(group, "precompound_factor", precompound)
    write_array(group, "angular_slope", slope)


def write_tabulated_energy(data, group):
    """
    LAW 4 (OutgoingEnergyDistributionData) -> flat arrays in MC/DC's 'tabulated'
    layout: energy (incident), offset, value (outgoing energies), pdf.  cdf is also
    written.

    ACE stores, per incident energy, INTT = 10*ND + I where I = 1 (histogram) or
    2 (linear-linear) and ND is the number of DISCRETE LINES at the start of that
    table (this is how TENDL photon spectra are stored: ND lines, then a continuum).
    For discrete lines the 'pdf' entries are line probabilities, not densities.
    Both are kept as `interpolation` and `n_discrete` (one entry per incident energy)
    so the consumer can tell them apart.  MC/DC's plain 'tabulated' reader ignores
    them and would treat the lines as continuum.
    """
    group.attrs["type"] = "tabulated"
    group.attrs["LAW"] = int(data.LAW)
    write_array(group, "energy", data.incident_energies, unit="MeV")

    offset, value, pdf, cdf, interpolation, n_discrete = [], [], [], [], [], []
    for i in range(1, data.number_incident_energies + 1):
        dist = data.distribution(i)
        # The first two words of a tabulated-energy table are INTT and NP. Check that
        # reading them this way agrees with the table length.
        intt, n_points = int(dist.xss(1)), int(dist.xss(2))
        if n_points != len(dist.outgoing_energies):
            fail(
                "Unexpected TabulatedEnergyDistribution layout: NP read from xss "
                f"({n_points}) != number of outgoing energies "
                f"({len(dist.outgoing_energies)})"
            )
        discrete, scheme = divmod(intt, 10)
        if scheme not in (1, 2) or discrete > n_points:
            fail(f"Unexpected INTT value {intt} for a table with {n_points} points")
        offset.append(len(value))
        value.extend(dist.outgoing_energies)
        pdf.extend(dist.pdf)
        cdf.extend(dist.cdf)
        interpolation.append(scheme)
        n_discrete.append(discrete)

    write_array(group, "offset", offset, dtype=np.int32)
    write_array(group, "value", value, unit="MeV")
    write_array(group, "pdf", pdf)
    write_array(group, "cdf", cdf)
    write_array(group, "interpolation", interpolation, dtype=np.int32)
    write_array(group, "n_discrete", n_discrete, dtype=np.int32)


def write_energy_distribution(data, group):
    """Write one outgoing-energy distribution; returns its `type` string."""
    C = ACEtk.continuous
    if isinstance(data, C.KalbachMannDistributionData):
        write_kalbach_mann(data, group)
    elif isinstance(data, C.OutgoingEnergyDistributionData):
        write_tabulated_energy(data, group)
    elif isinstance(data, C.LevelScatteringDistribution):
        group.attrs["type"] = "level-scattering"
        group.attrs["LAW"] = int(data.LAW)
        group.create_dataset("C1", data=data.C1)
        group.create_dataset("C2", data=data.C2)
    else:
        fail(
            f"Unsupported energy distribution {type(data).__name__} "
            "(the MC/DC proton loader reads Kalbach-Mann, tabulated and "
            "level-scattering spectra)"
        )
    return group.attrs["type"]


# ==============================================================================
# Secondary particles (neutrons, d, t, He3, alpha and photons)
# ==============================================================================


def write_secondary_product(
    parent, name, *, zap, ace_id, frame, yield_energy, yield_value, energy, angle
):
    """
    One product of one reaction, as `parent/<name>`:
      attrs   ZAP, particle_name, ace_particle_id, reference_frame ("LAB"/"COM")
      yield/  energy [MeV], value  -- mean number of products per reaction occurrence
      kalbach_mann/ or energy_spectrum/   the single outgoing-energy distribution
      angular_cosine_distribution/        angle (or 'given_in_energy_distribution')
    """
    C = ACEtk.continuous
    if name in parent:
        fail(f"Duplicate secondary product {name!r} in {parent.name}")
    if isinstance(energy, C.MultiDistributionData):
        fail(
            f"{parent.name}/{name}: several energy distributions for one product are "
            "not supported (MC/DC reads only the first)"
        )

    product = parent.create_group(name)
    product.attrs["ZAP"] = zap
    product.attrs["particle_name"] = name.split("_")[0]
    product.attrs["ace_particle_id"] = ace_id
    product.attrs["reference_frame"] = frame
    write_yield_table(product, "yield", yield_energy, yield_value)

    is_kalbach_mann = isinstance(energy, C.KalbachMannDistributionData)
    energy_group = product.create_group(
        "kalbach_mann" if is_kalbach_mann else "energy_spectrum"
    )
    write_energy_distribution(energy, energy_group)

    angle_group = product.create_group("angular_cosine_distribution")
    angle_in_energy_block = not write_cosine_distribution(angle, angle_group)
    if angle_in_energy_block and not is_kalbach_mann:
        fail(
            f"{parent.name}/{name}: angle 'given elsewhere' but the spectrum is not Kalbach-Mann"
        )
    return product


def _products_group(MT, reaction_groups, skipped_MTs, what, verbose):
    """The `secondary_products` group of reaction MT, or None if MT was deliberately skipped."""
    if MT in reaction_groups:
        return reaction_groups[MT].require_group("secondary_products")
    if MT in skipped_MTs:
        if verbose:
            note(f"{what}: MT-{MT:03} is a redundant sum in this file; skipped")
        return None
    fail(f"{what}: MT-{MT:03} has no reaction group in the output")


def write_secondary_particles(table, reaction_groups, skipped_MTs, verbose=False):
    """
    Write all non-proton products of all reactions.
      reaction_groups: {MT: HDF5 group of that reaction}
      skipped_MTs:     MTs deliberately left out of the output (redundant sums)
    Returns the number of products written.
    """
    C = ACEtk.continuous
    written = 0

    # ---- Particles with their own blocks: neutrons, d, t, He3, alpha ----------
    for p in range(1, table.NTYPE + 1):
        ace_id = table.PTYPE.particle_type(p)
        if ace_id not in ACE_PARTICLE_TO_PRODUCT:
            fail(f"Unknown ACE secondary-particle id {ace_id}")
        if ace_id == 9:
            fail(
                "Proton listed as a secondary particle type; protons are the primary here"
            )
        name, zap = ACE_PARTICLE_TO_PRODUCT[ace_id]

        n_rx = table.NTRO.NP(p)
        mtrh, tyrh, sigh = table.MTRH(p), table.TYRH(p), table.SIGH(p)
        dlwh, andh = table.DLWH(p), table.ANDH(p)
        sizes = (
            mtrh.number_reactions, tyrh.number_reactions, len(sigh.data),
            dlwh.number_reactions, andh.number_reactions,
        )  # fmt: skip
        if any(s != n_rx for s in sizes):
            fail(f"{name}: inconsistent block sizes {sizes}, expected {n_rx} each")

        for j in range(1, n_rx + 1):
            MT = mtrh.MT(j)
            what = f"{name} from MT-{MT:03}"
            if tyrh.multiplicity(j) != 1:
                fail(f"{what}: TYRH multiplicity {tyrh.multiplicity(j)} (expected 1)")
            yield_data = sigh.data[j - 1]
            if yield_data.MT != MT:
                fail(f"{what}: SIGH entry is for MT-{yield_data.MT:03}, not MT-{MT:03}")
            if yield_data.MFTYPE != 12:
                fail(
                    f"{what}: SIGH MFTYPE {yield_data.MFTYPE} (only 12, a yield, is supported)"
                )
            group = _products_group(MT, reaction_groups, skipped_MTs, what, verbose)
            if group is None:
                continue

            y_energy, y_value = read_tabulated_yield(yield_data, what)
            write_secondary_product(
                group, name, zap=zap, ace_id=ace_id,
                frame=frame_name(tyrh.reference_frame(j)),
                yield_energy=y_energy, yield_value=y_value,
                energy=dlwh.energy_distribution_data(j),
                angle=andh.angular_distribution_data(j),
            )  # fmt: skip
            written += 1
            if verbose:
                print(
                    f"      MT-{MT:03} {name}: yield {y_value.min():g}..{y_value.max():g}"
                )

    # ---- Photons (separate blocks MTRP / SIGP / DLWP / ANDP) ------------------
    n_photon = table.NTRP
    if n_photon:
        mtrp, sigp, dlwp, andp = table.MTRP, table.SIGP, table.DLWP, table.ANDP
        sizes = (
            mtrp.number_reactions, len(sigp.data),
            dlwp.number_reactions, andp.number_reactions,
        )  # fmt: skip
        if any(s != n_photon for s in sizes):
            fail(f"photon: inconsistent block sizes {sizes}, expected {n_photon} each")

        for k in range(1, n_photon + 1):
            MT, index = divmod(mtrp.MT(k), PHOTON_MT_FACTOR)
            name = PHOTON_NAME if index == 1 else f"{PHOTON_NAME}_{index}"
            what = f"{name} from MT-{MT:03}"
            yield_data = sigp.data[k - 1]
            if not isinstance(yield_data, C.TabulatedSecondaryParticleMultiplicity):
                fail(
                    f"{what}: SIGP entry is a {type(yield_data).__name__}, not a yield table"
                )
            if yield_data.MT != MT or yield_data.MFTYPE not in (12, 16):
                fail(
                    f"{what}: unexpected SIGP entry (MT={yield_data.MT}, MFTYPE={yield_data.MFTYPE})"
                )
            group = _products_group(MT, reaction_groups, skipped_MTs, what, verbose)
            if group is None:
                continue

            # ACE photon production has no reference-frame flag (ACEtk gives None):
            # photons are emitted in the laboratory frame.
            frame = dlwp.reference_frame(k)
            y_energy, y_value = read_tabulated_yield(yield_data, what)
            write_secondary_product(
                group, name, zap=PHOTON_ZAP, ace_id=PHOTON_ACE_ID,
                frame="LAB" if frame is None else frame_name(frame),
                yield_energy=y_energy, yield_value=y_value,
                energy=dlwp.energy_distribution_data(k),
                angle=andp.angular_distribution_data(k),
            )  # fmt: skip
            written += 1
            if verbose:
                print(
                    f"      MT-{MT:03} {name}: yield {y_value.min():g}..{y_value.max():g}"
                )
    return written


# ==============================================================================
# Primary reactions
# ==============================================================================


@dataclass
class Reaction:
    MT: int
    index: int  # 1-based position in the MTR / TYR / LQR / SIG blocks
    group: str  # "capture" (no outgoing proton) or "inelastic_scattering"
    multiplicity: int = 0  # fixed number of outgoing protons
    proton_yield: tuple = (
        None  # (energy, value) when the proton yield depends on energy
    )


def classify_reactions(table):
    """
    Sort the file's reactions into capture / inelastic and find the redundant ones.
    Returns (reactions, skipped_MTs).

    A reaction with outgoing-proton multiplicity 0 is "capture" (the proton is
    absorbed; any secondaries come from the product yields).  A positive multiplicity
    is "inelastic".  |TYR| > 100 means the proton yield is energy dependent; that
    table lives in the primary DLW block (MT-5 in TENDL files).
    """
    mtr = table.reaction_number_block
    tyr = table.frame_and_multiplicity_block
    dlw = table.energy_distribution_block
    n = tyr.number_reactions
    present = {mtr.MT(i) for i in range(1, n + 1)}

    skipped = set(ALWAYS_REDUNDANT_MTS)
    for sum_MT, partials in SUM_MT_PARTIALS.items():
        if sum_MT in present and present.intersection(partials):
            skipped.add(sum_MT)
    skipped |= {MT for MT in present if MT > MAX_MT}

    if (
        present.intersection((18, 19, 20, 21, 38))
        or table.fission_multiplicity_block is not None
    ):
        fail("Fission data in a proton ACE file are not supported by this converter")

    reactions = []
    for i in range(1, n + 1):
        MT = mtr.MT(i)
        if MT == 2 or MT in skipped:  # elastic comes from the principal xs block
            continue
        raw = tyr.multiplicity(i)
        if raw == 0:
            reactions.append(Reaction(MT, i, "capture"))
        elif raw > 100:
            if i > dlw.number_reactions:
                fail(f"MT-{MT:03}: energy-dependent multiplicity but no DLW entry")
            data = dlw.multiplicity_data(i)
            if not isinstance(data, ACEtk.continuous.TabulatedMultiplicity):
                fail(
                    f"MT-{MT:03}: TYR={raw} but multiplicity data is {type(data).__name__}"
                )
            reactions.append(
                Reaction(MT, i, "inelastic_scattering",
                         proton_yield=read_tabulated_yield(data, f"proton yield of MT-{MT:03}"))
            )  # fmt: skip
        else:
            reactions.append(Reaction(MT, i, "inelastic_scattering", multiplicity=raw))
    return reactions, skipped


def write_primary_spectra(group, data):
    """Outgoing-proton energy spectra with their probability weights."""
    C = ACEtk.continuous
    if not isinstance(data, C.MultiDistributionData):
        write_array(
            group, "spectrum_probability_grid", [0.0, SPECTRUM_GRID_MAX_MEV], unit="MeV"
        )
        group.create_dataset("spectrum_probability", data=np.array([[1.0]]))
        write_energy_distribution(data, group.create_group("energy_spectrum-1"))
        return

    n_dist = data.number_distributions
    probs = data.probabilities
    if all(p.number_interpolation_regions == 0 for p in probs):
        prob_grid = np.array([0.0, SPECTRUM_GRID_MAX_MEV])
        prob = np.zeros((1, n_dist))
        for k in range(n_dist):
            prob[0, k] = max(data.probability(k + 1).probabilities)
    elif all(p.number_interpolation_regions == 1 for p in probs) and all(
        p.interpolants[0] == 1 for p in probs
    ):
        prob_grid = np.array(data.probability(1).energies)
        prob = np.zeros((len(prob_grid) - 1, n_dist))
        for k in range(n_dist):
            prob[:, k] = np.array(data.probability(k + 1).probabilities[:-1])
    else:
        fail("Unsupported multi-distribution probability interpolation")

    write_array(group, "spectrum_probability_grid", prob_grid, unit="MeV")
    group.create_dataset("spectrum_probability", data=prob)
    for k in range(n_dist):
        write_energy_distribution(
            data.distribution(k + 1), group.create_group(f"energy_spectrum-{k + 1}")
        )


def write_reaction(group, table, rx, verbose):
    """xs, Q-value, frame and (for inelastic) multiplicity, angle and spectra."""
    tyr = table.frame_and_multiplicity_block
    xs_block = table.cross_section_block
    group.attrs["MT"] = rx.MT

    xs = write_array(group, "xs", xs_block.cross_sections(rx.index), unit="barns")
    xs.attrs["offset"] = xs_block.energy_index(rx.index) - 1
    write_array(
        group,
        "Q-value",
        table.reaction_qvalue_block.q_value(rx.index),
        unit="MeV",
        dtype=float,
    )
    group.create_dataset(
        "reference_frame", data=frame_name(tyr.reference_frame(rx.index))
    )

    if rx.group == "capture":
        return

    if rx.proton_yield is None:
        group.create_dataset("multiplicity", data=rx.multiplicity)
    else:  # energy-dependent: replaces the integer `multiplicity` dataset
        write_yield_table(group, "proton_yield", *rx.proton_yield)

    angle = table.angular_distribution_block.angular_distribution_data(rx.index)
    angle_group = group.create_group("angular_cosine_distribution")
    if not write_cosine_distribution(angle, angle_group) and verbose:
        note(f"MT-{rx.MT:03} angular distribution is given in the energy block")

    spectra = table.energy_distribution_block.energy_distribution_data(rx.index)
    write_primary_spectra(group, spectra)


# ==============================================================================
# Per-file processing
# ==============================================================================


def write_nuclide(h5, table, nuclide, Z, A, S, pstar_dir, verbose):
    hdr = table.header
    h5.attrs["source_title"] = hdr.title
    h5.attrs["source_version"] = hdr.version
    h5.attrs["source_date"] = hdr.date
    comments = getattr(hdr, "comments", None)  # optional ACE header field
    if comments is not None:
        h5.attrs["source_comments"] = comments

    h5.create_dataset("nuclide_name", data=nuclide)
    h5.create_dataset("excitation_level", data=S)
    write_array(h5, "temperature", T_KELVIN, unit="K", dtype=float)
    h5.create_dataset("atomic_number", data=table.atom_number)
    h5.create_dataset("mass_number", data=table.mass_number)
    h5.create_dataset("atomic_weight_ratio", data=table.atomic_weight_ratio)
    write_array(
        h5, "radiation_length", radiation_length(Z, A), unit="g/cm2", dtype=float
    )
    h5.create_dataset(
        "fissionable", data=False
    )  # fission is rejected in classify_reactions
    if pstar_dir is not None:
        write_stopping_power(h5, pstar_dir, ELEMENT_SYMBOLS[Z - 1], verbose=verbose)

    reactions, skipped = classify_reactions(table)
    proton_reactions = h5.create_group("proton_reactions")
    principal = table.principal_cross_section_block
    write_array(proton_reactions, "xs_energy_grid", principal.energies, unit="MeV")

    # Elastic scattering: cross section from the principal block, always COM.
    elastic = proton_reactions.create_group("elastic_scattering/MT-002")
    elastic.attrs["MT"] = 2
    xs = write_array(elastic, "xs", principal.elastic, unit="barns")
    xs.attrs["offset"] = 0
    write_array(elastic, "Q-value", 0.0, unit="MeV", dtype=float)
    elastic.create_dataset("reference_frame", data="COM")
    angle = table.angular_distribution_block.angular_distribution_data(0)
    if (
        not write_cosine_distribution(
            angle, elastic.create_group("angular_cosine_distribution")
        )
        and verbose
    ):
        note("MT-002 angular distribution is given in the energy block")

    reaction_groups = {}
    for rx in reactions:
        group = proton_reactions.create_group(f"{rx.group}/MT-{rx.MT:03}")
        write_reaction(group, table, rx, verbose)
        reaction_groups[rx.MT] = group
    if verbose:
        by_group = {
            g: [r.MT for r in reactions if r.group == g]
            for g in ("capture", "inelastic_scattering")
        }
        print(
            f"  Capture: {by_group['capture']}\n  Inelastic: {by_group['inelastic_scattering']}"
        )
        print(f"  Not written (redundant sums / MT > {MAX_MT}): {sorted(skipped)}")

    n_products = write_secondary_particles(
        table, reaction_groups, skipped, verbose=verbose
    )
    if verbose:
        print(f"  Secondary products written: {n_products}")


def process_ace_file(ace_path, output_dir, pstar_dir=None, verbose=False):
    """Convert one proton ACE file to HDF5. Returns the output file name."""
    with open(ace_path) as f:
        header = ACEtk.Header.from_string(f.readline())
    Z, A, S = decode_ace_zaid(header.zaid)
    if not 1 <= Z <= len(ELEMENT_SYMBOLS):
        fail(f"Unsupported Z={Z} in {ace_path}")
    symbol = ELEMENT_SYMBOLS[Z - 1]
    nuclide = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"

    table = ACEtk.ContinuousEnergyTable.from_file(ace_path)
    out_name = f"{nuclide}-{T_KELVIN}K.h5"
    out_path = os.path.join(output_dir, out_name)
    tmp_path = out_path + ".partial"
    if verbose:
        print(
            f"\n{'=' * 80}\n  {os.path.basename(ace_path)}  ->  {out_name}  (Z={Z} A={A} S={S})"
        )

    # Write to a temporary name: a failed conversion must not leave a half-written
    # file that a later (non --rewrite) run would mistake for a finished one.
    try:
        with h5py.File(tmp_path, "w") as h5:
            write_nuclide(h5, table, nuclide, Z, A, S, pstar_dir, verbose)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_name


def process_pstar_only_file(symbol, A, awr, output_dir, pstar_dir, verbose=False):
    """Minimal HDF5 file (stopping power only). Returns the file name, or None if no PSTAR file."""
    pstar_path = os.path.join(pstar_dir, f"{symbol}.txt")
    if not os.path.exists(pstar_path):
        return None
    Z = SYMBOL_TO_Z[symbol]
    nuclide = f"{symbol}{A}"
    out_name = f"{nuclide}-{T_KELVIN}K.h5"
    out_path = os.path.join(output_dir, out_name)
    tmp_path = out_path + ".partial"
    if verbose:
        print(f"\n{'=' * 80}\n  (no ACE)  ->  {out_name}  [stopping power only]")

    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["source_title"] = "PSTAR (NIST) stopping power only - no ACE data"
            h5.attrs["source_version"] = "N/A"
            h5.attrs["source_date"] = "N/A"
            h5.create_dataset("nuclide_name", data=nuclide)
            h5.create_dataset("excitation_level", data=0)
            write_array(h5, "temperature", T_KELVIN, unit="K", dtype=float)
            h5.create_dataset("atomic_number", data=Z)
            h5.create_dataset("mass_number", data=A)
            h5.create_dataset("atomic_weight_ratio", data=awr)
            write_array(
                h5,
                "radiation_length",
                radiation_length(Z, A),
                unit="g/cm2",
                dtype=float,
            )
            h5.create_dataset("fissionable", data=False)
            write_stopping_power(h5, pstar_dir, symbol, verbose=verbose)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_name


# ==============================================================================
# Main
# ==============================================================================


def nuclide_of(ace_path):
    """Nuclide name (e.g. 'Al27') of an ACE file, from its header."""
    with open(ace_path) as f:
        header = ACEtk.Header.from_string(f.readline())
    Z, A, S = decode_ace_zaid(header.zaid)
    symbol = ELEMENT_SYMBOLS[Z - 1] if 1 <= Z <= len(ELEMENT_SYMBOLS) else f"Z{Z}"
    return f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"


def require_env(name):
    value = os.getenv(name)
    if value is None:
        sys.exit(f"[ERROR] Environment variable ${name} is not set.")
    return value


def main():
    parser = argparse.ArgumentParser(description="MC/DC proton data generator")
    parser.add_argument(
        "--rewrite", action="store_true", help="regenerate existing files"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ACE_FILE",
        help="convert only these ACE file names",
    )
    args = parser.parse_args()

    ace_dir = require_env("MCDC_ACELIB_PROTON")
    pstar_dir = require_env("MCDC_PSTAR_LIB")
    output_dir = require_env("MCDC_LIB_PROTON")
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"\nACE directory   : {ace_dir}\nPSTAR directory : {pstar_dir}\nOutput directory: {output_dir}\n"
    )

    ace_files = sorted(f for f in os.listdir(ace_dir) if f.endswith(".ace"))
    if args.only:
        missing = [f for f in args.only if f not in ace_files]
        if missing:
            sys.exit(f"[ERROR] Not found in {ace_dir}: {missing}")
        ace_files = sorted(args.only)

    # Pass 1: ACE files
    done = {f.rsplit("-", 1)[0] for f in os.listdir(output_dir) if f.endswith(".h5")}
    errors, targets = [], []
    for name in ace_files:
        if args.rewrite:
            targets.append(name)
            continue
        try:
            if nuclide_of(os.path.join(ace_dir, name)) not in done:
                targets.append(name)
        except Exception:
            targets.append(name)  # unreadable header: let the conversion report it

    bar = tqdm(
        targets,
        disable=args.verbose,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
    )
    for name in bar:
        bar.set_postfix_str(name)
        try:
            out = process_ace_file(
                os.path.join(ace_dir, name), output_dir, pstar_dir, args.verbose
            )
            if args.verbose:
                print(f"  -> wrote {out}")
        except Exception as exc:
            errors.append((name, f"{type(exc).__name__}: {exc}"))
            if args.verbose:
                traceback.print_exc()

    # Pass 2: stopping-power-only isotopes (H, He)
    n_pstar = 0
    existing = set(os.listdir(output_dir))
    for symbol, A, awr in PSTAR_ONLY_ISOTOPES:
        nuclide = f"{symbol}{A}"
        if not args.rewrite and f"{nuclide}-{T_KELVIN}K.h5" in existing:
            continue
        try:
            out = process_pstar_only_file(
                symbol, A, awr, output_dir, pstar_dir, args.verbose
            )
            if out is None:
                if args.verbose:
                    print(f"  [skip] No PSTAR data for {nuclide}")
            else:
                n_pstar += 1
                if args.verbose:
                    print(f"  -> wrote {out}  [stopping power only]")
        except Exception as exc:
            errors.append((nuclide, f"{type(exc).__name__}: {exc}"))
            if args.verbose:
                traceback.print_exc()

    n_ok = len(targets) + n_pstar - sum(1 for name, _ in errors)
    print(f"\nDone. {n_ok} written, {len(errors)} failed.")
    if errors:
        print("\nFailed:")
        for name, message in errors:
            print(f"  {name}: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
