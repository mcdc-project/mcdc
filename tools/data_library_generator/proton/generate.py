"""Prepare MC/DC proton data from ACE and NIST PSTAR inputs.

Writes temperature-independent ``proton-{nuclide}.h5`` files in the mcdc-data
format. ACE reaction interpretation, distribution conversion, and HDF5
serialization are handled by ``common.ace_converter``. This module coordinates
proton-specific target metadata, optional PSTAR stopping powers, and batch I/O.

Environment variables
    MCDC_ACELIB_PROTON  directory with proton ACE files (*.ace)
    MCDC_PSTAR_LIB      directory with NIST PSTAR stopping-power tables (<Symbol>.txt)
    MCDC_LIB_PROTON     output directory for the MC/DC HDF5 files

Usage
    python generate.py                 # convert only nuclides with no output yet
    python generate.py --rewrite       # regenerate everything
    python generate.py --verbose       # detailed per-reaction output
    python generate.py --only p-Al027.ace [more.ace ...]
"""

import argparse
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_library_generator.common.ace_converter import (
    EV_PER_MEV,
    PROTON_SYMBOL_TO_Z,
    prepare_proton_ace,
    write_proton_ace_target_properties,
    write_proton_reactions,
)
from data_library_generator.common.batch import ConversionErrors, add_keep_going_option
from data_library_generator.common.hdf5_writer import (
    atomic_file,
    interpolation,
    prepared_file_exists,
    quantity,
)

# ==============================================================================
# Constants
# ==============================================================================

# Radiation length [g/cm2] for Z = 1..118 (same order as PROTON_ELEMENT_SYMBOLS)
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

# ==============================================================================
# Small helpers
# ==============================================================================


def radiation_length(Z, A):
    """Return the radiation length in g/cm2 for a target, or None if unknown."""
    if (Z, A) == (1, 2):
        return DEUTERIUM_RADIATION_LENGTH
    return RADIATION_LENGTH[Z - 1] if 1 <= Z <= len(RADIATION_LENGTH) else None


def load_pstar_file(filepath):
    """Read valid PSTAR rows into energy (MeV) and stopping-power arrays."""
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
    """Write a validated PSTAR table in H1 units; return False if it is absent."""
    path = Path(pstar_dir) / f"{symbol}.txt"
    if not path.is_file():
        if verbose:
            print(f"  [warn] No PSTAR file for {symbol}")
        return False
    energy, stopping = load_pstar_file(path)
    if (
        len(energy) < 2
        or energy.shape != stopping.shape
        or not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(stopping))
        or np.any(energy <= 0)
        or np.any(np.diff(energy) <= 0)
        or np.any(stopping < 0)
    ):
        raise ValueError(f"{path}: invalid PSTAR stopping-power table")
    group = h5.create_group("stopping-power")
    group.attrs["type"] = "table"
    quantity(group, "grid", energy * EV_PER_MEV, "eV")
    quantity(group, "value", stopping * EV_PER_MEV, "eV cm2/g")
    interpolation(group, len(energy))
    return True


# ==============================================================================
# Per-file processing
# ==============================================================================


def process_ace_file(
    ace_path,
    output_dir,
    pstar_dir=None,
    verbose=False,
    rewrite=False,
    *,
    source_library=None,
):
    """Create or reuse one ACE-backed proton file with optional PSTAR data.

    ACE interpretation is handled by ``common.ace_converter``; new files are
    written atomically.
    """
    ace_path, output_dir = Path(ace_path), Path(output_dir)
    data = prepare_proton_ace(ace_path, pstar_dir, source_library=source_library)
    if data.pstar_path is None and pstar_dir is not None and verbose:
        print(f"  [warn] No PSTAR file for {data.symbol}")
    out_path = output_dir / data.output_name
    if not rewrite and prepared_file_exists(
        out_path, data.required_paths, data.sources
    ):
        if verbose:
            print(f"Skip {data.output_name}: output file already exists")
        return data.output_name
    if verbose:
        print(f"Create {data.output_name} from {ace_path.name}")
    with atomic_file(out_path, data.sources) as file:
        write_proton_ace_target_properties(file, data, radiation_length(data.Z, data.A))
        if data.pstar_path is not None:
            write_stopping_power(file, data.pstar_path.parent, data.symbol, verbose)
        write_proton_reactions(
            file, data.table, data.reactions, data.skipped_mts, verbose
        )
    return data.output_name


def process_pstar_only_file(
    symbol, A, awr, output_dir, pstar_dir, verbose=False, rewrite=False
):
    """Create a PSTAR-only target file when available, unless ACE data exist.

    Return the output filename, or ``None`` if the element has no PSTAR table.
    """
    pstar_path = Path(pstar_dir) / f"{symbol}.txt"
    if not pstar_path.is_file():
        return None
    Z = PROTON_SYMBOL_TO_Z[symbol]
    out_name = f"proton-{symbol}{A}.h5"
    out_path = Path(output_dir) / out_name
    if out_path.is_file():
        with h5py.File(out_path, "r") as file:
            if "reactions" in file:
                return out_name
    sources = {
        "PSTAR": {
            "title": "NIST PSTAR stopping-power tables",
            "version": "",
            "release-date": "",
            "reference": "NIST PSTAR",
            "comments": "",
            "format": "PSTAR",
            "format-version": "",
            "processing-date": "",
        }
    }
    required = (
        "atomic-number",
        "mass-number",
        "atomic-weight-ratio",
        "excitation-level",
        "fissionable",
        "radiation-length",
        "stopping-power/grid",
        "stopping-power/value",
    )
    if not rewrite and prepared_file_exists(out_path, required, sources):
        return out_name
    if verbose:
        print(f"Create {out_name} from PSTAR")
    with atomic_file(out_path, sources) as file:
        quantity(file, "atomic-number", Z)
        quantity(file, "mass-number", A)
        quantity(file, "atomic-weight-ratio", awr)
        quantity(file, "excitation-level", 0)
        quantity(file, "fissionable", False)
        quantity(file, "radiation-length", radiation_length(Z, A), "g/cm2")
        write_stopping_power(file, pstar_path.parent, symbol, verbose)
    return out_name


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Run selected ACE and PSTAR conversions and report any failures."""
    parser = argparse.ArgumentParser(
        description="Prepare MC/DC proton data from ACE and NIST PSTAR"
    )
    parser.add_argument(
        "--rewrite", action="store_true", help="regenerate existing files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Report files and reaction details"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ACE_FILE",
        help="convert only these ACE file names",
    )
    parser.add_argument(
        "--source-library",
        metavar="NAME",
        help="Label ACE source provenance; default: recognize the header or use unknown",
    )
    add_keep_going_option(parser)
    args = parser.parse_args()

    for variable in ("MCDC_ACELIB_PROTON", "MCDC_PSTAR_LIB", "MCDC_LIB_PROTON"):
        if not os.getenv(variable):
            parser.error(f"Environment variable {variable} is not set")
    ace_dir = Path(os.environ["MCDC_ACELIB_PROTON"])
    pstar_dir = Path(os.environ["MCDC_PSTAR_LIB"])
    output_dir = Path(os.environ["MCDC_LIB_PROTON"])
    if not ace_dir.is_dir():
        parser.error(f"ACE directory does not exist: {ace_dir}")
    if not pstar_dir.is_dir():
        parser.error(f"PSTAR directory does not exist: {pstar_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ace_files = sorted(path for path in ace_dir.iterdir() if path.suffix == ".ace")
    if args.only:
        available = {path.name for path in ace_files}
        missing = [name for name in args.only if name not in available]
        if missing:
            parser.error(f"ACE filenames not found in {ace_dir}: {', '.join(missing)}")
        selected = set(args.only)
        ace_files = [path for path in ace_files if path.name in selected]

    print(
        f"ACE directory: {ace_dir}\nPSTAR directory: {pstar_dir}\n"
        f"Output directory: {output_dir}\n"
    )
    failures = ConversionErrors(parser, keep_going=args.keep_going)
    progress = tqdm(ace_files, disable=args.verbose, unit="file")
    for path in progress:
        progress.set_postfix_str(path.name)
        try:
            process_ace_file(
                path,
                output_dir,
                pstar_dir,
                args.verbose,
                args.rewrite,
                source_library=args.source_library,
            )
        except (ValueError, OSError, AttributeError) as error:
            failures.handle(path.name, error, progress=progress)

    for symbol, A, awr in PSTAR_ONLY_ISOTOPES:
        filename = f"proton-{symbol}{A}.h5"
        try:
            out = process_pstar_only_file(
                symbol, A, awr, output_dir, pstar_dir, args.verbose, args.rewrite
            )
            if out is None and args.verbose:
                print(f"Skip {filename}: no PSTAR data")
        except (ValueError, OSError) as error:
            failures.handle(filename, error)

    failures.finish()
    print(f"\nDone. {len(ace_files)} ACE files processed.")


if __name__ == "__main__":
    main()
