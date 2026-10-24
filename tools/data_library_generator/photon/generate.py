"""Prepare elemental photon mcdc-data from paired EPICS EPDL/EADL inputs.

``MCDC_EPICS_LIB`` contains the per-element ENDF files in one directory;
``MCDC_LIB`` selects the prepared library. Shared converters interpret ENDF
records, while the common writer manages provenance and staged publication.
This module selects elements and coordinates updates to their shared files.
"""

import argparse
import os
from pathlib import Path
import re
import sys

import h5py
from tqdm import tqdm

# ======================================================================================
# Shared converter and HDF5 writer imports
# ======================================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_library_generator.common.epics_converter import (
    PHOTON_REACTIONS,
    SYMBOL_TO_Z,
    Z_TO_SYMBOL,
    read_atomic_number,
    read_relaxation,
    read_sections,
    read_source_libraries,
    write_photon,
    write_relaxation,
)
from data_library_generator.common.hdf5_writer import (
    atomic_file,
    prepared_file_exists,
    quantity,
)

# ======================================================================================
# Conversion and publication of one element
# ======================================================================================


def convert_file(epdl_path, output_dir, eadl_path=None, rewrite=False, verbose=False):
    """Convert one EPDL evaluation and optional EADL data into an element file.

    Inputs must describe the same elemental target. Metadata is read from each
    source independently and stored under its library-release identifier.
    Compatible complete outputs are reused unless ``rewrite`` is requested.
    Only photon data and supplied atomic-relaxation data are replaced; unrelated
    projectile groups, source records, and element identity remain intact.
    Missing EADL input leaves any existing relaxation data unchanged.

    The common writer stages the entire update. Invalid source data, conflicting
    atomic numbers or provenance, incompatible output formats, and I/O errors
    propagate to the caller without replacing the previous element file.
    ``output_dir`` must already exist. Inputs and outputs are pathlib paths.
    """
    sections = read_sections(epdl_path)
    z = read_atomic_number(sections)
    symbol = Z_TO_SYMBOL[z]
    expected = f"EPDL.ZA{z:03d}000.endf"
    if epdl_path.name != expected:
        raise ValueError(f"{epdl_path}: filename does not match target {symbol}")
    sources = read_source_libraries(epdl_path, sections, "EPDL")
    shells = None
    if eadl_path is not None:
        atomic = read_sections(eadl_path)
        if read_atomic_number(atomic) != z:
            raise ValueError(f"{eadl_path}: EADL target differs from EPDL target")
        if (28, 533) not in atomic:
            raise ValueError(f"{eadl_path}: missing atomic relaxation MF 28/MT 533")
        shells = read_relaxation(atomic[28, 533])
        sources.update(read_source_libraries(eadl_path, atomic, "EADL"))
    path = output_dir / f"{symbol}.h5"
    required = ["atomic-number", "projectiles/photon/xs-energy-grid"]
    for mt, (name, _) in PHOTON_REACTIONS.items():
        base = f"projectiles/photon/reactions/{name}"
        required.extend(
            f"{base}/{field}"
            for field in (
                "xs/value",
                "xs/offset",
                "xs/interpolation/laws",
                "xs/interpolation/boundaries",
                "reference-frame",
                "cross-products",
            )
        )
        if mt == 502:
            required.append(f"{base}/form-factor")
        elif mt == 522:
            required.append(f"{base}/subshells")
        elif mt in (515, 517):
            required.append(f"{base}/field")
    if shells is not None:
        required.append("atomic-relaxation/subshells")
    if path.is_file():
        # Validate target identity even when the prepared-file reuse check passes.
        with h5py.File(path, "r") as file:
            if "atomic-number" in file and file["atomic-number"][()] != z:
                raise ValueError(f"{path}: inconsistent atomic-number")
    if not rewrite and prepared_file_exists(path, required, sources):
        if verbose:
            print(f"Skip {path.name}: output file already exists")
        return
    if verbose:
        print(f"Create {path.name} from {epdl_path.name}")
    with atomic_file(path, sources, preserve=True) as file:
        if "atomic-number" not in file:
            quantity(file, "atomic-number", z)
        if "projectiles/photon" in file:
            del file["projectiles/photon"]
        write_photon(file, sections, shells, verbose)
        if shells is not None:
            if "atomic-relaxation" in file:
                del file["atomic-relaxation"]
            write_relaxation(file.create_group("atomic-relaxation"), shells)


# ======================================================================================
# Command-line options, input selection, and batch conversion
# ======================================================================================


def main():
    """Select EPICS elements and prepare photon data under ``MCDC_LIB``.

    ``--elements`` selects chemical symbols; without it all correctly named
    EPDL files in ``MCDC_EPICS_LIB`` are processed. Matching EADL files are read
    from that same directory when present. ``--rewrite`` regenerates photon
    data, and ``--verbose`` replaces the progress bar with reaction details.
    Conversion stops at the first error and reports its source filename.
    """
    parser = argparse.ArgumentParser(description="Prepare MC/DC photon data from EPICS")
    parser.add_argument(
        "--elements", nargs="+", help="Select element symbols, e.g. H Mg Si"
    )
    parser.add_argument(
        "--rewrite", action="store_true", help="Regenerate existing photon data"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Report reaction details"
    )
    args = parser.parse_args()
    for variable in ("MCDC_EPICS_LIB", "MCDC_LIB"):
        if not os.getenv(variable):
            parser.error(f"Environment variable {variable} is not set")
    source_dir, output_dir = Path(os.environ["MCDC_EPICS_LIB"]), Path(
        os.environ["MCDC_LIB"]
    )
    if not source_dir.is_dir():
        parser.error(f"EPICS directory does not exist: {source_dir}")
    if args.elements:
        unknown = set(args.elements) - SYMBOL_TO_Z.keys()
        if unknown:
            parser.error(f"Unknown element symbols: {', '.join(sorted(unknown))}")
        paths = [
            source_dir / f"EPDL.ZA{SYMBOL_TO_Z[symbol]:03d}000.endf"
            for symbol in dict.fromkeys(args.elements)
        ]
    else:
        pattern = re.compile(r"EPDL\.ZA(\d{3})000\.endf")
        paths = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and pattern.fullmatch(path.name)
        )
    if not paths:
        parser.error(f"No EPDL element files found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"EPICS directory: {source_dir}\nOutput directory: {output_dir}\n")
    progress = tqdm(paths, disable=args.verbose, unit="element")
    for path in progress:
        progress.set_postfix_str(path.name)
        eadl_path = source_dir / path.name.replace("EPDL.", "EADL.", 1)
        try:
            convert_file(
                path,
                output_dir,
                eadl_path if eadl_path.is_file() else None,
                args.rewrite,
                args.verbose,
            )
        except (ValueError, OSError) as error:
            parser.exit(1, f"{path.name}: {error}\n")


if __name__ == "__main__":
    main()
