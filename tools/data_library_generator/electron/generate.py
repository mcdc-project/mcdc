"""Coordinate EPRDATA14 electron preparation using the shared ACE converter.

MCDC_ACE_EPRDATA_LIB selects the directory containing concatenated EPR ACE inputs;
MCDC_LIB selects the prepared library. Source interpretation belongs to
common.ace_converter, while common.hdf5_writer manages staged publication.
"""

import argparse
import os
from pathlib import Path
import sys

import ACEtk
import h5py
from tqdm import tqdm

# ======================================================================================
# Shared converter and HDF5 writer imports
# ======================================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_library_generator.common.ace_converter import (
    ELECTRON_REACTIONS,
    SYMBOL_TO_Z,
    electron_target,
    prepare_epr_atomic_data,
    read_source_libraries,
    write_electron,
)
from data_library_generator.common.batch import (
    ConversionErrors,
    ace_input_directory,
    add_keep_going_option,
)
from data_library_generator.common.hdf5_writer import (
    atomic_file,
    check_format,
    prepared_file_exists,
)

# ======================================================================================
# Conversion and publication of one element
# ======================================================================================


def convert_table(
    ace, output_dir, rewrite=False, verbose=False, *, source_library=None
):
    """Prepare one electron projectile file and fill any missing atomic record.

    Compatible electron files are reused unless rewrite is requested.
    Publication preserves the previous file if conversion fails. Existing
    atomic evaluations are retained, including when rewriting electron data.
    The selected .14p tables identify the EPRDATA14 processed library; the
    original EPICS2014 evaluation title remains in the provenance record.
    """
    z, symbol = electron_target(ace)
    output_dir = Path(output_dir)
    path = output_dir / f"electron-{symbol}.h5"
    sources = read_source_libraries(
        ace,
        source_library="EPRDATA14" if source_library is None else source_library,
    )
    if path.is_file():
        with h5py.File(path, "r") as file:
            check_format(file)
            if int(file["atomic-number"][()]) != z:
                raise ValueError(f"{path}: inconsistent atomic-number")
    required = ["atomic-number", "xs-energy-grid"]
    for name in ELECTRON_REACTIONS:
        required.extend(
            f"reactions/{name}/{field}"
            for field in (
                "xs/value",
                "xs/offset",
                "xs/interpolation/laws",
                "xs/interpolation/boundaries",
                "reference-frame",
                "cross-products",
            )
        )
    required.extend(
        [
            "reactions/elastic/transport-xs",
            "reactions/elastic/large-angle/xs",
            "reactions/elastic/large-angle/angular-cosine-distribution",
            "reactions/excitation/energy-loss",
            "reactions/bremsstrahlung/energy-loss",
            "reactions/bremsstrahlung/photon-energy-distribution",
            "reactions/ionization/subshells",
        ]
    )
    if rewrite or not prepared_file_exists(path, required, sources):
        if verbose:
            print(f"Create {path.name} from {ace.zaid}")
        with atomic_file(path, sources) as file:
            write_electron(file, ace, verbose)
    elif verbose:
        print(f"Skip {path.name}: output file already exists")
    prepare_epr_atomic_data(ace, output_dir, sources, verbose)


# ======================================================================================
# Command-line options, input selection, and batch conversion
# ======================================================================================


def main():
    """Select .14p tables from concatenated ACE files and prepare electron data.

    The input directory comes from MCDC_ACE_EPRDATA_LIB and the output directory from
    MCDC_LIB. Optional filenames and element symbols restrict selection;
    otherwise every input file and every EPRDATA14 element is considered.
    Failures identify both the source filename and elemental table. With
    --keep-going, successful elements are retained and failures are summarized.
    """
    parser = argparse.ArgumentParser(
        description="Prepare MC/DC electron data from EPRDATA14"
    )
    parser.add_argument(
        "--ace-files", nargs="+", help="Select filenames from MCDC_ACE_EPRDATA_LIB"
    )
    parser.add_argument(
        "--elements", nargs="+", help="Select element symbols, e.g. H Mg Si"
    )
    parser.add_argument(
        "--rewrite", action="store_true", help="Regenerate electron files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Report conversion details"
    )
    parser.add_argument(
        "--source-library",
        metavar="NAME",
        help="Override the EPRDATA14 provenance label",
    )
    add_keep_going_option(parser)
    args = parser.parse_args()
    if not os.getenv("MCDC_LIB"):
        parser.error("Environment variable MCDC_LIB is not set")
    source_dir = ace_input_directory(parser, "MCDC_ACE_EPRDATA_LIB")
    output_dir = Path(os.environ["MCDC_LIB"])
    if args.elements and set(args.elements) - SYMBOL_TO_Z.keys():
        parser.error(
            "Unknown element symbols: "
            + ", ".join(sorted(set(args.elements) - SYMBOL_TO_Z.keys()))
        )
    paths = (
        [source_dir / name for name in args.ace_files]
        if args.ace_files
        else sorted(path for path in source_dir.iterdir() if path.is_file())
    )
    if not paths:
        parser.error(f"No ACE files found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ACE directory: {source_dir}\nOutput directory: {output_dir}\n")
    failures = ConversionErrors(parser, keep_going=args.keep_going)
    selected = set(args.elements or ())
    found = set()
    progress = tqdm(paths, disable=args.verbose, unit="file")
    for path in progress:
        progress.set_postfix_str(path.name)
        try:
            tables = ACEtk.PhotoatomicTable.from_concatenated_file(str(path))
        except (ValueError, OSError, RuntimeError) as error:
            failures.handle(path.name, error, progress=progress)
            continue
        for ace in tables:
            if not ace.zaid.endswith(".14p"):
                continue
            try:
                _, symbol = electron_target(ace)
                if selected and symbol not in selected:
                    continue
                found.add(symbol)
                convert_table(
                    ace,
                    output_dir,
                    args.rewrite,
                    args.verbose,
                    source_library=args.source_library,
                )
            except (ValueError, OSError) as error:
                failures.handle(f"{path.name}:{ace.zaid}", error, progress=progress)
    missing = selected - found
    if missing:
        failures.handle(
            "element selection",
            ValueError("No .14p tables for " + ", ".join(sorted(missing))),
        )
    failures.finish()
    if not found:
        parser.error("No EPRDATA14 (.14p) tables found")


if __name__ == "__main__":
    main()
