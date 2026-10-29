"""Coordinate neutron ACE conversion into the MC/DC HDF5 data format.

The command-line interface selects input tables from ``MCDC_ACE_LIB`` and writes
prepared data to ``MCDC_LIB``. Each table contributes neutron reactions to a
self-contained neutron projectile file, including target properties and provenance.

This module handles input selection, output-file coordination, and progress
reporting. Source interpretation and unit conversion belong to
``common.ace_converter``; format checks, provenance merging, and file publication
belong to ``common.hdf5_writer``.
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

# Use the tools directory as the shared import root for projectile scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_library_generator.common.ace_converter import (
    read_source_libraries,
    read_target,
    write_temperature,
)
from data_library_generator.common.batch import ConversionErrors, add_keep_going_option
from data_library_generator.common.hdf5_writer import (
    atomic_file,
    check_format,
    prepared_file_exists,
    quantity,
)

# ======================================================================================
# Conversion of one ACE table and its coordinated output files
# ======================================================================================


def convert_file(
    ace_path, output_dir, rewrite=False, verbose=False, *, source_library=None
):
    """Write one self-contained neutron projectile file from an ACE table.

    The file contains element and nuclide properties, target temperature,
    neutron reactions, and source provenance. A complete compatible output is
    reused without changing its timestamp unless rewrite is requested.
    Publication is staged, so conversion errors leave an existing output intact.
    source_library optionally labels the source record without changing the
    temperature convention inferred from the original ACE header.
    """
    ace = ACEtk.ContinuousEnergyTable.from_file(str(ace_path))
    target = read_target(ace)
    sources = read_source_libraries(
        ace, target, with_temperature=True, source_library=source_library
    )
    path = output_dir / target.filename
    fields = {
        "atomic-number": target.atomic_number,
        "mass-number": target.mass_number,
        "atomic-weight-ratio": ace.atomic_weight_ratio,
        "excitation-level": target.excitation_level,
        "fissionable": ace.fission_multiplicity_block is not None,
    }
    if path.is_file():
        with h5py.File(path, "r") as file:
            check_format(file)
            for name in ("atomic-number", "mass-number", "excitation-level"):
                if name in file and file[name][()] != fields[name]:
                    raise ValueError(f"{path}: inconsistent {name}")
    required = [*fields, "temperature", "kT", "xs-energy-grid", "reactions"]
    if not rewrite and prepared_file_exists(path, required, sources):
        if verbose:
            print(f"Skip {path.name}: output file already exists")
        return
    if verbose:
        print(f"Create {path.name} from {ace_path.name}")
    with atomic_file(path, sources) as file:
        for name, value in fields.items():
            quantity(file, name, value)
        write_temperature(file, ace, target, verbose=verbose)


# ======================================================================================
# Command-line options, input selection, and batch conversion
# ======================================================================================


def main():
    """Convert selected neutron ACE files using command-line and environment input.

    Returns
    -------
    None
        Returns after every selected file has been processed successfully.

    Raises
    ------
    SystemExit
        Argument parsing handles help with status 0 and invalid arguments
        with status 2. Missing environment variables or a missing input
        directory also produce status 2. A ``ValueError`` or ``OSError``
        from conversion produces status 1 with the input filename and error.
        Without ``--keep-going``, the first failure terminates the batch;
        with it, failures are summarized after all selected files are tried.
    OSError
        Creating the output directory or listing the input directory fails.

    Notes
    -----
    Options are read from ``sys.argv``. ``MCDC_ACE_LIB`` identifies the input
    directory and ``MCDC_LIB`` the output directory, which is created along
    with any missing parents. ``--ace-files`` selects filenames in the order
    supplied; without it, every regular file in the input directory is
    selected in sorted order, without a filename-suffix filter.
    ``--source-library`` sets the ACE provenance label without changing
    temperature interpretation.

    ``--rewrite`` and ``--verbose`` are passed to each conversion. Verbose
    output replaces the progress bar. ``--keep-going`` retains successful
    outputs and reports all caught conversion failures at the end. Exceptions
    other than the caught ``ValueError`` and ``OSError`` propagate unchanged.
    """
    # Parse file selection, rewrite behavior, and reporting options
    parser = argparse.ArgumentParser(description="Prepare MC/DC neutron data from ACE")
    parser.add_argument(
        "--rewrite", action="store_true", help="Regenerate existing files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Report reaction details"
    )
    parser.add_argument(
        "--ace-files", nargs="+", help="Select filenames from MCDC_ACE_LIB"
    )
    parser.add_argument(
        "--source-library",
        metavar="NAME",
        help="Label ACE source provenance; default: recognize the header or use unknown",
    )
    add_keep_going_option(parser)
    args = parser.parse_args()

    # Validate configured directories and create the output location
    for variable in ("MCDC_ACE_LIB", "MCDC_LIB"):
        if not os.getenv(variable):
            parser.error(f"Environment variable {variable} is not set")
    ace_dir = Path(os.environ["MCDC_ACE_LIB"])
    output_dir = Path(os.environ["MCDC_LIB"])
    if not ace_dir.is_dir():
        parser.error(f"ACE directory does not exist: {ace_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use explicit filenames or discover all regular files in deterministic order
    paths = (
        [ace_dir / name for name in args.ace_files]
        if args.ace_files
        else sorted(path for path in ace_dir.iterdir() if path.is_file())
    )
    # Process inputs sequentially and attach filenames to reported conversion errors
    print(f"ACE directory: {ace_dir}\nOutput directory: {output_dir}\n")
    failures = ConversionErrors(parser, keep_going=args.keep_going)
    progress = tqdm(paths, disable=args.verbose, unit="file")
    for path in progress:
        progress.set_postfix_str(path.name)
        try:
            convert_file(
                path,
                output_dir,
                args.rewrite,
                args.verbose,
                source_library=args.source_library,
            )
        except (ValueError, OSError) as error:
            failures.handle(path.name, error, progress=progress)
    failures.finish()


if __name__ == "__main__":
    main()
