"""Coordinate neutron ACE conversion into the MC/DC HDF5 data format.

The command-line interface selects input tables from ``MCDC_ACE_LIB`` and writes
prepared data to ``MCDC_LIB``. Each table contributes neutron reactions to a
nuclide-temperature file and identity/provenance records to the shared element
and nuclide files.

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
from data_library_generator.common.hdf5_writer import (
    atomic_file,
    library_exists,
    write_companion,
)

# ======================================================================================
# Conversion of one ACE table and its coordinated output files
# ======================================================================================


def convert_file(ace_path, output_dir, rewrite=False, verbose=False):
    """Prepare neutron data from one ACE file and update its companion files.

    Parameters
    ----------
    ace_path : pathlib.Path
        Input ACE file parsed as an ACEtk continuous-energy table.
    output_dir : pathlib.Path
        Existing output directory for the element, nuclide, and
        nuclide-temperature files. The caller creates this directory.
    rewrite : bool, optional
        Rebuild the temperature file even if all three outputs pass the
        prepared-file reuse check. Shared companion files are merged.
    verbose : bool, optional
        Report whether the temperature file is skipped or created, and
        report the retained reaction channels during conversion.

    Returns
    -------
    None
        Returns after reusing prepared files or completing conversion.

    Raises
    ------
    ValueError
        Source metadata or temperature cannot be interpreted; an existing
        companion file has incompatible format or conflicting identity or
        provenance; or reaction/distribution conversion rejects source data.
        Without ``rewrite``, the reuse check also rejects incompatible
        format identifiers in an existing temperature file.
    OSError
        Input/output file access or output publication fails.

    Notes
    -----
    Target identity and provenance are extracted before checking for reusable
    outputs. Source metadata is assembled separately for each output file so
    element titles and temperature-specific fields match the file's scope.
    A reuse hit leaves output files and conversion timestamps unchanged.

    The temperature file is staged while neutron data and both companion
    files are written. It replaces its destination only after all those
    writes succeed. Companion updates are not part of the same transaction:
    a later failure can leave a companion updated while the previous
    temperature file remains intact. ACEtk parsing exceptions and exceptions
    from the shared converters and writers propagate to the caller.
    """
    # Read target identity and assemble provenance for each output file
    ace = ACEtk.ContinuousEnergyTable.from_file(str(ace_path))
    target = read_target(ace)
    sources_by_file = {
        target.filename: read_source_libraries(ace, target, with_temperature=True),
        f"{target.symbol}.h5": read_source_libraries(ace, target, for_element=True),
        f"{target.nuclide}.h5": read_source_libraries(ace, target),
    }
    # Reuse prepared outputs unless an explicit rewrite was requested
    if not rewrite and library_exists(
        output_dir,
        target.filename,
        target.nuclide,
        target.symbol,
        sources_by_file,
    ):
        if verbose:
            print(f"Skip {target.filename}: output files already exist")
        return

    # Stage neutron data and publish the temperature file after companion updates
    if verbose:
        print(f"Create {target.filename} from {ace_path.name}")
    with atomic_file(
        output_dir / target.filename, sources_by_file[target.filename]
    ) as file:
        write_temperature(file, ace, target, verbose=verbose)
        # Merge element and nuclide identities with their file-specific provenance
        write_companion(
            output_dir / f"{target.symbol}.h5",
            {"atomic-number": target.atomic_number},
            sources_by_file[f"{target.symbol}.h5"],
        )
        write_companion(
            output_dir / f"{target.nuclide}.h5",
            {
                "mass-number": target.mass_number,
                "atomic-weight-ratio": ace.atomic_weight_ratio,
                "excitation-level": target.excitation_level,
                "fissionable": ace.fission_multiplicity_block is not None,
            },
            sources_by_file[f"{target.nuclide}.h5"],
        )


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
        from conversion terminates the batch with status 1 and the input
        filename in the error message.
    OSError
        Creating the output directory or listing the input directory fails.

    Notes
    -----
    Options are read from ``sys.argv``. ``MCDC_ACE_LIB`` identifies the input
    directory and ``MCDC_LIB`` the output directory, which is created along
    with any missing parents. ``--ace-files`` selects filenames in the order
    supplied; without it, every regular file in the input directory is
    selected in sorted order, without a filename-suffix filter.

    ``--rewrite`` and ``--verbose`` are passed to each conversion. Verbose
    output replaces the progress bar. The batch stops at the first caught
    conversion error; files already completed are retained. Exceptions other
    than the caught ``ValueError`` and ``OSError`` propagate unchanged.
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
    progress = tqdm(paths, disable=args.verbose, unit="file")
    for path in progress:
        progress.set_postfix_str(path.name)
        try:
            convert_file(path, output_dir, args.rewrite, args.verbose)
        except (ValueError, OSError) as error:
            parser.exit(1, f"{path.name}: {error}\n")


if __name__ == "__main__":
    main()
