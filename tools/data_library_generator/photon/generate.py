import argparse
import glob
import h5py
import numpy as np
import os
import re

from tqdm import tqdm

####

import util
from util import print_error

parser = argparse.ArgumentParser(description="MC/DC photon data generator")
parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
args, unargs = parser.parse_known_args()
rewrite = args.rewrite
verbose = args.verbose

# Directories
output_dir = os.getenv("MCDC_LIB_PHOTON")
epdl_dir = os.getenv("MCDC_EPDL_LIB")
eadl_dir = os.getenv("MCDC_EADL_LIB")
if output_dir is None:
    print_error("Environment variable $MCDC_LIB_PHOTON is not set")
if epdl_dir is None:
    print_error("Environment variable $MCDC_EPDL_LIB is not set")
if eadl_dir is None:
    # EADL supplies atomic relaxation only. Without it the library still loads;
    # photoelectric absorption simply emits no fluorescence.
    eadl_dir = ""

os.makedirs(output_dir, exist_ok=True)
print(f"\nEPDL dir    : {epdl_dir}")
print(f"EADL dir    : {eadl_dir or '(none -- no atomic relaxation)'}")
print(f"Output dir  : {output_dir}\n")

_ZA_PATTERN = re.compile(r"ZA(\d{3})000", re.IGNORECASE)


def epdl_path(Z):
    return os.path.join(epdl_dir, f"EPDL.ZA{Z:03d}000.endf")


def eadl_path(Z):
    return os.path.join(eadl_dir, f"EADL.ZA{Z:03d}000.endf")


# Select target entries from whatever EPDL ships
target_entries = []
for path in sorted(glob.glob(os.path.join(epdl_dir, "*.endf"))):
    match = _ZA_PATTERN.search(os.path.basename(path))
    if match is None:
        continue

    Z = int(match.group(1))
    if Z not in util.Z_TO_SYMBOL:
        continue

    symbol = util.Z_TO_SYMBOL[Z]
    mcdc_name = f"{symbol}.h5"

    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    target_entries.append((Z, symbol, mcdc_name, path))

if not target_entries:
    print("Nothing to convert. Pass --rewrite to regenerate existing elements.\n")

# Loop over all elements
pbar = tqdm(
    target_entries,
    disable=verbose,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
)
for Z, symbol, mcdc_name, path in pbar:
    pbar.set_postfix_str(symbol)
    if verbose:
        print(f"Converting {symbol} (Z={Z})")

    sections = util.read_sections(path)

    if (23, util.MT_TOTAL) not in sections:
        print_error(f"{symbol}: EPDL file has no MF=23/MT={util.MT_TOTAL} total")

    # ==================================================================================
    # The shared energy grid
    # ==================================================================================
    # EPDL tabulates the total cross section on the unionized grid that every
    # other photon channel is a subset of, so MT-501's abscissa IS the library's
    # xs_energy_grid -- no union needs constructing, and the duplicated
    # absorption-edge energies that represent each discontinuity are preserved
    # exactly as EPDL writes them.

    energy_grid, _, _, _ = util.read_tab1(sections[(23, util.MT_TOTAL)])

    _, AWR = util.read_head_za_awr(sections[(23, util.MT_TOTAL)])

    with h5py.File(f"{output_dir}/{mcdc_name}", "w") as file:
        # ==============================================================================
        # Element identity
        # ==============================================================================
        # atomic_number and atomic_weight_ratio are read by
        # Element._compile_into_simulation for every particle type, so they are
        # not optional even in a photon-only library.

        file.create_dataset("element_symbol", data=symbol.encode())
        file.create_dataset("element_name", data=symbol.encode())
        file.create_dataset("atomic_number", data=np.int64(Z))
        file.create_dataset("atomic_weight_ratio", data=np.float64(AWR))

        file.attrs["source_title"] = "EPDL -- Evaluated Photon Data Library"
        file.attrs["source_comments"] = (
            "Photon interaction cross sections from ENDF MF=23, form factors and "
            "scattering functions from MF=27, atomic relaxation from EADL MF=28."
        )

        reactions = file.create_group("photon_reactions")
        util.write_energy(reactions, "xs_energy_grid", energy_grid)

        # EPDL's own total, on the grid it defines. Element.set_photon_data
        # derives the total by summing the channels and does not read this, but
        # the libraries already in circulation carry it.
        _, total_xs, _, _ = util.read_tab1(sections[(23, util.MT_TOTAL)])
        total_group = reactions.create_group("total")
        dataset = total_group.create_dataset("xs", data=total_xs)
        dataset.attrs["unit"] = "barns"

        # ==============================================================================
        # Channel cross sections
        # ==============================================================================

        for group_name, MTs in util.REACTION_GROUPS:
            group = reactions.create_group(group_name)

            for MT in MTs:
                if (23, MT) not in sections:
                    continue

                E_table, xs_table, bounds, laws = util.read_tab1(sections[(23, MT)])
                offset, xs = util.project_onto_grid(
                    energy_grid, E_table, xs_table, laws, bounds
                )

                mt_group = util.create_mt_group(group, MT)
                util.write_xs(mt_group, xs, offset)
                util.write_reference_frame(mt_group)

            # Pair production also carries its MT-516 sum, as an unnumbered
            # subgroup rather than a third MT group.
            if group_name == "pair_production" and (23, util.MT_PAIR_TOTAL) in sections:
                E_table, xs_table, bounds, laws = util.read_tab1(
                    sections[(23, util.MT_PAIR_TOTAL)]
                )
                offset, xs = util.project_onto_grid(
                    energy_grid, E_table, xs_table, laws, bounds
                )
                total_group = group.create_group("total")
                util.write_xs(total_group, xs, offset)

        # ==============================================================================
        # Coherent scattering: form factor and anomalous scattering
        # ==============================================================================

        coherent = reactions["coherent_scattering"]
        if f"MT{util.REACTION_GROUPS[0][1][0]}" in coherent:
            mt_group = coherent[f"MT{util.REACTION_GROUPS[0][1][0]}"]

            if (27, util.MF27_FORM_FACTOR) in sections:
                q, F, _, _ = util.read_tab1(sections[(27, util.MF27_FORM_FACTOR)])
                form_factor = mt_group.create_group("form_factor")
                util.write_momentum_transfer(form_factor, q)
                form_factor.create_dataset("value", data=F)

        # ==============================================================================
        # Incoherent scattering: the binding-correction scattering function
        # ==============================================================================

        incoherent = reactions["incoherent_scattering"]
        if f"MT{util.REACTION_GROUPS[1][1][0]}" in incoherent:
            mt_group = incoherent[f"MT{util.REACTION_GROUPS[1][1][0]}"]
            if (27, util.MF27_SCATTERING_FUNCTION) in sections:
                q, S, _, _ = util.read_tab1(
                    sections[(27, util.MF27_SCATTERING_FUNCTION)]
                )
                scattering_function = mt_group.create_group("scattering_function")
                util.write_momentum_transfer(scattering_function, q)
                scattering_function.create_dataset("value", data=S)

        # ==============================================================================
        # Photoelectric subshells
        # ==============================================================================
        # Each subshell keeps its own energy grid, because its cross section
        # begins at its own binding energy and EPDL tabulates it there.

        subshell_MTs = sorted(
            MT
            for (MF, MT) in sections
            if MF == 23 and util.MT_SUBSHELL_FIRST <= MT <= util.MT_SUBSHELL_LAST
        )

        if subshell_MTs:
            photoelectric = reactions["photoelectric"]
            mt_group = photoelectric[f"MT{util.REACTION_GROUPS[2][1][0]}"]
            subshells = mt_group.create_group("subshells")

            for MT in subshell_MTs:
                E_table, xs_table, bounds, laws = util.read_tab1(sections[(23, MT)])

                subshell = util.create_mt_group(subshells, MT, hyphenated=True)

                # Subshells are projected onto the shared grid as well, so that
                # the one grid serves every photoelectric table. The binding
                # energy, where the cross section opens, is kept from the native
                # table because projecting cannot recover it.
                _, subshell_xs = util.project_onto_grid(
                    energy_grid, E_table, xs_table, laws, bounds
                )
                util.write_energy(subshell, "energy_grid", energy_grid)
                dataset = subshell.create_dataset("xs", data=subshell_xs)
                dataset.attrs["unit"] = "barns"

                util.write_energy(subshell, "binding_energy", np.float64(E_table[0]))
                subshell.attrs["subshell_designator"] = np.int64(
                    util.subshell_designator(MT)
                )

        # ==============================================================================
        # Atomic relaxation
        # ==============================================================================
        # EADL MF=28 gives, per subshell, the binding energy, the electron
        # occupancy, and a transition table. secondary_designator distinguishes
        # the two de-excitation channels: zero is radiative, so a characteristic
        # X-ray is emitted; non-zero is non-radiative (Auger).

        if eadl_dir and os.path.isfile(eadl_path(Z)):
            eadl_sections = util.read_sections(eadl_path(Z))

            if (28, util.MF28_RELAXATION) in eadl_sections:
                subshell_data = util.read_relaxation(
                    eadl_sections[(28, util.MF28_RELAXATION)]
                )

                relaxation = file.create_group("atomic_relaxation")
                relaxation.create_dataset(
                    "n_subshells", data=np.int64(len(subshell_data))
                )
                subshell_group = relaxation.create_group("subshells")

                for designator in sorted(subshell_data):
                    entry = subshell_data[designator]
                    MT = util.subshell_mt(designator)

                    group = subshell_group.create_group(f"MT-{MT}")
                    util.write_energy(
                        group, "binding_energy", np.float64(entry["binding_energy"])
                    )
                    group.create_dataset("designator", data=np.int64(designator))
                    group.create_dataset(
                        "n_electrons", data=np.float64(entry["n_electrons"])
                    )

                    if len(entry["energy"]) == 0:
                        # Outermost subshells have nothing above them to fill the
                        # vacancy, so they have no transitions at all.
                        continue

                    transitions = group.create_group("transitions")
                    util.write_energy(transitions, "energy", entry["energy"])
                    transitions.create_dataset("probability", data=entry["probability"])
                    transitions.create_dataset(
                        "origin_designator", data=entry["origin_designator"]
                    )
                    transitions.create_dataset(
                        "secondary_designator", data=entry["secondary_designator"]
                    )

print("\nDone.\n")
