import ACEtk
import argparse
import h5py
import numpy as np
import os

from tqdm import tqdm

####

import util_electron
from util_electron import print_error, print_note

parser = argparse.ArgumentParser(description="MC/DC electron data generator")
parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
args, unargs = parser.parse_known_args()
rewrite = args.rewrite
verbose = args.verbose

# Directories
output_dir = os.getenv("MCDC_LIB_ELECTRON")
ace_file = os.getenv("MCDC_ACELIB_ELECTRON")
xsdir_file = os.getenv("MCDC_ACELIB_ELECTRON_XSDIR")

if output_dir is None:
    print_error("Environment variable $MCDC_LIB_ELECTRON is not set")
if ace_file is None:
    print_error("Environment variable $MCDC_ACELIB_ELECTRON is not set")
if xsdir_file is None:
    print_error("Environment variable $MCDC_ACELIB_ELECTRON_XSDIR is not set")

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)
print(f"\nACE file    : {ace_file}")
print(f"Xsdir file  : {xsdir_file}")
print(f"Output dir  : {output_dir}\n")

# Read xsdir and collect EPR entries
xsdir = ACEtk.Xsdir.from_file(xsdir_file)
target_entries = []
for entry in xsdir.entries:
    if not entry.zaid.endswith(".14p"):
        continue

    Z = util_electron.decode_epr_zaid(entry.zaid)
    symbol = util_electron.Z_TO_SYMBOL[Z]
    mcdc_name = f"{symbol}.h5"

    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    target_entries.append((entry, Z, symbol, mcdc_name))

# Loop over all elements
pbar = tqdm(
    target_entries,
    disable=verbose,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
)
for entry, Z, symbol, mcdc_name in pbar:

    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    if verbose:
        print("\n" + "=" * 80 + "\n")
        print(f"Create {mcdc_name} from {entry.zaid}\n")
    pbar.set_postfix_str(f"{mcdc_name}")

    # Load ACE table
    ace_table = ACEtk.PhotoatomicTable.from_file(ace_file, entry.file_start_line)

    # Create MC/DC file
    file = h5py.File(f"{output_dir}/{mcdc_name}", "w")

    # ==================================================================================
    # Basic properties
    # ==================================================================================

    header = ace_table.header
    file.attrs["source_title"] = header.title
    file.attrs["source_date"] = header.date

    file.create_dataset("element_symbol", data=symbol)
    file.create_dataset("atomic_number", data=Z)
    file.create_dataset("atomic_weight_ratio", data=ace_table.atomic_weight_ratio)

    # ==================================================================================
    # Principal cross sections (energy grid + xs for all reactions)
    # ==================================================================================

    xs0_block = ace_table.principal_cross_section_block

    energy_grid = np.array(xs0_block.energies)
    dataset = file.create_dataset("energy_grid", data=energy_grid)
    dataset.attrs["unit"] = "MeV"

    xs_group = file.create_group("cross_sections")

    dataset = xs_group.create_dataset("elastic", data=np.array(xs0_block.elastic))
    dataset.attrs["unit"] = "barns"

    dataset = xs_group.create_dataset("bremsstrahlung", data=np.array(xs0_block.bremsstrahlung))
    dataset.attrs["unit"] = "barns"

    dataset = xs_group.create_dataset("excitation", data=np.array(xs0_block.excitation))
    dataset.attrs["unit"] = "barns"

    dataset = xs_group.create_dataset("electroionization", data=np.array(xs0_block.electroionization))
    dataset.attrs["unit"] = "barns"

    dataset = xs_group.create_dataset("total", data=np.array(xs0_block.total))
    dataset.attrs["unit"] = "barns"

    # ==================================================================================
    # Elastic angular distribution
    # ==================================================================================

    angle_group = file.create_group("elastic_angular_distribution")
    util_electron.load_elastic_angular_distribution(
        ace_table.elastic_angular_distribution_block, angle_group
    )

    # ==================================================================================
    # Excitation energy loss
    # ==================================================================================

    excit_block = ace_table.excitation_block
    excit_group = file.create_group("excitation")

    energies = np.array(excit_block.energies)
    dataset = excit_group.create_dataset("energy", data=energies)
    dataset.attrs["unit"] = "MeV"

    dataset = excit_group.create_dataset("energy_loss", data=np.array(excit_block.energy_loss))
    dataset.attrs["unit"] = "MeV"

    # ==================================================================================
    # Bremsstrahlung energy distribution
    # ==================================================================================

    brems_group = file.create_group("bremsstrahlung")
    util_electron.load_bremsstrahlung(ace_table.bremsstrahlung_block, brems_group)

    # ==================================================================================
    # Electroionization: per-subshell cross sections and energy distributions
    # ==================================================================================

    subshell_block = ace_table.electron_subshell_block
    N_subshells = subshell_block.number_of_subshells

    ioniz_group = file.create_group("electroionization")

    for i in range(N_subshells):
        idx = i + 1
        shell_group = ioniz_group.create_group(f"subshell_{idx}")

        # Binding energy and cross section
        binding_energy = subshell_block.binding_energy(idx)
        dataset = shell_group.create_dataset("binding_energy", data=binding_energy)
        dataset.attrs["unit"] = "MeV"

        xs = np.array(subshell_block.cross_section(idx))
        dataset = shell_group.create_dataset("xs", data=xs)
        dataset.attrs["unit"] = "barns"

        # Knock-on electron energy distribution
        energy_dist_group = shell_group.create_group("energy_distribution")
        util_electron.load_electroionization_subshell(
            subshell_block.energy_distribution(idx), energy_dist_group
        )

    if verbose:
        print(f"  Z={Z} ({symbol}): {N_subshells} subshells")

    # ==================================================================================
    # Atomic relaxation (subshell transition data)
    # ==================================================================================

    relaxation_block = ace_table.subshell_transition_data_block
    relaxation_group = file.create_group("atomic_relaxation")

    for i in range(N_subshells):
        idx = i + 1
        data = relaxation_block.transition_data(idx)
        shell_group = relaxation_group.create_group(f"subshell_{idx}")

        N_transitions = data.number_of_transitions
        shell_group.create_dataset("number_of_transitions", data=N_transitions)

        if N_transitions > 0:
            primary_shells = []
            secondary_shells = []
            energies = []
            probabilities = []
            for j in range(N_transitions):
                jdx = j + 1
                primary_shells.append(data.primary_subshell(jdx))
                secondary_shells.append(data.secondary_subshell(jdx))
                energies.append(data.energy(jdx))
                probabilities.append(data.probability(jdx))

            shell_group.create_dataset("primary_subshell", data=np.array(primary_shells))
            shell_group.create_dataset("secondary_subshell", data=np.array(secondary_shells))
            dataset = shell_group.create_dataset("energy", data=np.array(energies))
            dataset.attrs["unit"] = "MeV"
            shell_group.create_dataset("probability", data=np.array(probabilities))

    # ==================================================================================
    # Finalize
    # ==================================================================================

    file.close()

print("")
