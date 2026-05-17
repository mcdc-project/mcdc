import ACEtk
import argparse
import h5py
import numpy as np
import os

from tqdm import tqdm

####

import util
from util import print_error

parser = argparse.ArgumentParser(description="MC/DC electron data generator")
parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
args, unargs = parser.parse_known_args()
rewrite = args.rewrite
verbose = args.verbose

# Directories
output_dir = os.getenv("MCDC_LIB_ELECTRON")
ace_file = os.getenv("MCDC_ACELIB_ELECTRON")
if output_dir is None:
    print_error("Environment variable $MCDC_LIB_ELECTRON is not set")
if ace_file is None:
    print_error("Environment variable $MCDC_ACELIB_ELECTRON is not set")
# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)
print(f"\nACE file    : {ace_file}")
print(f"Output dir  : {output_dir}\n")

# Load all tables from the concatenated EPRDATA14 file
print("Loading EPRDATA14 tables...")
all_tables = ACEtk.PhotoatomicTable.from_concatenated_file(ace_file)
table_map = {t.zaid: t for t in all_tables}
print(f"Loaded {len(table_map)} tables\n")

# Select target entries
target_entries = []
for zaid in table_map:
    if not zaid.endswith(".14p"):
        continue

    Z = util.decode_epr_zaid(zaid)
    symbol = util.Z_TO_SYMBOL[Z]
    mcdc_name = f"{symbol}.h5"

    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    target_entries.append((zaid, Z, symbol, mcdc_name))

# Loop over all elements
pbar = tqdm(
    target_entries,
    disable=verbose,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
)
for zaid, Z, symbol, mcdc_name in pbar:

    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    if verbose:
        print("\n" + "=" * 80 + "\n")
        print(f"Create {mcdc_name} from {zaid}\n")
    pbar.set_postfix_str(f"{mcdc_name}")

    # Load ACE table
    ace_table = table_map[zaid]

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
    # Reaction groups
    # ==================================================================================
    # Elastic scattering : MT-526
    # Excitation         : MT-528
    # Bremsstrahlung     : MT-527
    # Ionization  : MT-534 (K), MT-535 (L1), MT-536 (L2), ...

    reactions = file.create_group("electron_reactions")

    elastic_group = reactions.create_group("elastic_scattering")
    excitation_group = reactions.create_group("excitation")
    bremsstrahlung_group = reactions.create_group("bremsstrahlung")
    ionization_group = reactions.create_group("ionization")

    elastic_group.create_group("MT-526").attrs["MT"] = 526
    excitation_group.create_group("MT-528").attrs["MT"] = 528
    bremsstrahlung_group.create_group("MT-527").attrs["MT"] = 527

    subsh_block = ace_table.electron_subshell_block
    N_subshells = subsh_block.number_electron_subshells
    ionization_MTs = [533 + subsh_block.designator(i + 1) for i in range(N_subshells)]

    for MT in ionization_MTs:
        ionization_group.create_group(f"MT-{MT:03}").attrs["MT"] = MT

    if verbose:
        print(f"  Reaction group MTs")
        print(f"    - Elastic scattering MT : [526]")
        print(f"    - Excitation MT         : [528]")
        print(f"    - Bremsstrahlung MT     : [527]")
        print(f"    - Ionization MTs : {ionization_MTs}")

    # ==================================================================================
    # Cross sections
    # ==================================================================================

    xs0_block = ace_table.electron_cross_section_block

    xs_energy = np.array(xs0_block.energies)
    dataset = reactions.create_dataset("xs_energy_grid", data=xs_energy)
    dataset.attrs["unit"] = "MeV"

    xs = elastic_group.create_dataset("MT-526/xs", data=np.array(xs0_block.elastic))
    xs.attrs["unit"] = "barns"

    xs = excitation_group.create_dataset(
        "MT-528/xs", data=np.array(xs0_block.excitation)
    )
    xs.attrs["unit"] = "barns"

    xs = bremsstrahlung_group.create_dataset(
        "MT-527/xs", data=np.array(xs0_block.bremsstrahlung)
    )
    xs.attrs["unit"] = "barns"

    for i, MT in enumerate(ionization_MTs):
        xs = ionization_group.create_dataset(
            f"MT-{MT:03}/xs", data=np.array(xs0_block.electroionisation(i + 1))
        )
        xs.attrs["unit"] = "barns"

    # ==================================================================================
    # Elastic angular distribution
    # ==================================================================================

    angle_group = elastic_group.create_group("MT-526/angular_cosine_distribution")
    util.load_elastic_angular_distribution(
        ace_table.electron_elastic_angular_distribution_block, angle_group
    )

    # ==================================================================================
    # Excitation energy loss
    # ==================================================================================

    excit_block = ace_table.electron_excitation_energy_loss_block
    excit_group = excitation_group.create_group("MT-528/energy_loss")

    dataset = excit_group.create_dataset("energy", data=np.array(excit_block.energies))
    dataset.attrs["unit"] = "MeV"

    dataset = excit_group.create_dataset(
        "excitation_energy_loss", data=np.array(excit_block.excitation_energy_loss)
    )
    dataset.attrs["unit"] = "MeV"

    # ==================================================================================
    # Bremsstrahlung energy distribution
    # ==================================================================================

    brems_group = bremsstrahlung_group.create_group("MT-527/energy_distribution")
    util.load_bremsstrahlung(
        ace_table.bremsstrahlung_energy_distribution_block, brems_group
    )

    # ==================================================================================
    # Ionization: binding energy and knock-on electron energy distributions
    # ==================================================================================

    for i, MT in enumerate(ionization_MTs):
        idx = i + 1
        MT_group = ionization_group[f"MT-{MT:03}"]

        dataset = MT_group.create_dataset(
            "binding_energy", data=subsh_block.binding_energy(idx)
        )
        dataset.attrs["unit"] = "MeV"

        energy_dist_group = MT_group.create_group("energy_distribution")
        util.load_electroionization_subshell(
            ace_table.electroionisation_energy_distribution_block(idx),
            energy_dist_group,
        )

    # ==================================================================================
    # Atomic relaxation (subshell transition data)
    # ==================================================================================

    relax_block = ace_table.subshell_transition_data_block
    relaxation_group = file.create_group("atomic_relaxation")

    for i, MT in enumerate(ionization_MTs):
        idx = i + 1
        td = relax_block.transition_data(idx)
        MT_group = relaxation_group.create_group(f"MT-{MT:03}")
        MT_group.attrs["MT"] = MT

        N_transitions = td.number_transitions
        MT_group.create_dataset("number_of_transitions", data=N_transitions)

        if N_transitions > 0:
            primary_designators = []
            secondary_designators = []
            energies = []
            probabilities = []
            for j in range(N_transitions):
                jdx = j + 1
                t = td.transition(jdx)
                primary_designators.append(td.primary_designator(jdx))
                secondary_designators.append(td.secondary_designator(jdx))
                energies.append(td.energy(jdx))
                probabilities.append(td.probability(jdx))

            MT_group.create_dataset(
                "primary_designator", data=np.array(primary_designators)
            )
            MT_group.create_dataset(
                "secondary_designator", data=np.array(secondary_designators)
            )
            dataset = MT_group.create_dataset("energy", data=np.array(energies))
            dataset.attrs["unit"] = "MeV"
            MT_group.create_dataset("probability", data=np.array(probabilities))

    # ==================================================================================
    # Finalize
    # ==================================================================================

    file.close()

print("")