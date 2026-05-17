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
    # Ionization         : MT-534 (all subshells grouped under one reaction)

    reactions = file.create_group("electron_reactions")

    elastic_group = reactions.create_group("elastic_scattering")
    excitation_group = reactions.create_group("excitation")
    bremsstrahlung_group = reactions.create_group("bremsstrahlung")
    ionization_group = reactions.create_group("ionization")

    elastic_MT = elastic_group.create_group("MT-526")
    elastic_MT.attrs["MT"] = 526
    excitation_MT = excitation_group.create_group("MT-528")
    excitation_MT.attrs["MT"] = 528
    bremsstrahlung_MT = bremsstrahlung_group.create_group("MT-527")
    bremsstrahlung_MT.attrs["MT"] = 527
    ionization_MT = ionization_group.create_group("MT-534")
    ionization_MT.attrs["MT"] = 534

    subsh_block = ace_table.electron_subshell_block
    N_subshells = subsh_block.number_electron_subshells
    ionization_MTs = [533 + subsh_block.designator(i + 1) for i in range(N_subshells)]

    if verbose:
        print(f"  Reaction group MTs")
        print(f"    - Elastic scattering MT : [526]")
        print(f"    - Excitation MT         : [528]")
        print(f"    - Bremsstrahlung MT     : [527]")
        print(f"    - Ionization MT         : [534]")
        print(f"    - Ionization subshells  : {ionization_MTs}")

    # All electron reactions are tabulated in the laboratory frame
    for MT_group in [elastic_MT, excitation_MT, bremsstrahlung_MT, ionization_MT]:
        MT_group.create_dataset("reference_frame", data="LAB")

    # ==================================================================================
    # Cross sections
    # ==================================================================================
    # xs_energy_grid: shared electron XS energy grid (MeV) used for all reactions

    xs0_block = ace_table.electron_cross_section_block

    xs_energy = np.array(xs0_block.energies)
    dataset = reactions.create_dataset("xs_energy_grid", data=xs_energy)
    dataset.attrs["unit"] = "MeV"

    electroionisation = [
        np.array(xs0_block.electroionisation(i + 1)) for i in range(N_subshells)
    ]

    for MT_group, xs_data in [
        (elastic_MT, np.array(xs0_block.elastic)),
        (excitation_MT, np.array(xs0_block.excitation)),
        (bremsstrahlung_MT, np.array(xs0_block.bremsstrahlung)),
        (ionization_MT, np.sum(electroionisation, axis=0)),
    ]:
        xs = MT_group.create_dataset("xs", data=xs_data)
        xs.attrs["offset"] = 0
        xs.attrs["unit"] = "barns"

    # ==================================================================================
    # Elastic large-angle cross section and scattering cosine distribution
    # ==================================================================================

    elastic_xs_block = ace_table.electron_elastic_cross_section_block
    large_angle_group = elastic_MT.create_group("large_angle")

    dataset = large_angle_group.create_dataset("xs_energy", data=xs_energy)
    dataset.attrs["unit"] = "MeV"
    dataset = large_angle_group.create_dataset(
        "transport", data=np.array(elastic_xs_block.transport)
    )
    dataset.attrs["unit"] = "barns"
    dataset = large_angle_group.create_dataset(
        "total", data=np.array(elastic_xs_block.total)
    )
    dataset.attrs["unit"] = "barns"

    cosine_group = large_angle_group.create_group("scattering_cosine")
    util.load_elastic_angular_distribution(
        ace_table.electron_elastic_angular_distribution_block, cosine_group
    )

    # ==================================================================================
    # Excitation energy loss
    # ==================================================================================

    excit_block = ace_table.electron_excitation_energy_loss_block
    excit_group = excitation_MT.create_group("energy_loss")

    dataset = excit_group.create_dataset("energy", data=np.array(excit_block.energies))
    dataset.attrs["unit"] = "MeV"

    dataset = excit_group.create_dataset(
        "value", data=np.array(excit_block.excitation_energy_loss)
    )
    dataset.attrs["unit"] = "MeV"

    # ==================================================================================
    # Bremsstrahlung energy loss
    # ==================================================================================

    breml_block = ace_table.electron_energy_after_bremsstrahlung_block
    breml_energy = np.array(breml_block.energies)
    brems_group = bremsstrahlung_MT.create_group("energy_loss")

    dataset = brems_group.create_dataset("energy", data=breml_energy)
    dataset.attrs["unit"] = "MeV"

    dataset = brems_group.create_dataset(
        "value", data=breml_energy - np.array(breml_block.energy_after_bremsstrahlung)
    )
    dataset.attrs["unit"] = "MeV"

    # ==================================================================================
    # Ionization: per-subshell cross section, binding energy, and knock-on
    # electron energy distributions
    # ==================================================================================

    subshells_group = ionization_MT.create_group("subshells")

    for i, MT in enumerate(ionization_MTs):
        idx = i + 1
        subshell_group = subshells_group.create_group(f"subshell-{idx:03}")
        subshell_group.attrs["MT"] = MT

        dataset = subshell_group.create_dataset("energy_grid", data=xs_energy)
        dataset.attrs["unit"] = "MeV"

        dataset = subshell_group.create_dataset("xs", data=electroionisation[i])
        dataset.attrs["unit"] = "barns"

        dataset = subshell_group.create_dataset(
            "binding_energy", data=subsh_block.binding_energy(idx)
        )
        dataset.attrs["unit"] = "MeV"

        product_group = subshell_group.create_group("product")
        util.load_electroionization_subshell(
            ace_table.electroionisation_energy_distribution_block(idx),
            product_group,
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
