import argparse
import h5py
import numpy as np
import os

from tqdm import tqdm

import util
from util import print_error, print_note

try:
    import ACEtk
    import tools as njoy_tools

    # Sanity check: if this is not a compiled extension, it's likely the *wrong* "tools" package
    tools_path = getattr(njoy_tools, "__file__", "")
    if not tools_path or not tools_path.endswith((".so", ".pyd", ".dylib")):
        print_error(
            "Imported a Python package named 'tools' that doesn't look like the NJOY tools bindings.\n"
            f"Resolved tools at: {tools_path}\n"
            "Set PYTHONPATH=$PYTHONPATH:<ACEtk dir>/build/_deps/tools-build/python\n"
        )
except:
    print_error(
        "ACEtk is not installed\n"
        "   1. Follow installation instructions at:\n"
        "      https://github.com/njoy/ACEtk\n"
        "   2. Set\n"
        "        PYTHONPATH=$PYTHONPATH:<ACEtk dir>/build/python\n"
        "        PYTHONPATH=$PYTHONPATH:<ACEtk dir>/build/_deps/tools-build/python\n"
    )


parser = argparse.ArgumentParser(description="MC/DC data generator")
parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
args, unargs = parser.parse_known_args()
rewrite = args.rewrite
verbose = args.verbose

# Directories
output_dir = os.getenv("MCDC_LIB")
ace_dir = os.getenv("MCDC_ACELIB")

if output_dir is None:
    print_error("Environment variable $MCDC_LIB is not set")
if ace_dir is None:
    print_error("Environment variable $MCDC_ACELIB is not set")

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)
print(f"\nACE directory: {ace_dir}")
print(f"Output directory: {output_dir}\n")

# Select the files
if rewrite:
    target_files = os.listdir(ace_dir)
else:
    target_files = []
    for file_name in os.listdir(ace_dir):
        # File header
        with open(f"{ace_dir}/{file_name}", "r") as f:
            header = ACEtk.Header.from_string(f.readline())

        # Decode ACE name to MC/DC name
        Z, A, S, T = util.decode_ace_name(header.zaid)
        symbol = util.Z_TO_SYMBOL[Z]
        nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"
        mcdc_name = f"{nuclide_name}-{T}K.h5"

        if not os.path.exists(f"{output_dir}/{mcdc_name}"):
            target_files.append(file_name)

# Loop over all files
pbar = tqdm(
    target_files,
    disable=verbose,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
)

for ace_name in pbar:
    # File header
    with open(f"{ace_dir}/{ace_name}", "r") as f:
        header = ACEtk.Header.from_string(f.readline())

    # Decode ACE name to MC/DC name
    Z, A, S, T = util.decode_ace_name(header.zaid)
    symbol = util.Z_TO_SYMBOL[Z]
    nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"
    mcdc_name = f"{nuclide_name}-{T}K.h5"

    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    # Create MC/DC file
    if verbose:
        print("\n" + "=" * 80 + "\n")
        print(f"Create {mcdc_name} from {ace_name}\n")
    pbar.set_postfix_str(f"{mcdc_name[:-3]} from {ace_name}")
    file = h5py.File(f"{output_dir}/{mcdc_name}", "w")

    # ==================================================================================
    # Basic properties
    # ==================================================================================

    # Load ACE tables
    ace_table = ACEtk.ContinuousEnergyTable.from_file(f"{ace_dir}/{ace_name}")

    # ACE data source description
    header = ace_table.header
    file.attrs["source_title"] = header.title
    file.attrs["source_version"] = header.version
    file.attrs["source_date"] = header.date
    if "comments" in dir(header):
        file.attrs["source_comments"] = header.comments

    # Name and excitation level
    file.create_dataset("nuclide_name", data=nuclide_name)
    file.create_dataset("excitation_level", data=S)

    # Temperature
    temperature = file.create_dataset("temperature", data=T)
    temperature.attrs["unit"] = "K"

    # Atomic weight ratio
    atomic_weight_ratio = ace_table.atomic_weight_ratio
    file.create_dataset("atomic_weight_ratio", data=atomic_weight_ratio)

    # Fissionable?
    fissionable = ace_table.fission_multiplicity_block is not None
    file.create_dataset("fissionable", data=fissionable)

    # ==================================================================================
    # Reaction groups
    # ==================================================================================
    # Elastic scattering: MT=2
    # Capture: Reactions with zero multiplicity
    # Fission: MT=18 or MT=(19, 20, 21, and 38) if given
    # Inelastic: Non-fission reactions with non-zero multiplicity
    # Ignored: MT=(1, 3, 4, 10) and MT>117

    reactions = file.create_group("neutron_reactions")

    # ACE blocks
    nu_block = ace_table.frame_and_multiplicity_block
    rx_block = ace_table.reaction_number_block
    N_reaction = nu_block.number_reactions

    if nu_block.number_reactions != rx_block.number_reactions:
        print_error("Non-equal reaction number in reaction and multiplicity blocks")

    # The groups
    elastic_group = reactions.create_group("elastic_scattering")
    capture_group = reactions.create_group("capture")
    inelastic_group = reactions.create_group("inelastic_scattering")
    fission_group = reactions.create_group("fission")

    # MT groups
    elastic_MTs = [2]
    capture_MTs = []
    inelastic_MTs = []
    fission_MTs = []

    # Redundant MTs
    fission_chance_MTs = [19, 20, 21, 38]
    redundant_MTs = [1, 3, 4, 10]

    # Set fission MTs
    total_fission_given = rx_block.has_MT(18)
    if total_fission_given:
        fission_MTs = [18]
        # The component should not be given
        for MT in fission_chance_MTs:
            if rx_block.has_MT(MT):
                print_error("Both total fission and its components are given")
    else:
        for MT in fission_chance_MTs:
            if rx_block.has_MT(MT):
                fission_MTs.append(MT)

    # Capture and inelastic MTs
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs + elastic_MTs + fission_MTs or MT > 117:
            continue

        nu = nu_block.multiplicity(idx)

        if type(nu) != int:
            print_error(f"Non-integer multiplicity for inelastic scattering")

        if nu == 0:
            capture_MTs.append(MT)
        elif nu > 0:
            inelastic_MTs.append(MT)
        else:
            print_error(f"Negative multiplicity for MT-{MT:03}")

    # Create MTs
    for rx_group, rx_MTs in [
        (elastic_group, elastic_MTs),
        (capture_group, capture_MTs),
        (inelastic_group, inelastic_MTs),
        (fission_group, fission_MTs),
    ]:
        for MT in rx_MTs:
            MT_group = rx_group.create_group(f"MT-{MT:03}")
            MT_group.attrs["MT"] = MT

    # Report MT groups
    if verbose:
        print(f"  Reaction group MTs")
        print(f"    - Elastic scattering MTs: {elastic_MTs}")
        print(f"    - Capture MTs: {capture_MTs}")
        print(f"    - Inelastic scattering MTs: {inelastic_MTs}")
        if fissionable:
            print(f"    - Fission MT: {fission_MTs}")

    # Delete empty groups
    if not fissionable:
        del file["neutron_reactions/fission"]
    if len(inelastic_MTs) == 0:
        del file["neutron_reactions/inelastic_scattering"]

    # ==================================================================================
    # Cross-sections
    # ==================================================================================

    xs0_block = ace_table.principal_cross_section_block
    xs_block = ace_table.cross_section_block

    xs_energy = np.array(xs0_block.energies.to_list())
    xs_elastic = xs0_block.elastic
    cross_sections = xs_block.cross_sections
    offsets = xs_block.energy_index

    # Energy grid
    xs_energy = np.array(xs_energy)
    dataset = reactions.create_dataset("xs_energy_grid", data=xs_energy)
    dataset.attrs["unit"] = "MeV"

    # Elastic scattering
    xs = elastic_group.create_dataset("MT-002/xs", data=xs_elastic)
    xs.attrs["offset"] = 0
    xs.attrs["unit"] = "barns"

    # Capture, inelastic scattering, and fission
    for MTs, group in [
        (capture_MTs, capture_group),
        (inelastic_MTs, inelastic_group),
        (fission_MTs, fission_group),
    ]:
        for MT in MTs:
            idx = rx_block.index(MT)
            xs = group.create_dataset(f"MT-{MT:03}/xs", data=cross_sections(idx))
            xs.attrs["offset"] = offsets(idx) - 1
            xs.attrs["unit"] = "barns"

    # ==================================================================================
    # Reference frames and inelastic scattering multiplicities
    # ==================================================================================
    # Elastic is always in COM frame (per ACE standard)

    # Elastic scattering reference frame
    for MT in elastic_MTs:
        elastic_group.create_dataset(f"MT-{MT:03}/reference_frame", data="COM")

    # Reference frames of the others
    for MTs, group in [
        (capture_MTs, capture_group),
        (inelastic_MTs, inelastic_group),
        (fission_MTs, fission_group),
    ]:
        for MT in MTs:
            idx = rx_block.index(MT)
            reference_frame = nu_block.reference_frame(idx)
            if reference_frame == ACEtk.ReferenceFrame.Laboratory:
                reference_frame = "LAB"
            elif reference_frame == ACEtk.ReferenceFrame.CentreOfMass:
                reference_frame = "COM"
            else:
                print_error(f"Unknown reaction reference frame type for MT-{MT:03}")
            group.create_dataset(f"MT-{MT:03}/reference_frame", data=reference_frame)

    # Inelastic multiplicity
    for MT in inelastic_MTs:
        idx = rx_block.index(MT)
        nu = nu_block.multiplicity(idx)
        inelastic_group.create_dataset(f"MT-{MT:03}/multiplicity", data=nu)

    # ==================================================================================
    # Angular distributions
    # ==================================================================================

    angle_block = ace_table.angular_distribution_block

    # Elastic scattering
    angle_group = elastic_group.create_group("MT-002/angular_cosine_distribution")
    data = angle_block.angular_distribution_data(0)
    for subdata in data.distributions:
        if not isinstance(subdata, ACEtk.continuous.TabulatedAngularDistribution):
            print_error("Unsupported elastic scattering angular distribution")
    util.load_cosine_distribution(data, angle_group)

    # Inelastic scattering and fission
    for MTs, group in [
        (inelastic_MTs, inelastic_group),
        (fission_MTs, fission_group),
    ]:
        for MT in MTs:
            idx = rx_block.index(MT)
            angle_group = group.create_group(f"MT-{MT:03}/angular_cosine_distribution")
            data = angle_block.angular_distribution_data(idx)
            util.load_cosine_distribution(data, angle_group)

    # ==================================================================================
    # Energy distributions
    # ==================================================================================

    energy_block = ace_table.energy_distribution_block

    for MTs, group in [
        (inelastic_MTs, inelastic_group),
        (fission_MTs, fission_group),
    ]:
        for MT in MTs:
            idx = rx_block.index(MT)
            data = energy_block.energy_distribution_data(idx)

            if not isinstance(data, ACEtk.continuous.MultiDistributionData):
                # Probabilities
                dataset = group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability_grid", data=np.array([0.0, 30.0])
                )
                dataset.attrs["unit"] = "MeV"
                dataset = group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability", data=np.array([[1.0]])
                )

                # The distributions
                energy_group = group.create_group(f"MT-{MT:03}/energy_spectrum-1")
                util.load_energy_distribution(
                    data, energy_group, incident_grid=xs_energy
                )

            else:
                N_dist = data.number_distributions

                # ======================================================================
                # Probabilities
                # ======================================================================

                # Constant probability
                if all(
                    np.array(
                        [x.number_interpolation_regions for x in data.probabilities]
                    )
                    == 0
                ):
                    probability_grid = np.array([0.0, 30.0])
                    probability = np.zeros((1, N_dist))
                    for i in range(N_dist):
                        probability[0, i] = max(data.probability(i + 1).probabilities)

                # Histogram probability
                elif all(
                    np.array(
                        [x.number_interpolation_regions for x in data.probabilities]
                    )
                    == 1
                ) and all(np.array([x.interpolants for x in data.probabilities]) == 1):
                    probability_grid = np.array(data.probability(1).energies)
                    probability = np.zeros((len(probability_grid) - 1, N_dist))
                    for i in range(N_dist):
                        if not all(
                            probability_grid
                            == np.array(data.probability(i + 1).energies)
                        ):
                            print_error("Unsupported multi-distribution energy spetrum")
                        probability[:, i] = np.array(
                            data.probability(i + 1).probabilities[:-1]
                        )

                else:
                    print_error("Unsupported multi-distribution energy spetrum")

                dataset = group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability_grid", data=probability_grid
                )
                dataset.attrs["unit"] = "MeV"
                dataset = group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability", data=probability
                )

                # ======================================================================
                # The disributions
                # ======================================================================

                for i in range(N_dist):
                    energy_group = group.create_group(
                        f"MT-{MT:03}/energy_spectrum-{i+1}"
                    )
                    distribution = data.distribution(i + 1)
                    util.load_energy_distribution(
                        data, energy_group, incident_grid=xs_energy
                    )

    # Fissionable zone below
    if not fissionable:
        continue

    # ==================================================================================
    # Fission multiplicities and delayed neutron precursor fractions and decay rates
    # ==================================================================================

    prompt_block = ace_table.fission_multiplicity_block
    delayed_block = ace_table.delayed_fission_multiplicity_block
    dnp_block = ace_table.delayed_neutron_precursor_block

    # Prompt multiplicity
    data = prompt_block.multiplicity
    h5_group = fission_group.create_group("prompt_multiplicity")
    util.load_fission_multiplicity(data, h5_group)

    # Delayed multiplicity
    if delayed_block is not None:
        data = delayed_block.multiplicity
        h5_group = fission_group.create_group("delayed_multiplicity")
        util.load_fission_multiplicity(data, h5_group)

    # Delayed neutron precursor fractions and decay rates
    if dnp_block is not None:
        N_DNP = dnp_block.number_delayed_precursors
        fractions = np.zeros(N_DNP)
        decay_rates = np.zeros(N_DNP)

        for i in range(N_DNP):
            idx = i + 1
            data = dnp_block.precursor_group_data(idx)

            if (
                not data.number_interpolation_regions == 0
                or not len(data.probabilities[:]) == 2
                or not data.probabilities[0] == data.probabilities[1]
            ):
                print_error("Non-constant delayed neutron precursor fraction")

            fractions[i] = data.probabilities[0]
            decay_rates[i] = data.decay_constant

        precursors = fission_group.create_group("delayed_neutron_precursors")
        precursors.create_dataset("fractions", data=fractions)
        decay_rates = precursors.create_dataset("decay_rates", data=decay_rates)
        decay_rates.attrs["unit"] = "/s"

    # ==================================================================================
    # Delayed fission spectra
    # ==================================================================================

    delayed_spectrum_block = ace_table.delayed_neutron_energy_distribution_block
    if dnp_block is not None:
        N_DNP = dnp_block.number_delayed_precursors

        for i in range(N_DNP):
            idx = i + 1
            data = delayed_spectrum_block.energy_distribution_data(idx)

            if not isinstance(data, ACEtk.continuous.OutgoingEnergyDistributionData):
                print_error(f"Unsupported delayed fission neutron spectrum: {data}")

            energy_group = fission_group.create_group(
                f"delayed_neutron_precursors/energy_spectrum-{i+1}"
            )
            util.load_energy_distribution(data, energy_group, incident_grid=xs_energy)

    # ==================================================================================
    # Finalize
    # ==================================================================================

    file.close()

print("")
