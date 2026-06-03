# The majority of this script was written by Anthropic's Claude

"""
proton_ace_to_hdf5.py  —  Convert proton ACE files (TENDL etc.) to HDF5 for MC/DC

Usage
-----
    python proton_ace_to_hdf5.py
    python proton_ace_to_hdf5.py --ace_dir /path/to/ace --output_dir /path/to/hdf5 [--pstar_dir /path/to/pstar]
    python proton_ace_to_hdf5.py ...  --rewrite   # overwrite existing files
    python proton_ace_to_hdf5.py ...  --verbose   # per-reaction detail
    

Environment variable fallbacks:  $MCDC_ACELIB, $MCDC_LIB, $PSTAR_LIB

HDF5 layout
-----------
<nuclide>-<T>K.h5
  attrs: source_title, source_version, source_date
  nuclide_name, excitation_level, temperature (K),
  atomic_number, atomic_weight_ratio, fissionable

  stopping_power/           (if PSTAR data available)
    energy (MeV), total_stopping_power (MeV cm2/g)

  proton_reactions/
    xs_energy_grid (MeV)
    elastic_scattering/MT-002/
      xs (barns, offset=0), Q-value (MeV), reference_frame,
      angular_cosine_distribution/
    capture/MT-{NNN}/
      xs (barns), Q-value (MeV), reference_frame
    nonelastic_reaction/MT-{NNN}/
      xs (barns), Q-value (MeV), reference_frame, multiplicity
      angular_cosine_distribution/
      energy_spectrum-{k}/  (law attr; kalbach-mann: energy, offset,
                              energy_out, pdf, cdf, precompound_factor, angular_slope)
    fission/  (only if fissionable)

  secondary_particles/ZAP_{zap}/MT-{NNN}/
    attrs: ZAP, particle_name, MT, multiplicity, reference_frame
    production_xs (barns, offset)
    kalbach_mann/  (energy, offset, energy_out, pdf, cdf,
                    precompound_factor, angular_slope)

ZAP identity:  1=n, 1001=p, 1002=d, 1003=t, 2003=He3, 2004=alpha, 0=gamma

TabulatedKalbachMannDistribution properties used (from ACEtk):
    outgoing_energies, pdf, cdf,
    precompound_fraction_values, angular_distribution_slope_values
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm
import ACEtk

# -- Constants -----------------------------------------------------------------

ZAP_NAMES = {
    0: "photon",
    1: "neutron",
    1001: "proton",
    1002: "deuteron",
    1003: "triton",
    2003: "He3",
    2004: "alpha",
}

Z_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
}

# Redundant sum MTs that must not be double-counted
REDUNDANT_MTS = [1, 3, 4, 10, 101, 103, 104, 105, 106, 107]
FISSION_CHANCE_MTS = [19, 20, 21, 38]


# -- Utility -------------------------------------------------------------------


def print_error(msg):
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def print_note(msg):
    print(f"  [note] {msg}")


def decode_ace_zaid(zaid):
    """Return (Z, A, S, T=0) from an ACE ZAID string."""
    za = int(zaid.strip().split(".")[0])
    S = 0
    if za >= 600000:
        S = (za % 1000) // 400
        za = za - S * 400
    return za // 1000, za % 1000, S, 0


def load_pstar_file(filepath):
    """
    Load a two-column PSTAR file (Energy MeV, Stopping power MeV cm2/g).
    Returns (energies, stopping_powers) as float64 arrays.
    """
    energies, sps = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                energies.append(float(parts[0]))
                sps.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(energies), np.array(sps)


# -- Distribution writers ------------------------------------------------------


def load_cosine_distribution(data, h5_group):
    """
    Write a tabulated angular distribution into h5_group.
    Returns False if the distribution is embedded in a Kalbach-Mann block
    (DistributionGivenElsewhere), True otherwise.
    """
    if isinstance(data, ACEtk.continuous.DistributionGivenElsewhere):
        h5_group.attrs["type"] = "given_in_energy_distribution"
        return False

    h5_group.attrs["type"] = "tabulated"
    h5_group.attrs["unit"] = "MeV"
    h5_group.create_dataset("incident_energies", data=np.array(data.incident_energies))

    for i, subdist in enumerate(data.distributions):
        eg = h5_group.create_group(f"E_in_{i + 1}")
        if isinstance(subdist, ACEtk.continuous.TabulatedAngularDistribution):
            eg.attrs["type"] = "tabulated"
            eg.create_dataset("cosines", data=np.array(subdist.cosines))
            eg.create_dataset("pdf", data=np.array(subdist.pdf))
            eg.create_dataset("cdf", data=np.array(subdist.cdf))
        else:
            eg.attrs["type"] = "isotropic"

    return True


def _write_kalbach_mann(km_data, h5_group):
    """
    Write a KalbachMannDistributionData into h5_group as flat arrays.
    offset[i] gives the starting index in the flat arrays for incident energy i.
    """
    h5_group.attrs["type"] = "kalbach-mann"

    NE = km_data.number_incident_energies
    h5_group.create_dataset("energy", data=np.array(km_data.incident_energies)).attrs[
        "unit"
    ] = "MeV"

    offset, energy_out, pdf, cdf, r_vals, a_vals = [], [], [], [], [], []
    for i in range(1, NE + 1):
        dist = km_data.distribution(i)
        offset.append(len(energy_out))
        energy_out.extend(dist.outgoing_energies)
        pdf.extend(dist.pdf)
        cdf.extend(dist.cdf)
        r_vals.extend(dist.precompound_fraction_values)
        a_vals.extend(dist.angular_distribution_slope_values)

    h5_group.create_dataset("offset", data=np.array(offset, dtype=np.int32))
    h5_group.create_dataset("energy_out", data=np.array(energy_out)).attrs["unit"] = (
        "MeV"
    )
    h5_group.create_dataset("pdf", data=np.array(pdf))
    h5_group.create_dataset("cdf", data=np.array(cdf))
    h5_group.create_dataset("precompound_factor", data=np.array(r_vals))
    h5_group.create_dataset("angular_slope", data=np.array(a_vals))


def load_energy_distribution(data, h5_group):
    """Write a primary-particle outgoing energy distribution into h5_group."""
    if isinstance(data, ACEtk.continuous.KalbachMannDistributionData):
        h5_group.attrs["law"] = 44
        _write_kalbach_mann(data, h5_group)

    elif isinstance(data, ACEtk.continuous.OutgoingEnergyDistributionData):
        h5_group.attrs["law"] = 4
        h5_group.create_dataset(
            "incident_energies", data=np.array(data.incident_energies)
        )
        for k, dist in enumerate(data.distributions):
            eg = h5_group.create_group(f"E_in_{k + 1}")
            eg.create_dataset(
                "outgoing_energies", data=np.array(dist.outgoing_energies)
            )
            eg.create_dataset("pdf", data=np.array(dist.pdf))
            eg.create_dataset("cdf", data=np.array(dist.cdf))

    elif isinstance(data, ACEtk.continuous.LevelScatteringData):
        h5_group.attrs["law"] = 3
        h5_group.create_dataset("C1", data=data.C1)
        h5_group.create_dataset("C2", data=data.C2)

    elif isinstance(data, ACEtk.continuous.EquiprobableOutgoingEnergyBins):
        h5_group.attrs["law"] = 1
        h5_group.create_dataset(
            "incident_energies", data=np.array(data.incident_energies)
        )
        for k, dist in enumerate(data.distributions):
            h5_group.create_group(f"E_in_{k + 1}").create_dataset(
                "energies", data=np.array(dist.energies)
            )

    else:
        h5_group.attrs["law"] = -1
        h5_group.attrs["type_name"] = type(data).__name__
        try:
            h5_group.create_dataset("xss_array", data=np.array(data.xss_array))
        except Exception:
            pass


def load_fission_multiplicity(data, h5_group):
    if isinstance(data, ACEtk.continuous.TabulatedFissionMultiplicity):
        h5_group.attrs["type"] = "tabulated"
        h5_group.create_dataset("energies", data=np.array(data.energies))
        h5_group.create_dataset("multiplicities", data=np.array(data.multiplicities))
    elif isinstance(data, ACEtk.continuous.PolynomialFissionMultiplicity):
        h5_group.attrs["type"] = "polynomial"
        h5_group.create_dataset("coefficients", data=np.array(data.coefficients))
    else:
        h5_group.attrs["type"] = "unknown"
        h5_group.attrs["type_name"] = type(data).__name__


# -- Secondary particles -------------------------------------------------------


def load_secondary_particles(ace_table, file, verbose=False):
    n_types = ace_table.number_secondary_particle_types
    if n_types == 0:
        return

    type_block = ace_table.secondary_particle_type_block
    info_block = ace_table.secondary_particle_information_block
    rx_block = ace_table.secondary_particle_reaction_number_block
    tyr_block = ace_table.secondary_particle_frame_and_multiplicity_block
    xs_block = ace_table.secondary_particle_production_cross_section_block
    edy_block = ace_table.secondary_particle_energy_distribution_block

    has_ang = False
    try:
        ang_block = ace_table.secondary_particle_angular_distribution_block
        has_ang = True
    except Exception:
        pass

    sec_group = file.create_group("secondary_particles")

    pi_method = next(
        (
            c
            for c in ["particle_identifier", "ZAP", "type", "particle_type"]
            if hasattr(type_block, c)
        ),
        None,
    )
    if pi_method is None:
        raise AttributeError(
            f"Cannot find particle identifier on {type(type_block).__name__}. "
            f"Available: {[x for x in dir(type_block) if not x.startswith('_')]}"
        )

    for i in range(1, n_types + 1):
        zap = getattr(type_block, pi_method)(i)
        name = ZAP_NAMES.get(zap, f"ZAP_{zap}")
        n_rx = int(info_block.number_reactions[i - 1])

        if verbose:
            print(f"    Secondary type {i}: ZAP={zap} ({name}), {n_rx} reactions")

        zap_group = sec_group.create_group(f"ZAP_{zap}")
        zap_group.attrs["ZAP"] = zap
        zap_group.attrs["particle_name"] = name

        rx_i = rx_block(i)
        tyr_i = tyr_block(i)
        xs_i = xs_block(i)
        edy_i = edy_block(i)
        ang_i = ang_block(i) if has_ang else None

        xs_method = next(
            (c for c in ["cross_sections", "cross_section", "xs"] if hasattr(xs_i, c)),
            None,
        )
        off_method = next(
            (
                c
                for c in ["energy_index", "offset", "locator", "index"]
                if hasattr(xs_i, c)
            ),
            None,
        )
        edy_method = next(
            (
                c
                for c in [
                    "energy_distribution_data",
                    "distribution_data",
                    "distribution",
                ]
                if hasattr(edy_i, c)
            ),
            None,
        )

        for j in range(1, n_rx + 1):
            MT = rx_i.MT(j)
            nu_raw = tyr_i.multiplicity(j)
            nu = nu_raw - 100 if nu_raw >= 100 else nu_raw
            rf_raw = tyr_i.reference_frame(j)
            rf = (
                "LAB"
                if rf_raw == ACEtk.ReferenceFrame.Laboratory
                else (
                    "COM"
                    if rf_raw == ACEtk.ReferenceFrame.CentreOfMass
                    else str(rf_raw)
                )
            )

            mt = zap_group.create_group(f"MT-{MT:03}")
            mt.attrs["MT"] = MT
            mt.attrs["multiplicity"] = nu
            mt.attrs["reference_frame"] = rf

            if verbose:
                print(f"      MT={MT:03}  nu_raw={nu_raw}  nu={nu}  frame={rf}")

            # Production cross section
            empty_xs = np.zeros(0, dtype=float)
            if xs_method and off_method:
                try:
                    ds = mt.create_dataset(
                        "production_xs", data=np.array(getattr(xs_i, xs_method)(j))
                    )
                    ds.attrs["offset"] = int(getattr(xs_i, off_method)(j)) - 1
                    ds.attrs["unit"] = "barns"
                except Exception as exc:
                    ds = mt.create_dataset("production_xs", data=empty_xs)
                    ds.attrs["offset"] = 0
                    ds.attrs["unit"] = "barns"
                    if verbose:
                        print(f"        [warn] production xs: {exc}")
            else:
                ds = mt.create_dataset("production_xs", data=empty_xs)
                ds.attrs["offset"] = 0
                ds.attrs["unit"] = "barns"
                if verbose:
                    print(
                        f"        [warn] xs methods not found: "
                        f"{[x for x in dir(xs_i) if not x.startswith('_')]}"
                    )

            # Kalbach-Mann energy-angle distribution
            if edy_method:
                try:
                    _write_kalbach_mann(
                        getattr(edy_i, edy_method)(j), mt.create_group("kalbach_mann")
                    )
                except Exception as exc:
                    if verbose:
                        print(f"        [warn] energy dist: {exc}")
            elif verbose:
                print(
                    f"        [warn] edy method not found: "
                    f"{[x for x in dir(edy_i) if not x.startswith('_')]}"
                )

            if ang_i is not None:
                try:
                    load_cosine_distribution(
                        ang_i.angular_distribution_data(j),
                        mt.create_group("angular_cosine_distribution"),
                    )
                except Exception:
                    pass


# -- Per-file processing -------------------------------------------------------


def process_ace_file(ace_path, output_dir, pstar_dir=None, verbose=False):
    with open(ace_path) as f:
        header = ACEtk.Header.from_string(f.readline())

    Z, A, S, _ = decode_ace_zaid(header.zaid)
    symbol = Z_TO_SYMBOL.get(Z, f"Z{Z}")
    nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"

    ace_table = ACEtk.ContinuousEnergyTable.from_file(ace_path)
    T_kelvin = 293.6  # TENDL proton files report 0 K as a placeholder

    mcdc_name = f"{nuclide_name}-{T_kelvin}K.h5"
    out_path = os.path.join(output_dir, mcdc_name)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  {os.path.basename(ace_path)}  ->  {mcdc_name}")
        print(f"  Z={Z}  A={A}  S={S}  T={T_kelvin} K")

    file = h5py.File(out_path, "w")

    # Metadata
    hdr = ace_table.header
    file.attrs["source_title"] = hdr.title
    file.attrs["source_version"] = hdr.version
    file.attrs["source_date"] = hdr.date
    if hasattr(hdr, "comments"):
        file.attrs["source_comments"] = hdr.comments

    file.create_dataset("nuclide_name", data=nuclide_name)
    file.create_dataset("excitation_level", data=S)
    file.create_dataset("temperature", data=T_kelvin).attrs["unit"] = "K"
    file.create_dataset("atomic_number", data=ace_table.atom_number)
    file.create_dataset("atomic_weight_ratio", data=ace_table.atomic_weight_ratio)
    fissionable = ace_table.fission_multiplicity_block is not None
    file.create_dataset("fissionable", data=fissionable)

    # Stopping power
    if pstar_dir is not None:
        pstar_path = os.path.join(pstar_dir, f"{symbol}.txt")
        if os.path.exists(pstar_path):
            if verbose:
                print(f"  Loading PSTAR from {pstar_path}")
            E_s, S_s = load_pstar_file(pstar_path)
            sp = file.create_group("stopping_power")
            sp.create_dataset("energy", data=E_s).attrs["unit"] = "MeV"
            sp.create_dataset("total_stopping_power", data=S_s).attrs["unit"] = (
                "MeV cm2/g"
            )
        elif verbose:
            print(f"  [warn] No PSTAR file for {symbol}")

    # Reaction classification
    nu_block = ace_table.frame_and_multiplicity_block
    rx_block = ace_table.reaction_number_block
    N_reaction = nu_block.number_reactions

    proton_reactions = file.create_group("proton_reactions")
    elastic_group = proton_reactions.create_group("elastic_scattering")
    capture_group = proton_reactions.create_group("capture")
    nonelastic_group = proton_reactions.create_group("nonelastic_reaction")
    fission_group = proton_reactions.create_group("fission")

    elastic_MTs = [2]
    capture_MTs = []
    nonelastic_MTs = []
    fission_MTs = (
        [18]
        if rx_block.has_MT(18)
        else [MT for MT in FISSION_CHANCE_MTS if rx_block.has_MT(MT)]
    )

    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)
        if MT in REDUNDANT_MTS + elastic_MTs + fission_MTs or MT > 891:
            continue
        nu_raw = nu_block.multiplicity(idx)
        if not isinstance(nu_raw, int):
            print_error(f"Non-integer multiplicity for MT-{MT:03} in {ace_path}")
        nu = nu_raw - 100 if nu_raw >= 100 else nu_raw
        if nu == 0:
            capture_MTs.append(MT)
        elif nu > 0:
            nonelastic_MTs.append(MT)
        else:
            print_error(f"Negative multiplicity for MT-{MT:03} in {ace_path}")

    for grp, mts in [
        (elastic_group, elastic_MTs),
        (capture_group, capture_MTs),
        (nonelastic_group, nonelastic_MTs),
        (fission_group, fission_MTs),
    ]:
        for MT in mts:
            grp.create_group(f"MT-{MT:03}").attrs["MT"] = MT

    if verbose:
        print(
            f"  Elastic: {elastic_MTs}  Capture: {capture_MTs}  "
            f"Nonelastic: {nonelastic_MTs}"
            + (f"  Fission: {fission_MTs}" if fissionable else "")
        )

    if not fissionable:
        del file["proton_reactions/fission"]
    if not nonelastic_MTs:
        del file["proton_reactions/nonelastic_reaction"]

    # Cross sections
    xs0 = ace_table.principal_cross_section_block
    xs_main = ace_table.cross_section_block

    proton_reactions.create_dataset(
        "xs_energy_grid", data=np.array(xs0.energies)
    ).attrs["unit"] = "MeV"

    ds = elastic_group.create_dataset("MT-002/xs", data=np.array(xs0.elastic))
    ds.attrs["offset"] = 0
    ds.attrs["unit"] = "barns"

    for mts, grp in [
        (capture_MTs, capture_group),
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs, fission_group if fissionable else None),
    ]:
        if grp is None:
            continue
        for MT in mts:
            idx = rx_block.index(MT)
            ds = grp.create_dataset(
                f"MT-{MT:03}/xs", data=np.array(xs_main.cross_sections(idx))
            )
            ds.attrs["offset"] = xs_main.energy_index(idx) - 1
            ds.attrs["unit"] = "barns"

    # Q-values
    q_block = ace_table.reaction_qvalue_block
    elastic_group.create_dataset("MT-002/Q-value", data=0.0).attrs["unit"] = "MeV"

    for mts, grp in [
        (capture_MTs, capture_group),
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs, fission_group if fissionable else None),
    ]:
        if grp is None:
            continue
        for MT in mts:
            idx = rx_block.index(MT)
            grp.create_dataset(f"MT-{MT:03}/Q-value", data=q_block.q_value(idx)).attrs[
                "unit"
            ] = "MeV"

    # Reference frames
    elastic_group.create_dataset("MT-002/reference_frame", data="COM")

    for mts, grp in [
        (capture_MTs, capture_group),
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs, fission_group if fissionable else None),
    ]:
        if grp is None:
            continue
        for MT in mts:
            idx = rx_block.index(MT)
            rf = nu_block.reference_frame(idx)
            rf_str = (
                "LAB"
                if rf == ACEtk.ReferenceFrame.Laboratory
                else "COM" if rf == ACEtk.ReferenceFrame.CentreOfMass else str(rf)
            )
            grp.create_dataset(f"MT-{MT:03}/reference_frame", data=rf_str)

    # Nonelastic multiplicities
    for MT in nonelastic_MTs:
        idx = rx_block.index(MT)
        nu_raw = nu_block.multiplicity(idx)
        nonelastic_group.create_dataset(
            f"MT-{MT:03}/multiplicity", data=nu_raw - 100 if nu_raw >= 100 else nu_raw
        )

    # Angular distributions
    angle_block = ace_table.angular_distribution_block

    ag = elastic_group.create_group("MT-002/angular_cosine_distribution")
    ag.attrs["type"] = "energy-correlated"
    if (
        not load_cosine_distribution(angle_block.angular_distribution_data(0), ag)
        and verbose
    ):
        print_note("MT-002 angular distribution is given in energy block")

    for mts, grp in [
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs, fission_group if fissionable else None),
    ]:
        if grp is None:
            continue
        for MT in mts:
            idx = rx_block.index(MT)
            ag = grp.create_group(f"MT-{MT:03}/angular_cosine_distribution")
            if (
                not load_cosine_distribution(
                    angle_block.angular_distribution_data(idx), ag
                )
                and verbose
            ):
                print_note(f"MT-{MT:03} angular distribution is given in energy block")

    # Primary energy distributions
    energy_block = ace_table.energy_distribution_block

    for mts, grp in [
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs, fission_group if fissionable else None),
    ]:
        if grp is None:
            continue
        for MT in mts:
            idx = rx_block.index(MT)
            data = energy_block.energy_distribution_data(idx)

            if not isinstance(data, ACEtk.continuous.MultiDistributionData):
                grp.create_dataset(
                    f"MT-{MT:03}/spectrum_probability_grid", data=np.array([0.0, 30.0])
                ).attrs["unit"] = "MeV"
                grp.create_dataset(
                    f"MT-{MT:03}/spectrum_probability", data=np.array([[1.0]])
                )
                load_energy_distribution(
                    data, grp.create_group(f"MT-{MT:03}/energy_spectrum-1")
                )
            else:
                N_dist = data.number_distributions
                probs = data.probabilities

                if all(p.number_interpolation_regions == 0 for p in probs):
                    prob_grid = np.array([0.0, 30.0])
                    prob = np.zeros((1, N_dist))
                    for k in range(N_dist):
                        prob[0, k] = max(data.probability(k + 1).probabilities)
                elif all(p.number_interpolation_regions == 1 for p in probs) and all(
                    p.interpolants[0] == 1 for p in probs
                ):
                    prob_grid = np.array(data.probability(1).energies)
                    prob = np.zeros((len(prob_grid) - 1, N_dist))
                    for k in range(N_dist):
                        prob[:, k] = np.array(
                            data.probability(k + 1).probabilities[:-1]
                        )
                else:
                    print_error(
                        f"Unsupported multi-distribution probability for MT-{MT:03}"
                    )

                grp.create_dataset(
                    f"MT-{MT:03}/spectrum_probability_grid", data=prob_grid
                ).attrs["unit"] = "MeV"
                grp.create_dataset(f"MT-{MT:03}/spectrum_probability", data=prob)
                for k in range(N_dist):
                    load_energy_distribution(
                        data.distribution(k + 1),
                        grp.create_group(f"MT-{MT:03}/energy_spectrum-{k + 1}"),
                    )

    # Secondary particles
    load_secondary_particles(ace_table, file, verbose=verbose)

    # Fission data
    if fissionable:
        prompt_block = ace_table.fission_multiplicity_block
        delayed_block = ace_table.delayed_fission_multiplicity_block
        dnp_block = ace_table.delayed_neutron_precursor_block

        load_fission_multiplicity(
            prompt_block.multiplicity, fission_group.create_group("prompt_multiplicity")
        )
        if delayed_block is not None:
            load_fission_multiplicity(
                delayed_block.multiplicity,
                fission_group.create_group("delayed_multiplicity"),
            )

        if dnp_block is not None:
            N_DNP = dnp_block.number_delayed_precursors
            fractions = np.zeros(N_DNP)
            decay_rates = np.zeros(N_DNP)
            for k in range(N_DNP):
                d = dnp_block.precursor_group_data(k + 1)
                if (
                    d.number_interpolation_regions != 0
                    or len(d.probabilities[:]) != 2
                    or d.probabilities[0] != d.probabilities[1]
                ):
                    print_error("Non-constant delayed neutron precursor fraction")
                fractions[k] = d.probabilities[0]
                decay_rates[k] = d.decay_constant

            prec = fission_group.create_group("delayed_neutron_precursors")
            prec.create_dataset("fractions", data=fractions)
            prec.create_dataset("decay_rates", data=decay_rates).attrs["unit"] = "/s"

            delayed_spectrum_block = ace_table.delayed_neutron_energy_distribution_block
            for k in range(N_DNP):
                load_energy_distribution(
                    delayed_spectrum_block.energy_distribution_data(k + 1),
                    prec.create_group(f"energy_spectrum-{k + 1}"),
                )

    file.close()
    return mcdc_name


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert proton ACE files to MC/DC-compatible HDF5"
    )
    parser.add_argument("--ace_dir", default=os.getenv("MCDC_ACELIB"))
    parser.add_argument("--output_dir", default=os.getenv("MCDC_LIB"))
    parser.add_argument("--pstar_dir", default=os.getenv("PSTAR_LIB"))
    parser.add_argument("--rewrite", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.ace_dir is None:
        print_error("No ACE directory. Use --ace_dir or set $MCDC_ACELIB.")
    if args.output_dir is None:
        print_error("No output directory. Use --output_dir or set $MCDC_LIB.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nACE directory   : {args.ace_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"PSTAR directory : {args.pstar_dir}\n")

    all_files = sorted(os.listdir(args.ace_dir))

    if args.rewrite:
        target_files = all_files
    else:
        target_files = []
        for fname in all_files:
            try:
                with open(os.path.join(args.ace_dir, fname)) as f:
                    hdr = ACEtk.Header.from_string(f.readline())
                Z, A, S, _ = decode_ace_zaid(hdr.zaid)
                symbol = Z_TO_SYMBOL.get(Z, f"Z{Z}")
                nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"
                if not any(
                    f.startswith(nuclide_name + "-")
                    for f in os.listdir(args.output_dir)
                ):
                    target_files.append(fname)
            except Exception:
                target_files.append(fname)

    errors = []
    pbar = tqdm(
        target_files,
        disable=args.verbose,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
    )

    for ace_name in pbar:
        pbar.set_postfix_str(ace_name)
        try:
            out = process_ace_file(
                os.path.join(args.ace_dir, ace_name),
                args.output_dir,
                pstar_dir=args.pstar_dir,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"  -> wrote {out}")
        except Exception as exc:
            errors.append((ace_name, str(exc)))
            if args.verbose:
                import traceback

                traceback.print_exc()

    print(f"\nDone. {len(target_files) - len(errors)} succeeded, {len(errors)} failed.")
    if errors:
        print("\nFailed files:")
        for name, msg in errors:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
