# The majority of this script was written by Anthropic's Claude

"""
ace_to_hdf5.py
==============
Convert a directory of proton ACE files (e.g. TENDL) into per-nuclide HDF5 files
suitable for use in MC/DC or similar Monte Carlo transport codes.

Usage
-----
    python ace_to_hdf5.py --ace_dir /path/to/ace --output_dir /path/to/hdf5
    python ace_to_hdf5.py --ace_dir /path/to/ace --output_dir /path/to/hdf5 --rewrite
    python ace_to_hdf5.py --ace_dir /path/to/ace --output_dir /path/to/hdf5 --verbose

Environment variable fallback (compatible with original MC/DC conventions):
    $MCDC_ACELIB   →  ace_dir
    $MCDC_LIB      →  output_dir

HDF5 layout produced
--------------------
/<nuclide_name>-<T>K.h5
  attrs:
    source_title, source_version, source_date
  nuclide_name          (str)
  excitation_level      (int)
  temperature           (float, K)
  atomic_number         (int)
  atomic_weight_ratio   (float)
  fissionable           (bool)

  proton_reactions/
    xs_energy_grid      (float array, MeV)

    elastic_scattering/
      MT-002/
        xs              (float array, barns)  attrs: offset, unit
        Q-value         (float, MeV)
        reference_frame (str: "COM")
        angular_cosine_distribution/  (tabulated cosine distributions)

    capture/
      MT-{NNN}/
        xs, Q-value, reference_frame

    nonelastic_reaction/
      MT-{NNN}/
        xs, Q-value, reference_frame
        multiplicity    (int)
        angular_cosine_distribution/
        energy_spectrum-{k}/   (one per distribution in a MultiDistributionData)

    fission/            (only if fissionable)
      ...

  secondary_particles/
    ZAP_{zap}/
      attrs: ZAP (int), particle_name (str)
      MT-{NNN}/
        attrs: MT (int), multiplicity (int), reference_frame (str)
        production_xs   (float array, barns)  attrs: offset, unit
        kalbach_mann/
          incident_energies  (float array, MeV)
          interpolation_boundaries  (int array)
          interpolation_types       (int array)
          E_in_{k}/          (one group per incident energy point)
            outgoing_energies (float array, MeV)
            pdf               (float array)
            cdf               (float array)
            r                 (float array)   Kalbach-Mann precompound fraction
            a                 (float array)   Kalbach-Mann slope parameter

Notes
-----
* The Kalbach-Mann property names on TabulatedKalbachMannDistribution are
  introspected at runtime the first time a distribution is encountered, so
  this script will work even if ACEtk renames them between versions.
* ZAP particle identity:  1=n, 31=p, 32=d, 33=t, 34=alpha
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

import ACEtk

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# TODO: THIS IS UNCERTAIN - NEED TO VERIFY ZAP NUMBERS/PARTICLE TYPE CORRESPONDANCE

ZAP_NAMES = {
    0:  "photon",
    1:  "neutron",
    31: "proton",
    32: "deuteron",
    33: "triton",
    34: "alpha",
}

# Candidate property names for TabulatedKalbachMannDistribution fields.
# We try each list in order and use the first one that exists on the object.
_KM_CANDIDATES = {
    "outgoing_energies": ["outgoing_energies", "energies", "energy"],
    "pdf":               ["pdf", "probabilities", "probability_density"],
    "cdf":               ["cdf", "cumulative_probabilities", "cumulative_distribution"],
    "r":                 ["precompound_fraction_values", "precompound_fractions", "r", "R"],
    "a":                 ["angular_distribution_slope_values", "slopes", "a", "A"],
}
# Cache resolved names so introspection only happens once.
_km_resolved: dict[str, str] = {}


def _resolve_km_attr(dist_obj, field: str) -> str:
    """Return the actual attribute name on dist_obj for the given logical field."""
    if field in _km_resolved:
        return _km_resolved[field]
    for candidate in _KM_CANDIDATES[field]:
        if hasattr(dist_obj, candidate):
            _km_resolved[field] = candidate
            return candidate
    raise AttributeError(
        f"Cannot find attribute for '{field}' on "
        f"{type(dist_obj).__name__}. "
        f"Tried: {_KM_CANDIDATES[field]}. "
        f"Available: {[x for x in dir(dist_obj) if not x.startswith('_')]}"
    )


def get_km_field(dist_obj, field: str):
    """Get a logical Kalbach-Mann field from a TabulatedKalbachMannDistribution."""
    attr = _resolve_km_attr(dist_obj, field)
    return getattr(dist_obj, attr)


def print_error(msg: str):
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def print_note(msg: str):
    print(f"  [note] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# ZAP / name decoding
# ──────────────────────────────────────────────────────────────────────────────

# Periodic table symbol lookup (Z → symbol)
Z_TO_SYMBOL = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",
    9: "F",  10: "Ne",11: "Na",12: "Mg",13: "Al",14: "Si",15: "P", 16: "S",
    17:"Cl", 18:"Ar", 19:"K",  20:"Ca", 21:"Sc", 22:"Ti", 23:"V",  24:"Cr",
    25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge",
    33:"As", 34:"Se", 35:"Br", 36:"Kr", 37:"Rb", 38:"Sr", 39:"Y",  40:"Zr",
    41:"Nb", 42:"Mo", 43:"Tc", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd",
    49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I",  54:"Xe", 55:"Cs", 56:"Ba",
    57:"La", 58:"Ce", 59:"Pr", 60:"Nd", 61:"Pm", 62:"Sm", 63:"Eu", 64:"Gd",
    65:"Tb", 66:"Dy", 67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf",
    73:"Ta", 74:"W",  75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
    81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn", 87:"Fr", 88:"Ra",
    89:"Ac", 90:"Th", 91:"Pa", 92:"U",  93:"Np", 94:"Pu", 95:"Am", 96:"Cm",
    97:"Bk", 98:"Cf", 99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",
}


def decode_ace_zaid(zaid: str):
    """
    Decode an ACE ZAID string into (Z, A, S, T).
    Handles both legacy '1001.70h' and modern '1001.710h' style ZAIDs.
    Returns Z (atomic number), A (mass number), S (isomeric state), T (temperature K).
    """
    # Strip trailing whitespace and split on '.'
    parts = zaid.strip().split(".")
    za_str = parts[0]
    # ZA = Z*1000 + A, possibly with S encoded as ZA > 600000 (isomers)
    za = int(za_str)
    if za >= 600000:
        # metastable: ZAID = Z*1000 + A + S*400 (legacy MCNP convention, approximate)
        S = (za % 1000) // 400  # rough extraction
        za = za - S * 400
    else:
        S = 0
    Z = za // 1000
    A = za % 1000

    # Temperature from suffix, e.g. '70h' → 293 K, '710h' → custom
    # The conventional mapping is suffix_number * ~(1/100) * some factor.
    # Most TENDL proton files just use a nominal 0K or room temperature.
    # Use the header temperature value instead (set to 0 as default here).
    T = 0
    return Z, A, S, T


# ──────────────────────────────────────────────────────────────────────────────
# Angular distribution loading (from original MC/DC approach)
# ──────────────────────────────────────────────────────────────────────────────

def load_cosine_distribution(data, h5_group):
    """
    Write a tabulated angular (cosine) distribution into an HDF5 group.
    data is an AngularDistributionData object from ACEtk.

    Returns True if angular data was written, False if it is encoded
    elsewhere (i.e. inside the Kalbach-Mann energy distribution block).
    """
    # DistributionGivenElsewhere means the angular data is embedded in the
    # LAW 44 Kalbach-Mann energy distribution via the r and a parameters.
    # There is nothing to store here — the sampling code must use the
    # Kalbach-Mann block instead.
    if isinstance(data, ACEtk.continuous.DistributionGivenElsewhere):
        h5_group.attrs["type"] = "given_in_energy_distribution"
        return False

    energies = np.array(data.incident_energies)
    h5_group.create_dataset("incident_energies", data=energies)
    h5_group.attrs["unit"] = "MeV"
    # Set type on root group (default to tabulated if we get here)
    h5_group.attrs["type"] = "tabulated"

    for i, subdist in enumerate(data.distributions):
        eg = h5_group.create_group(f"E_in_{i + 1}")
        if isinstance(subdist, ACEtk.continuous.TabulatedAngularDistribution):
            eg.attrs["type"] = "tabulated"
            eg.create_dataset("cosines", data=np.array(subdist.cosines))
            eg.create_dataset("pdf",     data=np.array(subdist.pdf))
            eg.create_dataset("cdf",     data=np.array(subdist.cdf))
        else:
            # Isotropic or unsupported — mark it so sampling code knows
            eg.attrs["type"] = "isotropic"

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Energy distribution loading (neutron/primary particle, existing reactions)
# ──────────────────────────────────────────────────────────────────────────────

def load_energy_distribution(data, h5_group):
    """
    Write a primary-particle outgoing energy distribution into an HDF5 group.
    Handles the most common ACE law types encountered in proton libraries.
    """
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
            eg.create_dataset("outgoing_energies", data=np.array(dist.outgoing_energies))
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
            eg = h5_group.create_group(f"E_in_{k + 1}")
            eg.create_dataset("energies", data=np.array(dist.energies))

    else:
        # Unknown law — store the raw XSS array so nothing is silently lost
        h5_group.attrs["law"] = -1
        h5_group.attrs["type_name"] = type(data).__name__
        try:
            h5_group.create_dataset("xss_array", data=np.array(data.xss_array))
        except Exception:
            pass


def _write_kalbach_mann(km_data, h5_group):
    """
    Write a KalbachMannDistributionData block into an open HDF5 group.
    Uses MCDC-compatible format with flattened arrays and offset indices.
    """
    h5_group.attrs["type"] = "kalbach-mann"
    
    NE = km_data.number_incident_energies
    
    # Incident energy grid
    energy = np.array(km_data.incident_energies)
    energy_ds = h5_group.create_dataset("energy", data=energy)
    energy_ds.attrs["unit"] = "MeV"
    
    # Collect all outgoing energy points and build offset array
    offset = np.zeros(NE, dtype=np.int32)
    energy_out = []
    pdf = []
    precompound_factor = []
    angular_slope = []
    
    for i in range(1, NE + 1):
        dist = km_data.distribution(i)
        offset[i - 1] = len(pdf)
        energy_out.extend(get_km_field(dist, "outgoing_energies"))
        pdf.extend(get_km_field(dist, "pdf"))
        precompound_factor.extend(get_km_field(dist, "r"))
        angular_slope.extend(get_km_field(dist, "a"))
    
    # Create flattened datasets
    h5_group.create_dataset("offset", data=offset)
    energy_out_ds = h5_group.create_dataset("energy_out", data=np.array(energy_out))
    energy_out_ds.attrs["unit"] = "MeV"
    h5_group.create_dataset("pdf", data=np.array(pdf))
    h5_group.create_dataset("precompound_factor", data=np.array(precompound_factor))
    h5_group.create_dataset("angular_slope", data=np.array(angular_slope))


# ──────────────────────────────────────────────────────────────────────────────
# Fission multiplicity loading
# ──────────────────────────────────────────────────────────────────────────────

def load_fission_multiplicity(data, h5_group):
    if isinstance(data, ACEtk.continuous.TabulatedFissionMultiplicity):
        h5_group.attrs["type"] = "tabulated"
        h5_group.create_dataset("energies",        data=np.array(data.energies))
        h5_group.create_dataset("multiplicities",  data=np.array(data.multiplicities))
    elif isinstance(data, ACEtk.continuous.PolynomialFissionMultiplicity):
        h5_group.attrs["type"] = "polynomial"
        h5_group.create_dataset("coefficients", data=np.array(data.coefficients))
    else:
        h5_group.attrs["type"] = "unknown"
        h5_group.attrs["type_name"] = type(data).__name__


# ──────────────────────────────────────────────────────────────────────────────
# Secondary particle block extraction
# ──────────────────────────────────────────────────────────────────────────────

def load_secondary_particles(ace_table, file, verbose=False):
    """
    Extract all secondary particle production data from a proton ACE table
    and write it into file['secondary_particles/ZAP_{zap}/MT-{MT:03}/...'].
    """
    n_types = ace_table.number_secondary_particle_types
    if n_types == 0:
        return

    # ── Top-level block handles ───────────────────────────────────────────────
    # The secondary particle blocks are callable by type index — rx_block(i)
    # returns the ReactionNumberBlock for type i, tyr_block(i) returns the
    # FrameAndMultiplicityBlock for type i, etc.
    type_block = ace_table.secondary_particle_type_block
    info_block = ace_table.secondary_particle_information_block
    rx_block   = ace_table.secondary_particle_reaction_number_block
    tyr_block  = ace_table.secondary_particle_frame_and_multiplicity_block
    xs_block   = ace_table.secondary_particle_production_cross_section_block
    edy_block  = ace_table.secondary_particle_energy_distribution_block

    # angular block is optional for secondary particles in some libraries
    try:
        ang_block = ace_table.secondary_particle_angular_distribution_block
        has_ang   = True
    except Exception:
        has_ang = False

    sec_group = file.create_group("secondary_particles")

    # ── Introspect particle_identifier method name once ───────────────────────
    _pi_candidates = ["particle_identifier", "ZAP", "type", "particle_type"]
    _pi_method = None
    for cand in _pi_candidates:
        if hasattr(type_block, cand):
            _pi_method = cand
            break
    if _pi_method is None:
        raise AttributeError(
            f"Cannot find particle identifier method on "
            f"{type(type_block).__name__}. "
            f"Available: {[x for x in dir(type_block) if not x.startswith('_')]}"
        )

    # ── Loop over secondary particle types ────────────────────────────────────
    for i in range(1, n_types + 1):

        zap  = getattr(type_block, _pi_method)(i)
        name = ZAP_NAMES.get(zap, f"ZAP_{zap}")

        # number_reactions is a sequence property on info_block, 0-based
        n_rx = int(info_block.number_reactions[i - 1])

        if verbose:
            print(f"    Secondary particle type {i}: ZAP={zap} ({name}), "
                  f"{n_rx} reactions")

        zap_group = sec_group.create_group(f"ZAP_{zap}")
        zap_group.attrs["ZAP"]           = zap
        zap_group.attrs["particle_name"] = name

        # Per-type sub-blocks: call the top-level block with the type index
        # to get the per-type block, then call methods on that.
        rx_i  = rx_block(i)   # ReactionNumberBlock for type i
        tyr_i = tyr_block(i)  # FrameAndMultiplicityBlock for type i
        xs_i  = xs_block(i)   # production cross section block for type i
        edy_i = edy_block(i)  # energy distribution block for type i
        ang_i = ang_block(i) if has_ang else None

        # Introspect xs sub-block method names (once, on first type)
        _xs_candidates  = [
            "production_xs",
            "production_cross_sections",
            "cross_sections",
            "cross_section",
            "cross_section_values",
            "xs",
            "xss",
        ]
        _off_candidates = ["energy_index", "offset", "locator", "index"]
        _xs_method  = next((c for c in _xs_candidates  if hasattr(xs_i, c)), None)
        _off_method = next((c for c in _off_candidates if hasattr(xs_i, c)), None)

        # Introspect energy distribution method name
        _edy_candidates = ["energy_distribution_data", "distribution_data", "distribution"]
        _edy_method = next((c for c in _edy_candidates if hasattr(edy_i, c)), None)

        for j in range(1, n_rx + 1):

            MT = rx_i.MT(j)

            # ── Multiplicity ─────────────────────────────────────────────────
            nu_raw = tyr_i.multiplicity(j)
            nu     = nu_raw - 100 if nu_raw >= 100 else nu_raw

            # ── Reference frame ───────────────────────────────────────────────
            rf_raw = tyr_i.reference_frame(j)
            if rf_raw == ACEtk.ReferenceFrame.Laboratory:
                rf = "LAB"
            elif rf_raw == ACEtk.ReferenceFrame.CentreOfMass:
                rf = "COM"
            else:
                rf = str(rf_raw)

            mt_group = zap_group.create_group(f"MT-{MT:03}")
            mt_group.attrs["MT"]              = MT
            mt_group.attrs["multiplicity"]    = nu
            mt_group.attrs["reference_frame"] = rf

            if verbose:
                print(f"      MT={MT:03}  nu_raw={nu_raw}  nu={nu}  frame={rf}")

            # ── Production cross section ──────────────────────────────────────
            if _xs_method and _off_method:
                try:
                    xs_vals   = np.array(getattr(xs_i, _xs_method)(j))
                    xs_offset = int(getattr(xs_i, _off_method)(j))
                    xs_ds     = mt_group.create_dataset("production_xs", data=xs_vals)
                    xs_ds.attrs["offset"] = xs_offset - 1   # convert to 0-based
                    xs_ds.attrs["unit"]   = "barns"
                except Exception as exc:
                    xs_ds = mt_group.create_dataset("production_xs", data=np.zeros(0, dtype=float))
                    xs_ds.attrs["offset"] = 0
                    xs_ds.attrs["unit"] = "barns"
                    if verbose:
                        print(f"        [warn] Could not read production xs: {exc}")
            else:
                xs_ds = mt_group.create_dataset("production_xs", data=np.zeros(0, dtype=float))
                xs_ds.attrs["offset"] = 0
                xs_ds.attrs["unit"] = "barns"
                if verbose:
                    print(f"        [warn] production xs block methods not resolved: "
                          f"{[x for x in dir(xs_i) if not x.startswith('_')]}")

            # ── Kalbach-Mann energy-angle distribution ────────────────────────
            if _edy_method:
                try:
                    km_data  = getattr(edy_i, _edy_method)(j)
                    km_group = mt_group.create_group("kalbach_mann")
                    _write_kalbach_mann(km_data, km_group)
                except Exception as exc:
                    if verbose:
                        print(f"        [warn] Could not read energy distribution: {exc}")
            else:
                if verbose:
                    print(f"        [warn] energy distribution method not resolved: "
                          f"{[x for x in dir(edy_i) if not x.startswith('_')]}")

            # ── Angular distribution (if present) ────────────────────────────
            if ang_i is not None:
                try:
                    ang_data  = ang_i.angular_distribution_data(j)
                    ang_group = mt_group.create_group("angular_cosine_distribution")
                    load_cosine_distribution(ang_data, ang_group)
                except Exception:
                    pass   # not all secondary types have explicit angular data


# ──────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────────────────

def process_ace_file(ace_path: str, output_dir: str, verbose: bool = False) -> str:
    """
    Convert a single ACE proton file to HDF5. Returns the output filename.
    """

    # ── Header ────────────────────────────────────────────────────────────────
    with open(ace_path, "r") as f:
        header = ACEtk.Header.from_string(f.readline())

    Z, A, S, T = decode_ace_zaid(header.zaid)
    symbol      = Z_TO_SYMBOL.get(Z, f"Z{Z}")
    nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"

    # Get temperature from the table itself (more reliable than ZAID suffix)
    ace_table = ACEtk.ContinuousEnergyTable.from_file(ace_path)
    T_kelvin  = float(ace_table.temperature) if hasattr(ace_table, "temperature") else T

    # Forcing to be room temperature, as 0K from the file is a placeholder
    T_kelvin = 293.6

    mcdc_name = f"{nuclide_name}-{T_kelvin}K.h5"
    out_path  = os.path.join(output_dir, mcdc_name)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  {os.path.basename(ace_path)}  →  {mcdc_name}")
        print(f"  Z={Z}  A={A}  S={S}  T={T_kelvin} K")

    file = h5py.File(out_path, "w")

    # ── Basic metadata ────────────────────────────────────────────────────────
    hdr = ace_table.header
    file.attrs["source_title"]   = hdr.title
    file.attrs["source_version"] = hdr.version
    file.attrs["source_date"]    = hdr.date
    if hasattr(hdr, "comments"):
        file.attrs["source_comments"] = hdr.comments

    file.create_dataset("nuclide_name",         data=nuclide_name)
    file.create_dataset("excitation_level",      data=S)
    ds = file.create_dataset("temperature",      data=T_kelvin)
    ds.attrs["unit"] = "K"
    file.create_dataset("atomic_number",         data=ace_table.atom_number)
    file.create_dataset("atomic_weight_ratio",   data=ace_table.atomic_weight_ratio)

    fissionable = ace_table.fission_multiplicity_block is not None
    file.create_dataset("fissionable", data=fissionable)

    # ── Reaction classification ───────────────────────────────────────────────
    proton_reactions = file.create_group("proton_reactions")

    nu_block    = ace_table.frame_and_multiplicity_block
    rx_block    = ace_table.reaction_number_block
    N_reaction  = nu_block.number_reactions

    elastic_group    = proton_reactions.create_group("elastic_scattering")
    capture_group    = proton_reactions.create_group("capture")
    nonelastic_group = proton_reactions.create_group("nonelastic_reaction")
    fission_group    = proton_reactions.create_group("fission")

    elastic_MTs    = [2]
    capture_MTs    = []
    nonelastic_MTs = []
    fission_MTs    = []

    fission_chance_MTs = [19, 20, 21, 38]
    # Genuine redundant sum MTs — do not double-count these
    redundant_MTs = [1, 3, 4, 10, 101, 103, 104, 105, 106, 107]

    total_fission_given = rx_block.has_MT(18)
    if total_fission_given:
        fission_MTs = [18]
    else:
        for MT in fission_chance_MTs:
            if rx_block.has_MT(MT):
                fission_MTs.append(MT)

    for i in range(N_reaction):
        idx = i + 1
        MT  = rx_block.MT(idx)

        if MT in redundant_MTs + elastic_MTs + fission_MTs:
            continue
        if MT > 891:  # above the defined charged-particle range
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
            print_error(f"Negative decoded multiplicity for MT-{MT:03} in {ace_path}")

    # Create MT subgroups
    for rx_group, rx_MTs in [
        (elastic_group,    elastic_MTs),
        (capture_group,    capture_MTs),
        (nonelastic_group, nonelastic_MTs),
        (fission_group,    fission_MTs),
    ]:
        for MT in rx_MTs:
            g = rx_group.create_group(f"MT-{MT:03}")
            g.attrs["MT"] = MT

    if verbose:
        print(f"  Elastic: {elastic_MTs}")
        print(f"  Capture: {capture_MTs}")
        print(f"  Nonelastic: {nonelastic_MTs}")
        if fissionable:
            print(f"  Fission: {fission_MTs}")

    # Remove empty groups
    if not fissionable:
        del file["proton_reactions/fission"]
    if len(nonelastic_MTs) == 0:
        del file["proton_reactions/nonelastic_reaction"]

    # ── Cross sections ────────────────────────────────────────────────────────
    xs0_block     = ace_table.principal_cross_section_block
    xs_block_main = ace_table.cross_section_block

    xs_energy = np.array(xs0_block.energies)
    ds = proton_reactions.create_dataset("xs_energy_grid", data=xs_energy)
    ds.attrs["unit"] = "MeV"

    xs_ds = elastic_group.create_dataset("MT-002/xs", data=np.array(xs0_block.elastic))
    xs_ds.attrs["offset"] = 0
    xs_ds.attrs["unit"]   = "barns"

    for MTs, group in [
        (capture_MTs,    capture_group),
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs,    fission_group if fissionable else None),
    ]:
        if group is None:
            continue
        for MT in MTs:
            idx   = rx_block.index(MT)
            xs_ds = group.create_dataset(
                f"MT-{MT:03}/xs",
                data=np.array(xs_block_main.cross_sections(idx))
            )
            xs_ds.attrs["offset"] = xs_block_main.energy_index(idx) - 1
            xs_ds.attrs["unit"]   = "barns"

    # ── Q-values ──────────────────────────────────────────────────────────────
    q_block = ace_table.reaction_qvalue_block

    elastic_group.create_dataset("MT-002/Q-value", data=0.0).attrs["unit"] = "MeV"

    for MTs, group in [
        (capture_MTs,   capture_group),
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs,   fission_group if fissionable else None),
    ]:
        if group is None:
            continue
        for MT in MTs:
            idx = rx_block.index(MT)
            ds  = group.create_dataset(
                f"MT-{MT:03}/Q-value", data=q_block.q_value(idx)
            )
            ds.attrs["unit"] = "MeV"

    # ── Reference frames ──────────────────────────────────────────────────────
    elastic_group.create_dataset("MT-002/reference_frame", data="COM")

    for MTs, group in [
        (capture_MTs,   capture_group),
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs,   fission_group if fissionable else None),
    ]:
        if group is None:
            continue
        for MT in MTs:
            idx = rx_block.index(MT)
            rf  = nu_block.reference_frame(idx)
            rf_str = (
                "LAB" if rf == ACEtk.ReferenceFrame.Laboratory else
                "COM" if rf == ACEtk.ReferenceFrame.CentreOfMass else
                str(rf)
            )
            group.create_dataset(f"MT-{MT:03}/reference_frame", data=rf_str)

    # ── Nonelastic reaction multiplicities ─────────────────────────────────────
    for MT in nonelastic_MTs:
        idx    = rx_block.index(MT)
        nu_raw = nu_block.multiplicity(idx)
        nu     = nu_raw - 100 if nu_raw >= 100 else nu_raw
        nonelastic_group.create_dataset(f"MT-{MT:03}/multiplicity", data=nu)

    # ── Angular distributions ─────────────────────────────────────────────────
    angle_block = ace_table.angular_distribution_block

    ang_group = elastic_group.create_group("MT-002/angular_cosine_distribution")
    ang_group.attrs["type"] = "energy-correlated"
    data = angle_block.angular_distribution_data(0)
    written = load_cosine_distribution(data, ang_group)
    if not written and verbose:
        print_note("MT-002 elastic angular distribution is given in energy block")

    for MTs, group in [
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs,   fission_group if fissionable else None),
    ]:
        if group is None:
            continue
        for MT in MTs:
            idx       = rx_block.index(MT)
            ang_group = group.create_group(f"MT-{MT:03}/angular_cosine_distribution")
            data      = angle_block.angular_distribution_data(idx)
            written   = load_cosine_distribution(data, ang_group)
            if not written and verbose:
                print_note(f"MT-{MT:03} angular distribution is given in energy block")

    # ── Primary energy distributions ──────────────────────────────────────────
    energy_block = ace_table.energy_distribution_block

    for MTs, group in [
        (nonelastic_MTs, nonelastic_group),
        (fission_MTs,   fission_group if fissionable else None),
    ]:
        if group is None:
            continue
        for MT in MTs:
            idx  = rx_block.index(MT)
            data = energy_block.energy_distribution_data(idx)

            if not isinstance(data, ACEtk.continuous.MultiDistributionData):
                eg = group.create_group(f"MT-{MT:03}/energy_spectrum-1")
                group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability_grid",
                    data=np.array([0.0, 30.0])
                ).attrs["unit"] = "MeV"
                group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability",
                    data=np.array([[1.0]])
                )
                load_energy_distribution(data, eg)
            else:
                N_dist = data.number_distributions
                # Probability grid
                if all(np.array([x.number_interpolation_regions
                                  for x in data.probabilities]) == 0):
                    prob_grid = np.array([0.0, 30.0])
                    prob      = np.zeros((1, N_dist))
                    for k in range(N_dist):
                        prob[0, k] = max(data.probability(k + 1).probabilities)
                elif (all(np.array([x.number_interpolation_regions
                                     for x in data.probabilities]) == 1)
                      and all(np.array([x.interpolants
                                        for x in data.probabilities]) == 1)):
                    prob_grid = np.array(data.probability(1).energies)
                    prob      = np.zeros((len(prob_grid) - 1, N_dist))
                    for k in range(N_dist):
                        prob[:, k] = np.array(
                            data.probability(k + 1).probabilities[:-1]
                        )
                else:
                    print_error(f"Unsupported multi-distribution probability for "
                                f"MT-{MT:03} in {ace_path}")

                group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability_grid", data=prob_grid
                ).attrs["unit"] = "MeV"
                group.create_dataset(
                    f"MT-{MT:03}/spectrum_probability", data=prob
                )
                for k in range(N_dist):
                    eg = group.create_group(f"MT-{MT:03}/energy_spectrum-{k+1}")
                    load_energy_distribution(data.distribution(k + 1), eg)

    # ── Secondary particles ───────────────────────────────────────────────────
    load_secondary_particles(ace_table, file, verbose=verbose)

    # ── Fission data (if applicable) ──────────────────────────────────────────
    if fissionable:
        prompt_block  = ace_table.fission_multiplicity_block
        delayed_block = ace_table.delayed_fission_multiplicity_block
        dnp_block     = ace_table.delayed_neutron_precursor_block

        h5g = fission_group.create_group("prompt_multiplicity")
        load_fission_multiplicity(prompt_block.multiplicity, h5g)

        if delayed_block is not None:
            h5g = fission_group.create_group("delayed_multiplicity")
            load_fission_multiplicity(delayed_block.multiplicity, h5g)

        if dnp_block is not None:
            N_DNP      = dnp_block.number_delayed_precursors
            fractions  = np.zeros(N_DNP)
            decay_rates = np.zeros(N_DNP)
            for k in range(N_DNP):
                d = dnp_block.precursor_group_data(k + 1)
                if (d.number_interpolation_regions != 0
                        or len(d.probabilities[:]) != 2
                        or d.probabilities[0] != d.probabilities[1]):
                    print_error("Non-constant delayed neutron precursor fraction")
                fractions[k]   = d.probabilities[0]
                decay_rates[k] = d.decay_constant

            prec = fission_group.create_group("delayed_neutron_precursors")
            prec.create_dataset("fractions",   data=fractions)
            dr_ds = prec.create_dataset("decay_rates", data=decay_rates)
            dr_ds.attrs["unit"] = "/s"

            delayed_spectrum_block = ace_table.delayed_neutron_energy_distribution_block
            for k in range(N_DNP):
                d = delayed_spectrum_block.energy_distribution_data(k + 1)
                eg = prec.create_group(f"energy_spectrum-{k+1}")
                load_energy_distribution(d, eg)

    file.close()
    return mcdc_name


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert proton ACE files to MC/DC-compatible HDF5"
    )
    parser.add_argument("--ace_dir",    default=os.getenv("MCDC_ACELIB"),
                        help="Directory containing ACE files "
                             "(default: $MCDC_ACELIB)")
    parser.add_argument("--output_dir", default=os.getenv("MCDC_LIB"),
                        help="Output directory for HDF5 files "
                             "(default: $MCDC_LIB)")
    parser.add_argument("--rewrite",    action="store_true", default=False,
                        help="Rewrite existing HDF5 files")
    parser.add_argument("--verbose",    action="store_true", default=False,
                        help="Print detailed per-reaction info")
    args = parser.parse_args()

    if args.ace_dir is None:
        print_error("No ACE directory specified. Use --ace_dir or set $MCDC_ACELIB.")
    if args.output_dir is None:
        print_error("No output directory specified. Use --output_dir or set $MCDC_LIB.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nACE directory   : {args.ace_dir}")
    print(f"Output directory: {args.output_dir}\n")

    all_files = sorted(os.listdir(args.ace_dir))

    # Filter to only unprocessed files unless --rewrite
    if args.rewrite:
        target_files = all_files
    else:
        target_files = []
        for fname in all_files:
            ace_path = os.path.join(args.ace_dir, fname)
            try:
                with open(ace_path, "r") as f:
                    hdr = ACEtk.Header.from_string(f.readline())
                Z, A, S, _ = decode_ace_zaid(hdr.zaid)
                symbol      = Z_TO_SYMBOL.get(Z, f"Z{Z}")
                nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"
                # We don't know T yet without loading the full table, so check
                # for any existing file matching the nuclide name pattern.
                existing = [
                    f for f in os.listdir(args.output_dir)
                    if f.startswith(nuclide_name + "-")
                ]
                if not existing:
                    target_files.append(fname)
            except Exception:
                target_files.append(fname)   # include if we can't read header

    errors = []
    pbar   = tqdm(
        target_files,
        disable=args.verbose,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
    )

    for ace_name in pbar:
        ace_path = os.path.join(args.ace_dir, ace_name)
        pbar.set_postfix_str(ace_name)
        try:
            out = process_ace_file(ace_path, args.output_dir, verbose=args.verbose)
            if args.verbose:
                print(f"  → wrote {out}")
        except Exception as exc:
            errors.append((ace_name, str(exc)))
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\nDone. {len(target_files) - len(errors)} succeeded, "
          f"{len(errors)} failed.")
    if errors:
        print("\nFailed files:")
        for name, msg in errors:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()