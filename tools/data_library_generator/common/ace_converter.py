"""Convert ACE data into MC/DC HDF5 records.

This module handles ACE parsing, physics-record conversion, unit conversion,
and HDF5 layout. Projectile generators coordinate output files and non-ACE
inputs. Source energies are in MeV; output energy coordinates use eV, cross
sections use barns, and dimensionless values have no unit attribute.
"""

from dataclasses import dataclass
from functools import wraps
from numbers import Integral
import re
from pathlib import Path

import ACEtk
import numpy as np

from .hdf5_writer import (
    PROBABILITY_TOLERANCE,
    ProbabilityTableError,
    incident_interpolation,
    interpolation,
    multi_table,
    pdf_table,
    pdf_tables,
    quantity,
)

# ======================================================================================
# ACE units, interpolation codes, and source-library conventions
# ======================================================================================


EV_PER_MEV = 1.0e6
BOLTZMANN_K = 8.61733326e-5
ACE_INTERPOLATION = {
    1: "histogram",
    2: "linear",
    3: "semilog-x",
    4: "semilog-y",
    5: "log",
}
ACE_NOMINAL_TEMPERATURE_LIBRARIES = ("Lib81", "Lib81x")
# Lib81 and Lib81x share these nominal temperature suffixes.
ACE_TEMPERATURE_LIB81 = {
    "10c": 293.6,
    "11c": 600.0,
    "12c": 900.0,
    "13c": 1200.0,
    "14c": 2500.0,
    "15c": 0.1,
    "16c": 233.15,
    "17c": 273.15,
}

# ======================================================================================
# Neutron reaction selection and transport category order
# ======================================================================================


FISSION_MTS = {18, 19, 20, 21, 38}
REDUNDANT_MTS = {1, 3, 4, 10}
# ENDF-6 Appendix B: MT 152--200 are independent reaction channels, including
# (n,5n)--(n,8n) and mixed neutron/charged-particle emission, not production totals.
# https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf
NEUTRON_REACTION_MT_RANGES = ((1, 117), (152, 200))
CATEGORY_ORDER = ("elastic-scattering", "capture", "inelastic-scattering", "fission")


# ======================================================================================
# Source context for conversion errors
# ======================================================================================


def _probability_table_context(source, error):
    """Describe a failed packed table by its ACE source coordinates."""
    index = error.index
    if error.path and error.path.endswith("/cosine-distributions"):
        for table in source.distributions:
            if index < len(table.outgoing_energies):
                return (
                    f"cosine table {index} (E_in = {table.incident_energy:g} MeV, "
                    f"E_out = {table.outgoing_energies[index]:g} MeV)"
                )
            index -= len(table.outgoing_energies)
    energies = getattr(source, "incident_energies", ())
    kind = "outgoing-energy" if hasattr(source, "LAW") else "angular-cosine"
    label = f"{kind} table {error.index}"
    if error.index < len(energies):
        label += f" (E_in = {energies[error.index]:g} MeV)"
    return label


def _distribution_errors(writer):
    """Add the HDF5 path and ACE law to distribution-conversion errors."""

    @wraps(writer)
    def write(group, source):
        """Run a distribution writer and add context to its ValueError."""
        try:
            return writer(group, source)
        except ValueError as error:
            location = group.name
            law = getattr(source, "LAW", None)
            if law is not None:
                location += f" (ACE LAW {int(law)})"
            if isinstance(error, ProbabilityTableError):
                detail = f"{_probability_table_context(source, error)}: {error.reason}"
            else:
                detail = str(error)
            raise ValueError(f"{location}: {detail}") from error

    return write


# ======================================================================================
# Target identity and source-library metadata
# ======================================================================================


def identify_source_library(ace):
    """Recognize a library label in header text, returning None if unknown.

    LANL labels match ``Lib<number>[x]`` and retain their original spelling.
    TENDL release labels use ``TENDL-YYYY`` or ``TENDL YYYY`` and are returned
    as ``TENDL-YYYY``. A LANL label takes precedence over a TENDL reference
    to the underlying evaluation. Library-specific temperature conventions
    apply only to explicitly supported releases; recognizing a label does
    not imply a temperature mapping or restrict which inputs can be converted.
    """
    header = ace.header
    title = str(getattr(header, "title", "") or "")
    comments = str(getattr(header, "comments", "") or "")
    text = f"{title} {comments}"
    tokens = text.split()
    library = next(
        (token for token in tokens if re.fullmatch(r"Lib[0-9]+x?", token)), None
    )
    if library is not None:
        return library
    release = re.search(r"\bTENDL[- ](\d{4})\b", text)
    return f"TENDL-{release.group(1)}" if release else None


@dataclass(frozen=True)
class Target:
    """Immutable target identity and temperature for a neutron evaluation.

    Attributes
    ----------
    symbol : str
        Element symbol with standard capitalization, such as ``"Co"``.
    atomic_number : int
        Number of protons in the target nucleus.
    mass_number : int
        Total number of protons and neutrons in the target nucleus.
    excitation_level : int
        Excitation-state index, with zero denoting the ground state. This is
        an index used in names such as ``Co58m1``, not an excitation energy.
    temperature : float
        Target temperature in kelvin. Recognized library conventions provide
        nominal values; otherwise it is derived from the ACE thermal energy.
    """

    symbol: str
    atomic_number: int
    mass_number: int
    excitation_level: int
    temperature: float

    @property
    def nuclide(self):
        """Return the nuclide name, including an isomer suffix when needed."""
        suffix = f"m{self.excitation_level}" if self.excitation_level else ""
        return f"{self.symbol}{self.mass_number}{suffix}"

    @property
    def filename(self):
        """Return the temperature-file basename, for example ``neutron-H2-293.6K.h5``.

        Returns
        -------
        str
            The neutron prefix, nuclide identifier, target temperature, and
            the suffix ``K.h5``. No directory is included.
        """
        return f"neutron-{self.nuclide}-{self.temperature}K.h5"


def read_target(ace):
    """Extract target identity and temperature from an ACE table.

    Parameters
    ----------
    ace : ACEtk.ContinuousEnergyTable
        Parsed table exposing its ZAID, atomic number, mass number, and
        isomeric-state index through ACEtk.

    Returns
    -------
    Target
        Nuclide identity and target temperature for the neutron output.

    Raises
    ------
    ValueError
        The header thermal energy is negative or nonfinite.
    KeyError
        The atomic number is absent from the element-symbol lookup.

    Notes
    -----
    ACEtk supplies the interpreted nuclide and isomer identities; this
    function does not decode them from the ZAID's leading digits. Lib81 and
    Lib81x suffixes use their nominal temperatures when a mapping exists.
    Other cases use header kT in MeV, converted to kelvin. The original ACE
    thermal energy is retained in provenance by read_source_libraries.
    Provenance label overrides do not change temperature interpretation.
    """
    # Use header thermal energy unless a recognized nominal convention applies.
    kT = float(ace.header.temperature) * EV_PER_MEV
    if not np.isfinite(kT) or kT < 0:
        raise ValueError("ACE header kT must be finite and nonnegative")
    temperature = kT / BOLTZMANN_K
    suffix = ace.header.zaid.split(".")[-1]
    if identify_source_library(ace) in ACE_NOMINAL_TEMPERATURE_LIBRARIES:
        temperature = ACE_TEMPERATURE_LIB81.get(suffix, temperature)
    return Target(
        Z_TO_SYMBOL[ace.atom_number],
        ace.atom_number,
        ace.mass_number,
        ace.isomeric_state,
        temperature,
    )


def read_source_libraries(
    ace, target=None, with_temperature=False, *, source_library=None
):
    """Build source-library metadata for one output file without writing it.

    Parameters
    ----------
    ace : ACEtk.ContinuousEnergyTable
        Parsed table whose header provides the source title, comments,
        format version, processing date, and thermal energy.
    target : Target, optional
        Interpreted target identity and temperature; required when including
        temperature metadata.
    with_temperature : bool, optional
        Include the target ``temperature`` in K and original ACE ``kT``
        converted from MeV to eV. Projectile generators enable this only
        for temperature-dependent projectile files.
    source_library : str, optional
        Provenance subgroup name. When omitted, use a recognized header label
        or ``unknown``. This label does not affect temperature interpretation.

    Returns
    -------
    dict of str to dict
        Library identifiers mapped to scalar metadata values, suitable for
        ``hdf5_writer.stamp``. Library labels never restrict conversion.
        String fields are ``title``, ``version``, ``release-date``,
        ``reference``, ``comments``, ``format``, ``format-version``, and
        ``processing-date``; optional temperature fields are numeric.

    Raises
    ------
    ValueError
        The supplied source identifier is empty, contains a slash, or is a
        reserved HDF5 path component; or temperature metadata lacks a target.

    Notes
    -----
    Library identity comes from header text or an explicit caller-supplied
    identifier, not the ZAID or input directory. Unknown libraries retain
    their title, comments, format, and processing date without rejection.
    The ACE header version and date describe the source format and processing;
    they are not assigned to the library's release version or release date.
    Unavailable release information and other missing strings are empty.
    A reference is extracted from the title's text following ``Ref.``.

    This function does not create ``conversion-time``. The HDF5 writer records
    that timestamp when it actually writes the source record, allowing
    skipped conversions to leave provenance unchanged.
    """
    # Resolve the provenance label while retaining the original header fields.
    header = ace.header
    title = str(getattr(header, "title", "") or "")
    comments = str(getattr(header, "comments", "") or "")
    detected_library = identify_source_library(ace)
    library = (
        source_library if source_library is not None else detected_library or "unknown"
    )
    if (
        not isinstance(library, str)
        or not library.strip()
        or library in (".", "..")
        or "/" in library
    ):
        raise ValueError(f"Invalid source-library identifier: {library!r}")
    # Collect provenance applicable to the target file
    source = {
        "title": title,
        "version": (
            detected_library.split("-", 1)[1]
            if detected_library and detected_library.startswith("TENDL-")
            else ""
        ),
        "release-date": "",
        "reference": title.split("Ref.", 1)[1].strip() if "Ref." in title else "",
        "comments": comments,
        "format": "ACE",
        "format-version": str(getattr(header, "version", "") or ""),
        "processing-date": str(getattr(header, "date", "") or ""),
    }
    # Attach source temperature information to temperature-specific records
    if with_temperature:
        if target is None:
            raise ValueError("Source temperature metadata requires a target")
        source["temperature"] = target.temperature
        source["kT"] = float(header.temperature) * EV_PER_MEV
    return {library: source}


# ======================================================================================
# Interpolation
# ======================================================================================


def interpolation_law(code):
    """Map a supported ACE interpolation code to its MC/DC law name."""
    try:
        return ACE_INTERPOLATION[int(code)]
    except KeyError:
        raise ValueError(f"Unsupported ACE interpolation law: {code}") from None


def write_interpolation(group, source, count):
    """Write ACE interpolation laws and point boundaries to a table group."""
    laws, boundaries = _interpolation_regions(source, count)
    interpolation(group, count, laws, boundaries)


def _interpolation_regions(source, count):
    """Return named interpolation laws and their point boundaries."""
    laws = [interpolation_law(code) for code in source.interpolants]
    boundaries = np.asarray(source.boundaries, dtype=np.int64)
    if not laws:
        laws, boundaries = ["linear"], [count]
    return laws, boundaries


# ======================================================================================
# Neutron reaction names, selection, and output
# ======================================================================================


def reaction_name(mt):
    """Return the MC/DC group name for an MT, with a fallback for unmapped MTs."""
    names = {
        2: "elastic",
        16: "neutron-emission-2",
        17: "neutron-emission-3",
        18: "fission-total",
        19: "fission-first-chance",
        20: "fission-second-chance",
        21: "fission-third-chance",
        38: "fission-fourth-chance",
        91: "inelastic-continuum",
        102: "radiative-capture",
        103: "proton-emission",
        104: "deuteron-emission",
        105: "triton-emission",
        106: "helium-3-emission",
        107: "alpha-emission",
        152: "neutron-emission-5",
        153: "neutron-emission-6",
        160: "neutron-emission-7",
        161: "neutron-emission-8",
    }
    if 51 <= mt <= 90:
        return f"inelastic-level-{mt - 50}"
    return names.get(mt, f"reaction-{mt}")


def retained_reactions(ace):
    """Select supported neutron reactions and assign their transport categories.

    Returns sorted ``(MT, ACE index, category)`` records, including elastic
    scattering with index zero. Rejects inconsistent reaction blocks and
    unsupported neutron multiplicities.
    """
    # Check that source reaction records can be interpreted consistently
    numbers = ace.reaction_number_block
    yields = ace.frame_and_multiplicity_block
    if numbers.number_reactions != yields.number_reactions:
        raise ValueError("Reaction and multiplicity blocks have different lengths")
    if numbers.has_MT(18) and any(numbers.has_MT(mt) for mt in FISSION_MTS - {18}):
        raise ValueError("Both total fission and its chance components are present")
    # Include elastic once and filter the remaining source reaction channels
    reactions = [(2, 0, "elastic-scattering")]
    for index in range(1, numbers.number_reactions + 1):
        mt = int(numbers.MT(index))
        if mt in REDUNDANT_MTS or mt == 2:
            continue
        if not any(first <= mt <= last for first, last in NEUTRON_REACTION_MT_RANGES):
            continue
        # Assign transport behavior from fission identity or outgoing neutron count
        if mt in FISSION_MTS:
            category = "fission"
        else:
            count = yields.multiplicity(index)
            if not isinstance(count, Integral) or count < 0:
                raise ValueError(
                    f"MT {mt}: expected a nonnegative integer neutron multiplicity"
                )
            category = "capture" if count == 0 else "inelastic-scattering"
        reactions.append((mt, index, category))
    return sorted(reactions, key=lambda r: (CATEGORY_ORDER.index(r[2]), r[0]))


def write_temperature(file, ace, target, verbose=False):
    """Write target temperature, the neutron grid, and selected reactions.

    Parameters
    ----------
    file : h5py.File
        Writable output containing no temperature datasets or neutron group.
        The caller manages its lifetime and stamps format/source metadata.
    ace : ACEtk.ContinuousEnergyTable
        Source of the principal energy grid and neutron reaction records.
    target : Target
        Target temperature in K used for ``temperature`` and for computing
        ``kT`` in eV with ``BOLTZMANN_K``.
    verbose : bool, optional
        Print each selected reaction's MT, output name, and category.

    Raises
    ------
    ValueError
        Reaction selection or a delegated reaction/distribution conversion
        encounters unsupported or inconsistent source data.

    Notes
    -----
    The source principal energy grid is converted from MeV to eV and stored
    once as ``xs-energy-grid``. Reaction groups are
    created in selection order and populated by ``write_reaction``.
    Aggregate or auxiliary data excluded by ``retained_reactions`` is not
    written. Source thermal-energy provenance is supplied separately by
    ``read_source_libraries``; it may differ from computed ``kT`` through
    source rounding.

    Writes are direct and may leave a partially populated open file on
    failure. The caller uses ``hdf5_writer.atomic_file`` to publish the
    projectile file only after conversion succeeds.
    """
    # Write the target thermodynamic state and shared neutron energy grid
    quantity(file, "temperature", target.temperature, "K")
    quantity(file, "kT", target.temperature * BOLTZMANN_K, "eV")
    neutron = file
    energy = np.asarray(ace.principal_cross_section_block.energies) * EV_PER_MEV
    quantity(neutron, "xs-energy-grid", energy, "eV")
    # Populate one group per retained reaction in transport category order
    reactions = neutron.create_group("reactions", track_order=True)
    for mt, index, category in retained_reactions(ace):
        if verbose:
            print(f"  MT {mt}: {reaction_name(mt)} ({category})")
        group = reactions.create_group(reaction_name(mt), track_order=True)
        try:
            write_reaction(group, ace, mt, index, category, len(energy))
        except ValueError as error:
            raise ValueError(f"MT {mt}: {error}") from error


def write_reaction(group, ace, mt, index, category, energy_count):
    """Write a neutron reaction's cross section and category-specific data.

    Inelastic and fission reactions also include their multiplicity and
    distributions. Cross sections must align with the shared energy grid.
    """
    # Resolve reaction identity, cross-section values, Q-value, and reference frame
    group.attrs["ENDF-MT"] = mt
    group.attrs["category"] = category
    if mt == 2:
        values = np.asarray(ace.principal_cross_section_block.elastic)
        offset, q_value, frame = 0, 0.0, "COM"
    else:
        values = np.asarray(ace.cross_section_block.cross_sections(index))
        offset = ace.cross_section_block.energy_index(index) - 1
        q_value = ace.reaction_qvalue_block.q_value(index) * EV_PER_MEV
        frames = {
            ACEtk.ReferenceFrame.Laboratory: "LAB",
            ACEtk.ReferenceFrame.CentreOfMass: "COM",
        }
        source_frame = ace.frame_and_multiplicity_block.reference_frame(index)
        if source_frame not in frames:
            raise ValueError(f"MT {mt}: unsupported reference frame {source_frame}")
        frame = frames[source_frame]
    # Validate the shared-grid tail and write the basic reaction fields
    if offset < 0 or offset + len(values) != energy_count:
        raise ValueError(
            f"MT {mt}: cross section does not match the shared energy grid"
        )
    xs = group.create_group("xs")
    quantity(xs, "offset", offset)
    quantity(xs, "value", values, "barn")
    interpolation(xs, len(values))
    quantity(group, "reference-frame", frame)
    quantity(group, "Q-value", q_value, "eV")
    group.create_group("cross-products")
    if category == "capture":
        return

    # Write separate angular data when it is not part of a joint distribution
    angle = ace.angular_distribution_block.angular_distribution_data(index)
    if not isinstance(angle, ACEtk.continuous.DistributionGivenElsewhere):
        name = (
            "prompt-angular-cosine-distribution"
            if category == "fission"
            else "angular-cosine-distribution"
        )
        write_angular_distribution(group.create_group(name), angle)
    # Add the neutron production fields required by the transport category
    if category == "inelastic-scattering":
        quantity(
            group, "multiplicity", ace.frame_and_multiplicity_block.multiplicity(index)
        )
        source = ace.energy_distribution_block.energy_distribution_data(index)
        write_mixture_distribution(group.create_group("energy-distribution"), source)
    elif category == "fission":
        source = ace.energy_distribution_block.energy_distribution_data(index)
        write_energy_distribution(
            group.create_group("prompt-energy-distribution"), source
        )
        write_fission(group, ace)


def write_mixture_distribution(group, source):
    """Write ACE spectra as a mixture with per-component probability Data."""
    if isinstance(source, ACEtk.continuous.MultiDistributionData):
        count = source.number_distributions
        spectra = [source.distribution(i + 1) for i in range(count)]
        selections = [source.probability(i + 1) for i in range(count)]
        if not spectra:
            raise ValueError("A spectrum mixture must contain at least one component")
        values = [np.asarray(selection.probabilities) for selection in selections]
        if all(len(value) and np.all(value == value[0]) for value in values):
            total = sum(value[0] for value in values)
            if not 1.0 - PROBABILITY_TOLERANCE <= total <= 1.0 + PROBABILITY_TOLERANCE:
                raise ValueError("Spectrum probabilities must sum to one")
    else:
        spectra, selections = [source], [None]
    group.attrs["type"] = "mixture"
    components = group.create_group("components")
    for i, (spectrum, selection) in enumerate(zip(spectra, selections)):
        component = components.create_group(str(i))
        _write_distribution_probability(
            component.create_group("probability"), selection
        )
        write_energy_distribution(component.create_group("distribution"), spectrum)


def _write_distribution_probability(group, source):
    """Write a component probability as constant or tabulated H1 Data."""
    if source is None:
        values, energy = np.array([1.0]), np.array([0.0])
    else:
        values = np.asarray(source.probabilities, dtype=float)
        energy = np.asarray(source.energies, dtype=float) * EV_PER_MEV
    if (
        values.ndim != 1
        or energy.shape != values.shape
        or not len(values)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values > 1.0 + PROBABILITY_TOLERANCE)
        or not np.all(np.isfinite(energy))
        or np.any(np.diff(energy) < 0)
    ):
        raise ValueError(
            "Spectrum probabilities require finite ordered energies and values in [0, 1]"
        )
    if np.all(values == values[0]):
        group.attrs["type"] = "polynomial"
        quantity(group, "coefficients", [values[0]])
        return
    if energy[-1] <= energy[0]:
        raise ValueError(
            "A varying probability table requires a positive-width energy grid"
        )
    laws, boundaries = _interpolation_regions(source.interpolation_data, len(energy))
    start = 0
    for law, end in zip(laws, boundaries):
        active = energy[start + 1 : end] > energy[start : end - 1]
        if law in ("semilog-x", "log") and np.any(energy[start : end - 1][active] <= 0):
            raise ValueError("Logarithmic probability coordinates must be positive")
        if law in ("semilog-y", "log") and (
            np.any(values[start : end - 1][active] <= 0)
            or np.any(values[start + 1 : end][active] <= 0)
        ):
            raise ValueError("Logarithmic probability ordinates must be positive")
        start = end - 1
    group.attrs["type"] = "table"
    quantity(group, "grid", energy, "eV")
    quantity(group, "value", values)
    interpolation(group, len(energy), laws, boundaries)


# ======================================================================================
# Fission multiplicities and delayed-neutron groups
# ======================================================================================


def write_multiplicity(group, source):
    """Write an ACE fission multiplicity as polynomial or tabulated H1 Data."""
    if source.type == 1:
        # Convert each polynomial coefficient for its corresponding power of eV
        group.attrs["type"] = "polynomial"
        coefficients = np.asarray(source.coefficients)
        units = [
            "1" if i == 0 else "1/eV" if i == 1 else f"1/eV^{i}"
            for i in range(len(coefficients))
        ]
        quantity(
            group,
            "coefficients",
            coefficients / EV_PER_MEV ** np.arange(len(coefficients)),
            units if len(coefficients) > 1 else None,
        )
    elif source.type == 2:
        # Convert the incident-energy grid and preserve the yield table's interpolation law
        group.attrs["type"] = "table"
        quantity(group, "grid", np.asarray(source.energies) * EV_PER_MEV, "eV")
        quantity(group, "value", source.multiplicities)
        write_interpolation(group, source.interpolation_data, len(source.energies))
    else:
        raise ValueError(
            f"Unsupported fission multiplicity representation: {source.type}"
        )


def write_fission(group, ace):
    """Copy ACE prompt, delayed, and precursor data into a fission reaction."""
    # Require an unambiguous prompt yield before writing production data
    prompt = ace.fission_multiplicity_block
    delayed = ace.delayed_fission_multiplicity_block
    if prompt is None:
        raise ValueError("Fission reaction has no fission multiplicity data")
    if not prompt.has_prompt_and_total_fission_multiplicity and delayed is not None:
        raise ValueError(
            "Total-only fission multiplicity with delayed data requires prompt-yield reconstruction"
        )
    # Copy shared source yields into the individual fission reaction
    group.attrs["fission-production-source"] = "shared ACE nuclide data"
    write_multiplicity(group.create_group("prompt-multiplicity"), prompt.multiplicity)
    delayed_yield = group.create_group("delayed-multiplicity")
    if delayed is None:
        delayed_yield.attrs["type"] = "polynomial"
        quantity(delayed_yield, "coefficients", np.array([0.0]))
    else:
        write_multiplicity(delayed_yield, delayed.multiplicity)
    # Keep precursor fractions, decay rates, and energy spectra aligned by index
    neutrons = group.create_group("delayed-neutrons")
    source = ace.delayed_neutron_precursor_block
    fractions, decay_rates = [], []
    if source is not None:
        for i in range(source.number_delayed_precursors):
            precursor = source.precursor_group_data(i + 1)
            if (
                precursor.number_interpolation_regions != 0
                or len(precursor.probabilities) != 2
                or precursor.probabilities[0] != precursor.probabilities[1]
            ):
                raise ValueError("Delayed-neutron precursor fractions must be constant")
            fractions.append(precursor.probabilities[0])
            decay_rates.append(precursor.decay_constant)
            distribution = (
                ace.delayed_neutron_energy_distribution_block.energy_distribution_data(
                    i + 1
                )
            )
            if not isinstance(
                distribution, ACEtk.continuous.OutgoingEnergyDistributionData
            ):
                raise ValueError("Unsupported delayed-neutron energy distribution")
            write_energy_distribution(
                neutrons.create_group(f"energy-distribution-{i}"), distribution
            )
    quantity(neutrons, "fractions", np.asarray(fractions))
    quantity(neutrons, "decay-rates", np.asarray(decay_rates), "1/s")


# ======================================================================================
# Proton reactions and secondary products
# ======================================================================================


@dataclass
class ProtonReaction:
    """Selected proton reaction with its ACE index and output classification."""

    MT: int
    index: int
    group: str
    multiplicity: int = 0
    proton_yield: tuple | None = None


PROTON_REDUNDANT_MTS = frozenset({1, 3, 10, 101})
PROTON_SUM_MT_PARTIALS = {
    4: range(50, 92),
    16: range(875, 892),
    103: range(600, 650),
    104: range(650, 700),
    105: range(700, 750),
    106: range(750, 800),
    107: range(800, 850),
}
PROTON_MAX_MT = 891
PROTON_ELEMENT_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
PROTON_SYMBOL_TO_Z = {
    symbol: z for z, symbol in enumerate(PROTON_ELEMENT_SYMBOLS, start=1)
}
ACE_PROTON_PRODUCT_NAMES = {
    1: "neutron",
    9: "proton",
    31: "deuteron",
    32: "triton",
    33: "helion",
    34: "alpha",
}
ACE_PHOTON_MT_FACTOR = 1000


@dataclass
class ProtonACEData:
    """Parsed ACE data and output metadata for one proton target."""

    table: object
    Z: int
    A: int
    state: int
    symbol: str
    nuclide: str
    output_name: str
    pstar_path: Path | None
    sources: dict
    reactions: list
    skipped_mts: set
    required_paths: list


def decode_ace_zaid(zaid):
    """Decode atomic number, mass number, and isomer state from a proton ZAID."""
    za = int(zaid.strip().split(".")[0])
    state = 0
    if za >= 600000:
        state = (za % 1000) // 400
        za -= state * 400
    return za // 1000, za % 1000, state


def prepare_proton_ace(ace_path, pstar_dir=None, *, source_library=None):
    """Parse proton ACE data and prepare target, reaction, and output metadata.

    The result includes provenance, an optional matching PSTAR file, and the
    HDF5 paths used to determine whether an existing output can be reused.
    ``source_library`` optionally overrides the ACE provenance label while
    preserving the original header metadata.
    """
    table = ACEtk.ContinuousEnergyTable.from_file(str(ace_path))
    Z, A, state = decode_ace_zaid(table.header.zaid)
    if not 1 <= Z <= len(PROTON_ELEMENT_SYMBOLS):
        raise ValueError(f"Unsupported Z={Z} in {ace_path}")
    symbol = PROTON_ELEMENT_SYMBOLS[Z - 1]
    nuclide = f"{symbol}{A}" if state == 0 else f"{symbol}{A}m{state}"
    pstar_path = Path(pstar_dir) / f"{symbol}.txt" if pstar_dir else None
    if pstar_path is not None and not pstar_path.is_file():
        pstar_path = None

    reactions, skipped_mts = classify_proton_reactions(table)
    required_paths = [
        "atomic-number",
        "mass-number",
        "atomic-weight-ratio",
        "excitation-level",
        "fissionable",
        "radiation-length",
        "xs-energy-grid",
        "reactions",
        "reactions/MT-002/xs/value",
        "reactions/MT-002/xs/offset",
        "reactions/MT-002/reference-frame",
        "reactions/MT-002/Q-value",
        "reactions/MT-002/cross-products",
        "reactions/MT-002/angular-cosine-distribution",
    ]
    for reaction in reactions:
        base = f"reactions/MT-{reaction.MT:03}"
        required_paths.extend(
            f"{base}/{field}"
            for field in (
                "xs/value",
                "xs/offset",
                "reference-frame",
                "Q-value",
                "cross-products",
            )
        )
        if reaction.group == "inelastic_scattering":
            required_paths.extend(
                (f"{base}/multiplicity", f"{base}/energy-distribution")
            )
    if pstar_path is not None:
        required_paths.extend(("stopping-power/grid", "stopping-power/value"))

    return ProtonACEData(
        table=table,
        Z=Z,
        A=A,
        state=state,
        symbol=symbol,
        nuclide=nuclide,
        output_name=f"proton-{nuclide}.h5",
        pstar_path=pstar_path,
        sources=read_proton_source_libraries(
            table, pstar_path is not None, source_library=source_library
        ),
        reactions=reactions,
        skipped_mts=skipped_mts,
        required_paths=required_paths,
    )


def ace_reference_frame_name(frame):
    """Map an ACE reference frame to its MC/DC name, ``LAB`` or ``COM``."""
    frames = {
        ACEtk.ReferenceFrame.Laboratory: "LAB",
        ACEtk.ReferenceFrame.CentreOfMass: "COM",
    }
    if frame not in frames:
        raise ValueError(f"Unknown ACE reference frame {frame!r}")
    return frames[frame]


def read_proton_source_libraries(ace, include_pstar, *, source_library=None):
    """Build shared ACE provenance and add a separate PSTAR record when used."""
    sources = read_source_libraries(ace, source_library=source_library)
    if include_pstar:
        if "PSTAR" in sources:
            raise ValueError("ACE source-library label conflicts with PSTAR provenance")
        sources["PSTAR"] = {
            "title": "NIST PSTAR stopping-power tables",
            "version": "",
            "release-date": "",
            "reference": "NIST PSTAR",
            "comments": "",
            "format": "PSTAR",
            "format-version": "",
            "processing-date": "",
        }
    return sources


def read_proton_yield(data, what):
    """Read a linear-linear ACE yield table as energy and multiplicity arrays."""
    offset = (
        2
        if isinstance(data, ACEtk.continuous.TabulatedSecondaryParticleMultiplicity)
        else 0
    )
    if not data.interpolation_data.is_linear_linear:
        regions = int(data.NB)
        boundaries = [int(data.xss(offset + i)) for i in range(2, regions + 2)]
        interpolants = [
            int(data.xss(offset + i)) for i in range(regions + 2, 2 * regions + 2)
        ]
        raise ValueError(
            f"{what}: yield table is not linear-linear "
            f"(NBT={boundaries}, INT={interpolants})"
        )
    regions = int(data.NB)
    count = int(data.NE)
    energy_start = offset + 2 * regions + 3
    value_start = energy_start + count
    energy = np.array(
        [data.xss(i) for i in range(energy_start, value_start)], dtype=np.float64
    )
    value = np.array(
        [data.xss(i) for i in range(value_start, value_start + count)],
        dtype=np.float64,
    )
    if len(energy) < 2:
        raise ValueError(f"{what}: yield table contains fewer than two points")
    if np.any(np.diff(energy) < 0.0):
        raise ValueError(f"{what}: yield energies are not non-decreasing")
    if np.any(value < 0.0):
        raise ValueError(f"{what}: negative yield")
    return energy, value


def _write_proton_yield(group, name, energy, value):
    """Write an incident-energy-dependent yield as MC/DC table Data."""
    energy = np.asarray(energy, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if (
        energy.ndim != 1
        or energy.shape != value.shape
        or len(energy) < 2
        or not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(value))
        or np.any(np.diff(energy) < 0)
        or np.any(value < 0)
    ):
        raise ValueError(f"{name}: invalid yield table")
    same_as_previous = np.zeros(len(energy), dtype=bool)
    same_as_previous[1:] = (energy[1:] == energy[:-1]) & (value[1:] == value[:-1])
    energy, value = energy[~same_as_previous], value[~same_as_previous]
    if len(energy) < 2 or not np.any(np.diff(energy) > 0):
        raise ValueError(
            f"{name}: yield table must contain at least two distinct energies"
        )

    table = group.create_group(name)
    table.attrs["type"] = "table"
    quantity(table, "grid", energy * EV_PER_MEV, "eV")
    quantity(table, "value", value)
    interpolation(table, len(energy))


def _is_joint_proton_spectrum(source):
    """Return whether an ACE spectrum also describes its angular distribution."""
    continuous = ACEtk.continuous
    joint_types = (
        continuous.KalbachMannDistributionData,
        continuous.EnergyAngleDistributionData,
        continuous.NBodyPhaseSpaceDistribution,
    )
    if isinstance(source, continuous.MultiDistributionData):
        joint = [
            _is_joint_proton_spectrum(source.distribution(index))
            for index in range(1, source.number_distributions + 1)
        ]
        if any(joint) and not all(joint):
            raise ValueError(
                "A product mixture cannot combine joint and independent angle-energy laws"
            )
        return all(joint)
    return isinstance(source, joint_types)


def _write_proton_cross_product(
    parent, name, *, frame, yield_energy, yield_value, energy, angle
):
    """Write a secondary-particle product and its yield and distributions."""
    continuous = ACEtk.continuous
    product = parent.create_group(str(len(parent)))
    quantity(product, "particle", name)
    quantity(product, "reference-frame", frame)
    _write_proton_yield(product, "yield", yield_energy, yield_value)
    quantity(product, "decay-rate", np.inf, "1/s")

    distribution = product.create_group("angle-energy-distribution")
    if _is_joint_proton_spectrum(energy):
        write_energy_distribution(distribution, energy)
        return
    if isinstance(angle, continuous.DistributionGivenElsewhere):
        raise ValueError(
            f"{parent.parent.name}/{name}: angle is not included in its energy distribution"
        )
    distribution.attrs["type"] = "independent-angle-energy"
    write_angular_distribution(
        distribution.create_group("angular-cosine-distribution"), angle
    )
    write_energy_distribution(distribution.create_group("energy-distribution"), energy)


def classify_proton_reactions(ace):
    """Select proton reactions and classify capture versus inelastic channels.

    Returns the selected records and MTs omitted as redundant or unsupported.
    Raises ``ValueError`` for unsupported fission or invalid yield data.
    """
    numbers = ace.reaction_number_block
    multiplicities = ace.frame_and_multiplicity_block
    distributions = ace.energy_distribution_block
    count = multiplicities.number_reactions
    present = {numbers.MT(index) for index in range(1, count + 1)}

    skipped = set(PROTON_REDUNDANT_MTS)
    for aggregate, partials in PROTON_SUM_MT_PARTIALS.items():
        if aggregate in present and present.intersection(partials):
            skipped.add(aggregate)
    skipped.update(mt for mt in present if mt > PROTON_MAX_MT)
    if present.intersection((18, 19, 20, 21, 38)) or (
        ace.fission_multiplicity_block is not None
    ):
        raise ValueError("Fission data in a proton ACE file are not supported")

    reactions = []
    for index in range(1, count + 1):
        mt = numbers.MT(index)
        if mt == 2 or mt in skipped:
            continue
        multiplicity = multiplicities.multiplicity(index)
        if multiplicity == 0:
            reactions.append(ProtonReaction(mt, index, "capture"))
        elif multiplicity > 100:
            if index > distributions.number_reactions:
                raise ValueError(
                    f"MT-{mt:03}: energy-dependent multiplicity but no DLW entry"
                )
            data = distributions.multiplicity_data(index)
            if not isinstance(data, ACEtk.continuous.TabulatedMultiplicity):
                raise ValueError(
                    f"MT-{mt:03}: TYR={multiplicity} but multiplicity data is "
                    f"{type(data).__name__}"
                )
            reactions.append(
                ProtonReaction(
                    mt,
                    index,
                    "inelastic_scattering",
                    proton_yield=read_proton_yield(data, f"proton yield of MT-{mt:03}"),
                )
            )
        else:
            reactions.append(
                ProtonReaction(mt, index, "inelastic_scattering", multiplicity)
            )
    return reactions, skipped


def _write_proton_reaction(group, ace, reaction):
    """Write one proton reaction's cross section and channel data."""
    index = reaction.index
    group.attrs["ENDF-MT"] = reaction.MT
    group.attrs["category"] = reaction.group.replace("_", "-")

    cross_sections = ace.cross_section_block
    xs = group.create_group("xs")
    quantity(xs, "value", cross_sections.cross_sections(index), "barn")
    quantity(xs, "offset", cross_sections.energy_index(index) - 1)
    interpolation(xs, len(xs["value"]))
    quantity(
        group,
        "Q-value",
        ace.reaction_qvalue_block.q_value(index) * EV_PER_MEV,
        "eV",
    )
    quantity(
        group,
        "reference-frame",
        ace_reference_frame_name(
            ace.frame_and_multiplicity_block.reference_frame(index)
        ),
    )
    group.create_group("cross-products")
    if reaction.group == "capture":
        return

    if reaction.proton_yield is None:
        quantity(group, "multiplicity", reaction.multiplicity)
    else:
        _write_proton_yield(group, "multiplicity", *reaction.proton_yield)
    angle = ace.angular_distribution_block.angular_distribution_data(index)
    if not isinstance(angle, ACEtk.continuous.DistributionGivenElsewhere):
        write_angular_distribution(
            group.create_group("angular-cosine-distribution"), angle
        )
    write_energy_distribution(
        group.create_group("energy-distribution"),
        ace.energy_distribution_block.energy_distribution_data(index),
    )


def _write_proton_cross_products(ace, reaction_groups, skipped_mts, verbose):
    """Write proton ACE secondary-particle and photon yields and distributions."""
    continuous = ACEtk.continuous
    written = 0

    for particle_index in range(1, ace.NTYPE + 1):
        ace_id = ace.PTYPE.particle_type(particle_index)
        if ace_id not in ACE_PROTON_PRODUCT_NAMES:
            raise ValueError(f"Unknown ACE secondary-particle id {ace_id}")
        if ace_id == 9:
            raise ValueError("Proton listed as a secondary particle type")
        name = ACE_PROTON_PRODUCT_NAMES[ace_id]
        count = ace.NTRO.NP(particle_index)
        reaction_numbers = ace.MTRH(particle_index)
        particle_multiplicity = ace.TYRH(particle_index)
        yields = ace.SIGH(particle_index)
        energy = ace.DLWH(particle_index)
        angles = ace.ANDH(particle_index)
        sizes = (
            reaction_numbers.number_reactions,
            particle_multiplicity.number_reactions,
            len(yields.data),
            energy.number_reactions,
            angles.number_reactions,
        )
        if any(size != count for size in sizes):
            raise ValueError(
                f"{name}: inconsistent block sizes {sizes}, expected {count} each"
            )

        for index in range(1, count + 1):
            mt = reaction_numbers.MT(index)
            what = f"{name} from MT-{mt:03}"
            if particle_multiplicity.multiplicity(index) != 1:
                raise ValueError(
                    f"{what}: TYRH multiplicity "
                    f"{particle_multiplicity.multiplicity(index)} (expected 1)"
                )
            yield_data = yields.data[index - 1]
            if yield_data.MT != mt or yield_data.MFTYPE != 12:
                raise ValueError(f"{what}: unsupported SIGH yield record")
            group = _proton_products_group(
                mt, reaction_groups, skipped_mts, what, verbose
            )
            if group is None:
                continue
            yield_energy, yield_value = read_proton_yield(yield_data, what)
            _write_proton_cross_product(
                group,
                name,
                frame=ace_reference_frame_name(
                    particle_multiplicity.reference_frame(index)
                ),
                yield_energy=yield_energy,
                yield_value=yield_value,
                energy=energy.energy_distribution_data(index),
                angle=angles.angular_distribution_data(index),
            )
            written += 1
            if verbose:
                print(
                    f"      MT-{mt:03} {name}: yield "
                    f"{yield_value.min():g}..{yield_value.max():g}"
                )

    count = ace.NTRP
    if count:
        reactions = ace.MTRP
        yields = ace.SIGP
        energy = ace.DLWP
        angles = ace.ANDP
        sizes = (
            reactions.number_reactions,
            len(yields.data),
            energy.number_reactions,
            angles.number_reactions,
        )
        if any(size != count for size in sizes):
            raise ValueError(
                f"photon: inconsistent block sizes {sizes}, expected {count} each"
            )
        for index in range(1, count + 1):
            mt, photon_index = divmod(reactions.MT(index), ACE_PHOTON_MT_FACTOR)
            name = "photon" if photon_index == 1 else f"photon_{photon_index}"
            what = f"{name} from MT-{mt:03}"
            yield_data = yields.data[index - 1]
            if not isinstance(
                yield_data, continuous.TabulatedSecondaryParticleMultiplicity
            ):
                raise ValueError(
                    f"{what}: SIGP entry is a {type(yield_data).__name__}, "
                    "not a yield table"
                )
            if yield_data.MT != mt or yield_data.MFTYPE not in (12, 16):
                raise ValueError(f"{what}: unexpected SIGP yield record")
            group = _proton_products_group(
                mt, reaction_groups, skipped_mts, what, verbose
            )
            if group is None:
                continue
            frame = energy.reference_frame(index)
            yield_energy, yield_value = read_proton_yield(yield_data, what)
            _write_proton_cross_product(
                group,
                name,
                frame="LAB" if frame is None else ace_reference_frame_name(frame),
                yield_energy=yield_energy,
                yield_value=yield_value,
                energy=energy.energy_distribution_data(index),
                angle=angles.angular_distribution_data(index),
            )
            written += 1
            if verbose:
                print(
                    f"      MT-{mt:03} {name}: yield "
                    f"{yield_value.min():g}..{yield_value.max():g}"
                )
    return written


def _proton_products_group(mt, reaction_groups, skipped_mts, what, verbose):
    """Return a product's parent group, or skip it when its MT was omitted."""
    if mt in reaction_groups:
        return reaction_groups[mt]["cross-products"]
    if mt in skipped_mts:
        if verbose:
            print(f"  [note] {what}: MT-{mt:03} is a redundant sum; skipped")
        return None
    raise ValueError(f"{what}: MT-{mt:03} has no reaction group in the output")


def write_proton_reactions(file, ace, reactions, skipped_mts, verbose=False):
    """Write the proton energy grid, reactions, and secondary products."""
    energy = np.asarray(ace.principal_cross_section_block.energies) * EV_PER_MEV
    if (
        energy.ndim != 1
        or len(energy) < 2
        or not np.all(np.isfinite(energy))
        or np.any(np.diff(energy) < 0)
    ):
        raise ValueError("Invalid principal proton energy grid")
    quantity(file, "xs-energy-grid", energy, "eV")
    root = file.create_group("reactions", track_order=True)

    elastic = root.create_group("MT-002", track_order=True)
    elastic.attrs["ENDF-MT"] = 2
    elastic.attrs["category"] = "elastic-scattering"
    xs = elastic.create_group("xs")
    quantity(xs, "value", ace.principal_cross_section_block.elastic, "barn")
    quantity(xs, "offset", 0)
    interpolation(xs, len(energy))
    quantity(elastic, "reference-frame", "COM")
    quantity(elastic, "Q-value", 0.0, "eV")
    elastic.create_group("cross-products")
    angle = ace.angular_distribution_block.angular_distribution_data(0)
    if isinstance(angle, ACEtk.continuous.DistributionGivenElsewhere):
        raise ValueError(
            "MT-002 elastic scattering has no explicit angular distribution"
        )
    write_angular_distribution(
        elastic.create_group("angular-cosine-distribution"), angle
    )
    reaction_groups = {2: elastic}

    for reaction in reactions:
        group = root.create_group(f"MT-{reaction.MT:03}", track_order=True)
        _write_proton_reaction(group, ace, reaction)
        reaction_groups[reaction.MT] = group
    product_count = _write_proton_cross_products(
        ace, reaction_groups, skipped_mts, verbose
    )
    if verbose:
        print(f"  Reactions written: {len(reactions) + 1}")
        print(f"  Redundant or unsupported MTs skipped: {sorted(skipped_mts)}")
        print(f"  Cross-products written: {product_count}")


def write_proton_ace_target_properties(file, data, radiation_length):
    """Write target identity and physical properties for a proton ACE file."""
    quantity(file, "atomic-number", data.Z)
    quantity(file, "mass-number", data.A)
    quantity(file, "atomic-weight-ratio", data.table.atomic_weight_ratio)
    quantity(file, "excitation-level", data.state)
    quantity(file, "fissionable", False)
    quantity(file, "radiation-length", radiation_length, "g/cm2")


# ======================================================================================
# Angular distributions
# ======================================================================================


@_distribution_errors
def write_angular_distribution(group, source):
    """Convert ACE angular data to isotropic, table, or multi-table H1 form.

    Supports isotropic, equiprobable-bin, and tabulated cosine distributions.
    Raises ``ValueError`` for unsupported types or invalid source tables.
    """
    # Represent isotropy without constructing an artificial cosine table
    if isinstance(source, ACEtk.continuous.FullyIsotropicDistribution):
        group.attrs["type"] = "isotropic"
        return
    # Collect conditional cosine tables, retaining each local interpolation law.
    tables = []
    for i, table in enumerate(source.distributions):
        kind = source.distribution_type(i + 1)
        if kind == ACEtk.AngularDistributionType.Isotropic:
            tables.append(([-1.0, 1.0], [0.5, 0.5], ["histogram"], None))
        elif kind == ACEtk.AngularDistributionType.Equiprobable:
            cosines = np.asarray(table.cosines, dtype=float)
            widths = np.diff(cosines)
            if (
                cosines.shape != (33,)
                or not np.all(np.isfinite(cosines))
                or np.any(cosines < -1.0)
                or np.any(cosines > 1.0)
                or np.any(widths <= 0.0)
            ):
                raise ValueError(
                    f"Angular entry {i}: equiprobable bins require 33 finite, "
                    "strictly increasing cosine boundaries within [-1, 1]"
                )
            density = np.r_[1.0 / (32 * widths), 0.0]
            tables.append((cosines, density, ["histogram"], None))
        elif kind == ACEtk.AngularDistributionType.Tabulated:
            tables.append(
                (
                    table.cosines,
                    table.pdf,
                    [interpolation_law(table.interpolation)],
                    None,
                )
            )
        else:
            raise ValueError(f"Angular entry {i}: unsupported distribution type {kind}")
    if len(tables) == 1:
        if source.distribution_type(1) == ACEtk.AngularDistributionType.Isotropic:
            group.attrs["type"] = "isotropic"
        else:
            cosines, density, laws, _ = tables[0]
            pdf_table(group, cosines, density, law=laws[0], unit=None)
        return
    multi_table(
        group,
        np.asarray(source.incident_energies) * EV_PER_MEV,
        tables,
        unit=None,
    )


# ======================================================================================
# Energy distributions
# ======================================================================================


@_distribution_errors
def write_energy_distribution(group, source):
    """Convert supported ACE energy and joint energy-angle laws to H1 form.

    Handles analytic, tabulated, correlated, and mixed spectra; converts
    energy units and preserves source interpolation. Unsupported data raise
    ``ValueError``.
    """
    if isinstance(source, ACEtk.continuous.MultiDistributionData):
        write_mixture_distribution(group, source)
    elif isinstance(source, ACEtk.continuous.LevelScatteringDistribution):
        # Preserve the deterministic relationship between incident and outgoing energy
        group.attrs["type"] = "level-scattering"
        quantity(group, "C1", source.C1 * EV_PER_MEV, "eV")
        quantity(group, "C2", source.C2)
    elif isinstance(
        source,
        (
            ACEtk.continuous.EvaporationSpectrum,
            ACEtk.continuous.SimpleMaxwellianFissionSpectrum,
        ),
    ):
        # Convert analytic spectrum parameters and their temperature function
        group.attrs["type"] = (
            "evaporation"
            if isinstance(source, ACEtk.continuous.EvaporationSpectrum)
            else "maxwellian"
        )
        quantity(
            group, "restriction-energy", source.restriction_energy * EV_PER_MEV, "eV"
        )
        temperature = group.create_group("temperature")
        temperature.attrs["type"] = "table"
        quantity(temperature, "grid", np.asarray(source.energies) * EV_PER_MEV, "eV")
        quantity(
            temperature, "value", np.asarray(source.temperatures) * EV_PER_MEV, "eV"
        )
        write_interpolation(
            temperature, source.interpolation_data, len(source.energies)
        )
    elif isinstance(source, ACEtk.continuous.NBodyPhaseSpaceDistribution):
        # Preserve the dimensionless phase-space density and kinematic parameters.
        count = source.number_particles
        mass_ratio = float(source.total_mass_ratio)
        values = np.asarray(source.values)
        if count < 3 or not np.isfinite(mass_ratio) or mass_ratio <= 1:
            raise ValueError("Invalid N-body particle count or total mass ratio")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("N-body energy fractions must lie in [0, 1]")
        pdf_table(
            group,
            values,
            source.pdf,
            law=interpolation_law(source.interpolation),
            unit=None,
        )
        group.attrs["type"] = "n-body"
        quantity(group, "number-particles", count)
        quantity(group, "total-mass-ratio", mass_ratio)
    elif isinstance(
        source,
        (
            ACEtk.continuous.OutgoingEnergyDistributionData,
            ACEtk.continuous.KalbachMannDistributionData,
            ACEtk.continuous.EnergyAngleDistributionData,
        ),
    ):
        # Establish the incident-energy grid shared by the outgoing distributions
        incident_energy = np.asarray(source.incident_energies) * EV_PER_MEV
        laws, boundaries = _interpolation_regions(
            source.interpolation_data, len(incident_energy)
        )
        if isinstance(source, ACEtk.continuous.OutgoingEnergyDistributionData):
            # Pack all marginal densities while retaining their individual PDF rules.
            tables, discrete_values, masses = [], [], []
            for i in range(source.number_incident_energies):
                table = source.distribution(i + 1)
                count = table.number_discrete_photon_lines
                energy = np.asarray(table.outgoing_energies) * EV_PER_MEV
                probability = np.asarray(table.pdf)
                if count < 0 or count > len(energy):
                    raise ValueError("Invalid number of discrete energy lines")
                discrete_values.append(energy[:count])
                masses.append(probability[:count])
                # Line probabilities are dimensionless; only the continuum is a density.
                tables.append(
                    None
                    if count == len(energy)
                    else (
                        energy[count:],
                        probability[count:] / EV_PER_MEV,
                        [interpolation_law(table.interpolation)],
                        None,
                    )
                )
            discrete = any(len(values) for values in discrete_values)
            multi_table(
                group,
                incident_energy,
                tables,
                discrete_value=discrete_values if discrete else None,
                pmf=masses if discrete else None,
                laws=laws,
                boundaries=boundaries,
            )
        else:
            # Pack both levels of correlated tables in their original source order.
            kalbach = isinstance(source, ACEtk.continuous.KalbachMannDistributionData)
            group.attrs["type"] = "kalbach-mann" if kalbach else "energy-angle-table"
            quantity(group, "energy", incident_energy, "eV")
            incident_interpolation(group, incident_energy, laws, boundaries)
            tables = group.create_group("tables")
            outgoing, angular = [], []
            if kalbach:
                parameters = {
                    name: tables.create_group(name)
                    for name in ("precompound-factor", "angular-slope")
                }
            for i, source_table in enumerate(source.distributions):
                energy = np.asarray(source_table.outgoing_energies) * EV_PER_MEV
                law = interpolation_law(source_table.interpolation)
                outgoing.append(
                    (energy, np.asarray(source_table.pdf) / EV_PER_MEV, [law], None)
                )
                if kalbach:
                    # Keep general Data representations for the two angular parameters.
                    for name, values in (
                        (
                            "precompound-factor",
                            source_table.precompound_fraction_values,
                        ),
                        (
                            "angular-slope",
                            source_table.angular_distribution_slope_values,
                        ),
                    ):
                        data = parameters[name].create_group(str(i))
                        data.attrs["type"] = "table"
                        quantity(data, "grid", energy, "eV")
                        quantity(data, "value", values)
                        interpolation(data, len(energy), [law])
                else:
                    cosines = list(source_table.distributions)
                    if len(cosines) != len(energy):
                        raise ValueError("Each outgoing energy requires a cosine table")
                    angular.extend(
                        (
                            cosine.cosines,
                            cosine.pdf,
                            [interpolation_law(cosine.interpolation)],
                            None,
                        )
                        for cosine in cosines
                    )
            pdf_tables(tables.create_group("outgoing-energy"), outgoing)
            if not kalbach:
                pdf_tables(
                    tables.create_group("cosine-distributions"), angular, unit=None
                )

    else:
        raise ValueError(f"Unsupported energy distribution: {type(source).__name__}")


# ======================================================================================
# Element symbols
# ======================================================================================

SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
}

Z_TO_SYMBOL = {value: key for key, value in SYMBOL_TO_Z.items()}
