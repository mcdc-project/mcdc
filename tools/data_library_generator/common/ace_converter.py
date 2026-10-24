"""Translate ACEtk objects into the MC/DC transport-data representations.

Projectile generators open ACE tables, select output filenames, and manage HDF5
file lifetimes. This module interprets source physics records, converts units,
and maps them into the MC/DC layout through :mod:`.hdf5_writer`. Related
conversion functions are grouped by their role and projectile.

Source energies are in MeV. Energy coordinates and parameters are multiplied
by ``EV_PER_MEV``, while energy probability densities are divided by the same
factor to preserve probability. Polynomial coefficients require a separate
conversion for each power. Cross sections remain in barns, and dimensionless
quantities have no unit attribute.
"""

from dataclasses import dataclass
from numbers import Integral

import ACEtk
import numpy as np

from .hdf5_writer import interpolation, multi_table, pdf_table, pdf_tables, quantity

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
# Target identity
# ======================================================================================


@dataclass(frozen=True)
class Target:
    """Immutable target identity used to coordinate the three output files.

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
        Nominal source-library temperature in kelvin, obtained from the Lib81
        suffix mapping rather than its rounded ACE thermal-energy value.
    """

    symbol: str
    atomic_number: int
    mass_number: int
    excitation_level: int
    temperature: float

    @property
    def nuclide(self):
        """Return the nuclide identifier used for filenames and source matching.

        Returns
        -------
        str
            Element symbol followed by mass number, with ``m`` and the state
            index appended for an excited state, for example ``Co58m1``.
        """
        suffix = f"m{self.excitation_level}" if self.excitation_level else ""
        return f"{self.symbol}{self.mass_number}{suffix}"

    @property
    def filename(self):
        """Return the temperature-file basename, for example ``H2-293.6K.h5``.

        Returns
        -------
        str
            Nuclide identifier, a hyphen, the nominal temperature converted
            to a string, and the suffix ``K.h5``. No directory is included.
        """
        return f"{self.nuclide}-{self.temperature}K.h5"


def read_target(ace):
    """Extract target identity and the nominal Lib81 temperature.

    Parameters
    ----------
    ace : ACEtk.ContinuousEnergyTable
        Parsed table exposing its ZAID, atomic number, mass number, and
        isomeric-state index through ACEtk.

    Returns
    -------
    Target
        Identity shared by the element, nuclide, and temperature outputs.

    Raises
    ------
    ValueError
        The suffix of ``ace.header.zaid`` is absent from
        ``ACE_TEMPERATURE_LIB81``.
    KeyError
        The atomic number is absent from the element-symbol lookup.

    Notes
    -----
    ACEtk supplies the interpreted nuclide and isomer identities; this
    function does not decode them from the ZAID's leading digits. Only the
    suffix is interpreted here. The nominal temperature controls filenames
    and the output target temperature. The original ACE thermal energy is
    preserved separately in source provenance by ``read_source_libraries``.
    """
    # Resolve the nominal library temperature used in output filenames
    suffix = ace.header.zaid.split(".")[-1]
    if suffix not in ACE_TEMPERATURE_LIB81:
        raise ValueError(f"Unknown Lib81 temperature suffix: {suffix}")
    return Target(
        Z_TO_SYMBOL[ace.atom_number],
        ace.atom_number,
        ace.mass_number,
        ace.isomeric_state,
        ACE_TEMPERATURE_LIB81[suffix],
    )


def read_source_libraries(ace, target, for_element=False, with_temperature=False):
    """Build source-library metadata for one output file without writing it.

    Parameters
    ----------
    ace : ACEtk.ContinuousEnergyTable
        Parsed table whose header provides the source title, comments,
        format version, processing date, and thermal energy.
    target : Target
        Interpreted target identity and nominal temperature.
    for_element : bool, optional
        Clear a title beginning with the nuclide identifier because that
        title does not describe the shared element file. Other applicable
        header values are retained unchanged.
    with_temperature : bool, optional
        Include the nominal ``temperature`` in K and original ACE ``kT``
        converted from MeV to eV. Projectile generators enable this only
        for nuclide-temperature files.

    Returns
    -------
    dict of str to dict
        Library identifiers mapped to scalar metadata values, suitable for
        ``hdf5_writer.stamp``. The supported identifier is ``"Lib81"``.
        String fields are ``title``, ``version``, ``release-date``,
        ``reference``, ``comments``, ``format``, ``format-version``, and
        ``processing-date``; optional temperature fields are numeric.

    Raises
    ------
    ValueError
        Neither the title nor comments identify Lib81 as a whitespace-
        separated token.

    Notes
    -----
    Library identity comes from header text, not the ZAID or input filename.
    The ACE header version and date describe the source format and processing;
    they are not assigned to the library's release version or release date.
    Unavailable release information and other missing strings are empty.
    A reference is extracted from the title's text following ``Ref.``.

    This function does not create ``conversion-time``. The HDF5 writer records
    that timestamp when it actually writes the source record, allowing
    skipped conversions to leave provenance unchanged.
    """
    # Identify the source library from its header text
    header = ace.header
    title = str(getattr(header, "title", "") or "")
    comments = str(getattr(header, "comments", "") or "")
    if "Lib81" not in f"{title} {comments}".split():
        raise ValueError(
            "ACE header does not identify a supported source library (Lib81)"
        )
    # Collect provenance applicable to the target file
    source = {
        "title": (
            "" if for_element and title.startswith(f"{target.nuclide} ") else title
        ),
        "version": "",
        "release-date": "",
        "reference": title.split("Ref.", 1)[1].strip() if "Ref." in title else "",
        "comments": comments,
        "format": "ACE",
        "format-version": str(getattr(header, "version", "") or ""),
        "processing-date": str(getattr(header, "date", "") or ""),
    }
    # Attach source temperature information to temperature-specific records
    if with_temperature:
        source["temperature"] = target.temperature
        source["kT"] = float(header.temperature) * EV_PER_MEV
    return {"Lib81": source}


# ======================================================================================
# Interpolation
# ======================================================================================


def interpolation_law(code):
    """Translate an ACE interpolation code into an MC/DC law name.

    Parameters
    ----------
    code : int
        ACE code: 1 for histogram, 2 for linear-linear, 3 for linear in
        log(x), 4 for linear in log(y), or 5 for log-log interpolation.

    Returns
    -------
    str
        ``histogram``, ``linear``, ``semilog-x``, ``semilog-y``, or ``log``.

    Raises
    ------
    ValueError
        The code cannot be converted to an integer or has no supported law.

    Notes
    -----
    This converts a rule identifier only; it does not interpolate values or
    verify that coordinates satisfy a logarithmic law's positivity domain.
    """
    try:
        return ACE_INTERPOLATION[int(code)]
    except KeyError:
        raise ValueError(f"Unsupported ACE interpolation law: {code}") from None


def write_interpolation(group, source, count):
    """Write ACE interpolation regions using MC/DC names and local indices.

    Parameters
    ----------
    group : h5py.Group
        Parent table representation in which to create ``interpolation``.
    source : object
        ACEtk interpolation data exposing ``interpolants`` and ``boundaries``.
        Source boundaries are one-based upper-point indices.
    count : int
        Number of points in the table being described, not necessarily the
        length of a projectile's full shared energy grid.

    Raises
    ------
    ValueError
        A source law is unsupported or the converted regions fail the HDF5
        writer's count, ordering, or endpoint checks.

    Notes
    -----
    The source's one-based upper-point indices already equal the exclusive
    point boundaries used by MC/DC, so the final boundary remains ``count``.
    An empty source law list means a single linear region.
    The child group contains ``laws`` and ``boundaries`` datasets; existing
    interpolation groups are not replaced. No table values are modified.
    """
    # Translate law names and retain point counts as exclusive region boundaries.
    laws = [interpolation_law(code) for code in source.interpolants]
    boundaries = np.asarray(source.boundaries, dtype=np.int64)
    if not laws:
        laws, boundaries = ["linear"], [count]
    interpolation(group, count, laws, boundaries)


# ======================================================================================
# Neutron reaction names, selection, and output
# ======================================================================================


def reaction_name(mt):
    """Return the MC/DC group name assigned to a neutron reaction MT.

    Parameters
    ----------
    mt : int
        ENDF reaction identifier.

    Returns
    -------
    str
        A descriptive name for explicitly mapped reactions, an
        ``inelastic-level-{i}`` name for MT 51--90, or ``reaction-{mt}``
        for an identifier without a dedicated name.

    Notes
    -----
    Naming does not select reactions or assign transport categories. An
    unmapped identifier can still have a stable fallback name. This name is
    an MC/DC key, not an original GNDS label; the MT value is preserved
    independently in the reaction's ``ENDF-MT`` attribute.
    """
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
    """Select and categorize the neutron reaction records to be written.

    Parameters
    ----------
    ace : ACEtk.ContinuousEnergyTable
        Parsed table with matching reaction-number and frame/multiplicity
        blocks. Those blocks use one-based reaction indices.

    Returns
    -------
    list of tuple
        Records ``(mt, index, category)`` sorted first by ``CATEGORY_ORDER``
        and then by MT. ``index`` is the ACEtk reaction index, except that
        ``(2, 0, "elastic-scattering")`` is an explicit elastic sentinel.
        Elastic cross sections are read from the principal block instead
        of the ordinary reaction cross-section list.

    Raises
    ------
    ValueError
        The reaction and multiplicity blocks have different lengths; total
        fission and chance-fission entries coexist; or a selected non-fission
        reaction does not have a nonnegative integer neutron multiplicity.

    Notes
    -----
    The selection is not a lossless enumeration of the ACE table. MT 1, 3,
    4, and 10 are omitted as aggregate quantities. MT 2 is skipped in the
    ordinary list because it has already been added through the sentinel.
    ``NEUTRON_REACTION_MT_RANGES`` admits MT 1--117 and 152--200 before
    aggregate filtering. MT 152--200 describe 49 independent channels,
    including (n,5n)--(n,8n) and mixed-particle emission. Other MT families,
    including inclusive production totals, auxiliary quantities, and
    resolved product branches, are excluded. The function does not verify
    that a retained aggregate covers each omitted branch.

    Fission MTs are recognized explicitly. Other selected reactions with
    zero outgoing neutrons are categorized as ``capture``, including
    charged-particle emission channels, while positive neutron multiplicity
    gives ``inelastic-scattering``. Within MT 152--200, the neutron-free
    channels are 155, 182, 191--193, and 197. Categories use the source
    neutron multiplicities; cross-species emissions do not change them.
    Energy-dependent multiplicity is not converted here. The function
    reads source records without modifying the ACE table or writing HDF5
    objects.
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
        Nominal temperature in K used for ``temperature`` and for computing
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
    once as ``projectiles/neutron/xs-energy-grid``. Reaction groups are
    created in selection order and populated by ``write_reaction``.
    Aggregate or auxiliary data excluded by ``retained_reactions`` is not
    written. Source thermal-energy provenance is supplied separately by
    ``read_source_libraries``; it may differ from computed ``kT`` through
    source rounding.

    Writes are direct and may leave a partially populated open file on
    failure. The caller uses ``hdf5_writer.atomic_file`` to publish the
    temperature file only after conversion succeeds. Element and nuclide
    companion files are not written by this function.
    """
    # Write the target thermodynamic state and shared neutron energy grid
    quantity(file, "temperature", target.temperature, "K")
    quantity(file, "kT", target.temperature * BOLTZMANN_K, "eV")
    neutron = file.create_group("projectiles/neutron")
    energy = np.asarray(ace.principal_cross_section_block.energies) * EV_PER_MEV
    quantity(neutron, "xs-energy-grid", energy, "eV")
    # Populate one group per retained reaction in transport category order
    reactions = neutron.create_group("reactions", track_order=True)
    for mt, index, category in retained_reactions(ace):
        if verbose:
            print(f"  MT {mt}: {reaction_name(mt)} ({category})")
        group = reactions.create_group(reaction_name(mt), track_order=True)
        write_reaction(group, ace, mt, index, category, len(energy))


def write_reaction(group, ace, mt, index, category, energy_count):
    """Populate one selected neutron reaction in its final HDF5 layout.

    Parameters
    ----------
    group : h5py.Group
        New reaction group, already named by the caller.
    ace : ACEtk.ContinuousEnergyTable
        Source table providing cross sections, Q-values, reference frames,
        angular distributions, energy spectra, and fission production data.
    mt : int
        Reaction identifier to store as ``ENDF-MT``.
    index : int
        One-based ACEtk reaction index, or zero for elastic scattering.
    category : str
        Selected ``elastic-scattering``, ``capture``,
        ``inelastic-scattering``, or ``fission`` category.
    energy_count : int
        Full length of the containing projectile's shared energy grid.

    Raises
    ------
    ValueError
        A source frame is unsupported, cross sections do not extend from
        their offset to the end of the shared grid, multiple prompt fission
        spectra are present, or a delegated conversion rejects source data.

    Notes
    -----
    Common output consists of category/source attributes, ``xs`` with a
    zero-based offset and linear interpolation, ``reference-frame``, a
    Q-value converted to eV, and an empty ``cross-products`` group. Cross
    sections remain in barns. Elastic data uses the principal block, offset
    zero, Q-value zero, and the COM frame.

    Capture stops after the common fields. Inelastic reactions add neutron
    multiplicity, spectrum-selection data, and numbered spectra. Fission
    adds a single prompt spectrum and reaction-local multiplicities and
    delayed-neutron data. Separate angular data is written only when ACE
    does not identify it as supplied through the energy-angle distribution.
    This function does not extract cross-species products or validate sums
    against total cross sections. It does not roll back partial group writes.
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
        components = write_spectrum_selection(group, source)
        for i, component in enumerate(components):
            write_energy_distribution(
                group.create_group(f"energy-spectrum-{i}"), component
            )
    elif category == "fission":
        source = ace.energy_distribution_block.energy_distribution_data(index)
        if isinstance(source, ACEtk.continuous.MultiDistributionData):
            if source.number_distributions != 1:
                raise ValueError("Multiple prompt fission spectra are not supported")
            source = source.distribution(1)
        write_energy_distribution(
            group.create_group("prompt-energy-distribution"), source
        )
        write_fission(group, ace)


def write_spectrum_selection(group, source):
    """Write incident-energy bins and probabilities for selecting spectra.

    Parameters
    ----------
    group : h5py.Group
        Reaction group receiving ``spectrum-probability-grid`` in eV and
        dimensionless ``spectrum-probability``.
    source : object
        A single ACEtk energy distribution or ``MultiDistributionData``
        containing distributions and their selection probability tables.

    Returns
    -------
    list
        Source distribution objects in the same order as probability-matrix
        columns. The caller writes these as ``energy-spectrum-{i}`` groups.

    Raises
    ------
    ValueError
        Selection probabilities vary without explicit histogram regions,
        use unsupported interpolation, or have different incident grids.

    Notes
    -----
    B bins and S spectra produce an edge grid of length B + 1 and a matrix
    of shape (B, S). A single distribution gets probability one. Constant
    selections use one bin with fixed edges 0 and 30 MeV, converted to eV;
    this envelope is not inferred from the source grid. Source records
    without interpolation regions must have constant values in each table.

    Energy-dependent selections must all be explicit histograms on exactly
    the same incident grid. The last probability at the upper endpoint is
    omitted because it does not define another bin. Source probabilities
    are copied without renormalization or additional positivity/sum checks.
    The returned distributions themselves are not written here.
    """
    # A constant selection is represented by one bin covering 0--30 MeV.
    grid = np.array([0.0, 30.0 * EV_PER_MEV])
    if not isinstance(source, ACEtk.continuous.MultiDistributionData):
        components, probabilities = [source], np.ones((1, 1))
    else:
        # Collect spectra and their incident-energy-dependent selection weights
        count = source.number_distributions
        components = [source.distribution(i + 1) for i in range(count)]
        selections = [source.probability(i + 1) for i in range(count)]
        values = [np.asarray(selection.probabilities) for selection in selections]
        # Represent constant weights by one bin and histogram weights by their bins
        if all(selection.number_interpolation_regions == 0 for selection in selections):
            if not all(np.all(value == value[0]) for value in values):
                raise ValueError(
                    "Spectrum probabilities require explicit histogram regions when they vary with energy"
                )
            probabilities = np.array([[value[0] for value in values]])
        elif all(
            selection.number_interpolation_regions > 0
            and all(code == 1 for code in selection.interpolants)
            for selection in selections
        ):
            grid = np.asarray(selections[0].energies) * EV_PER_MEV
            if not all(
                np.array_equal(grid, np.asarray(selection.energies) * EV_PER_MEV)
                for selection in selections
            ):
                raise ValueError("Spectrum probabilities must use a common energy grid")
            # Each bin uses its lower-edge probability; the final edge has no bin
            probabilities = np.column_stack([value[:-1] for value in values])
        else:
            raise ValueError(
                "Spectrum probabilities must be constant or histogram tables on a common grid"
            )
    # Store the bin edges and the matching bin-by-spectrum probability matrix
    quantity(group, "spectrum-probability-grid", grid, "eV")
    quantity(group, "spectrum-probability", probabilities)
    return components


# ======================================================================================
# Fission multiplicities and delayed-neutron groups
# ======================================================================================


def write_multiplicity(group, source):
    """Write an ACE fission multiplicity as a Data polynomial or table.

    Parameters
    ----------
    group : h5py.Group
        Empty destination for a dimensionless mean-neutron-yield function
        whose input is incident energy in eV.
    source : object
        ACEtk multiplicity record: type 1 supplies ascending polynomial
        coefficients, and type 2 supplies energies, multiplicities, and
        interpolation data.

    Raises
    ------
    ValueError
        The multiplicity type or interpolation representation is unsupported.

    Notes
    -----
    For a source polynomial in MeV, coefficient i is divided by
    ``EV_PER_MEV**i`` so evaluation with an eV input gives the same yield.
    Multiple coefficients carry individual unit strings: ``1``, ``1/eV``,
    ``1/eV^2``, and so on. A lone dimensionless coefficient has no unit
    attribute and retains the polynomial representation.

    Table records receive an eV grid, unchanged dimensionless yields,
    and converted interpolation regions. This function does not infer
    whether the source yield is prompt, delayed, or total; its caller must
    select the physically appropriate source record.
    """
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
    """Copy nuclide-wide ACE fission yields and delayed groups into a reaction.

    Parameters
    ----------
    group : h5py.Group
        Fission reaction receiving ``prompt-multiplicity``,
        ``delayed-multiplicity``, and ``delayed-neutrons``. Prompt spectra
        and angular data are written separately by ``write_reaction``.
    ace : ACEtk.ContinuousEnergyTable
        Table supplying fission multiplicity, delayed multiplicity,
        precursor, and delayed-energy-distribution blocks.

    Raises
    ------
    ValueError
        Fission multiplicity is missing; only total multiplicity is supplied
        alongside delayed data; a precursor fraction is not represented by
        two equal probabilities with no interpolation regions; or a delayed
        spectrum has an unsupported representation.

    Notes
    -----
    The source fission-block multiplicity is written as prompt yield. A
    total-only source is accepted only when no delayed-multiplicity block
    is present; the function does not reconstruct prompt yield by subtraction.
    An absent delayed yield is represented explicitly as constant zero.

    Delayed group i maps to index i in ``fractions`` and ``decay-rates`` and
    to ``energy-distribution-{i}``. Fractions are copied without
    renormalization; ACEtk decay constants are written directly with unit
    ``1/s``. Absent precursor data produces empty arrays and no group spectra.
    Energy conversion is delegated to ``write_energy_distribution``.

    Each reaction owns a copy of these data, including chance-fission
    reactions whose ACE source shares a single nuclide-wide yield block.
    The ``fission-production-source`` attribute records that provenance.
    """
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
# Angular distributions
# ======================================================================================


def write_angular_distribution(group, source):
    """Write explicit isotropy or incident-dependent cosine-density tables.

    Parameters
    ----------
    group : h5py.Group
        Empty destination Distribution group.
    source : object
        ACEtk ``FullyIsotropicDistribution`` or an angular distribution
        exposing incident energies and cosine distribution tables.

    Raises
    ------
    ValueError
        A per-energy distribution is not a table, a table has fewer than
        two points, or its interpolation law is unsupported.

    Notes
    -----
    Fully isotropic data needs only ``type = isotropic``. Other inputs
    use ``type = multi-table`` with an eV ``grid``, linear interpolation
    between incident energies, and packed conditional cosine tables.
    Each table retains its source PDF interpolation law. Both cosine and
    cosine-density datasets are dimensionless and omit unit attributes.
    Angles stored jointly with outgoing energy are handled by the energy
    distribution writer instead of this function.
    """
    # Represent isotropy without constructing an artificial cosine table
    if isinstance(source, ACEtk.continuous.FullyIsotropicDistribution):
        group.attrs["type"] = "isotropic"
        return
    # Collect conditional cosine tables, retaining each local interpolation law.
    tables = []
    for i, table in enumerate(source.distributions):
        if source.distribution_type(i + 1) != ACEtk.AngularDistributionType.Tabulated:
            raise ValueError("Angular distributions must use tables or be isotropic")
        tables.append(
            (table.cosines, table.pdf, [interpolation_law(table.interpolation)], None)
        )
    multi_table(
        group,
        np.asarray(source.incident_energies) * EV_PER_MEV,
        tables,
        unit=None,
    )


# ======================================================================================
# Energy distributions
# ======================================================================================


def write_energy_distribution(group, source):
    """Write a supported outgoing-energy or joint energy-angle Distribution.

    Parameters
    ----------
    group : h5py.Group
        Empty destination group for a single distribution, not a mixture's
        selection probabilities.
    source : object
        ACEtk level-scattering, evaporation, simple Maxwellian fission,
        N-body phase-space, outgoing-energy table, Kalbach--Mann, or
        energy-angle table distribution object.

    Raises
    ------
    ValueError
        The source class or N-body parameters are unsupported,
        a multi-incident-energy source is not linear-linear between incident
        energies, or a delegated table/interpolation writer rejects data.

    Notes
    -----
    Energy coordinates and parameters are converted from MeV to eV, while
    energy PDFs are divided by ``EV_PER_MEV``. Angular cosines, cosine PDFs,
    level-scattering C2, precompound fractions, angular slopes, and N-body
    energy fractions remain dimensionless. PDF tables are copied without
    normalization.

    Level scattering writes C1 and C2. Evaporation and Maxwellian spectra
    write restriction energy and spectral-temperature Data tables,
    preserving that function's interpolation regions. N-body data preserves
    the particle count, total mass ratio, normalized energy-fraction grid,
    and its PDF interpolation. Target mass and reaction Q-value supply the
    incident-energy-dependent energy scale during transport.

    Outgoing-energy tables use ``multi-table`` with packed PDF/PMF components
    on an eV incident-energy grid. Leading discrete entries retain their
    dimensionless masses and corresponding line indices; continuous densities
    retain their individual interpolation laws and combined probability.
    Correlated sources pack outgoing
    densities in ``tables/outgoing-energy``. Energy-angle sources additionally
    pack conditional angular densities in ``tables/cosine-distributions``,
    ordered by the concatenated outgoing-energy points. Kalbach--Mann sources
    retain one Data record per incident energy under each parameter group.
    ACE supplies these parameter values on the outgoing-energy grid, which
    is retained in each parameter's Data table.
    Correlated representations have inherent linear incident-energy dependence
    rather than a separate outer interpolation group.

    Source interpolation within each PDF or parameter table is retained.
    Selection among multiple spectra is handled by ``write_spectrum_selection``.
    Existing objects are not replaced, and an error can leave partial output
    in the destination group; the caller owns file publication and cleanup.
    """
    if isinstance(source, ACEtk.continuous.LevelScatteringDistribution):
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
        if not source.interpolation_data.is_linear_linear:
            raise ValueError(
                "This distribution requires linear interpolation between incident energies"
            )
        incident_energy = np.asarray(source.incident_energies) * EV_PER_MEV
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
            )
        else:
            # Pack both levels of correlated tables in their original source order.
            kalbach = isinstance(source, ACEtk.continuous.KalbachMannDistributionData)
            group.attrs["type"] = "kalbach-mann" if kalbach else "energy-angle-table"
            quantity(group, "energy", incident_energy, "eV")
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
