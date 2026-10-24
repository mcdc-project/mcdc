"""Read mcdc-data representations into transport objects without unit conversion.

The HDF5 format and runtime use eV and exclusive interpolation boundaries.
Packed PDF collections are read directly into specialized MultiTable arrays.
Source-library metadata stays in the files.
"""

import numpy as np

from mcdc.constant import INTERPOLATION_LINEAR, REFERENCE_FRAME_COM, REFERENCE_FRAME_LAB

from mcdc.object_.data import DataPolynomial, DataTable, encode_interpolation
from mcdc.object_.distribution import (
    DistributionIsotropic,
    DistributionTable,
    DistributionMultiTable,
    DistributionLevelScattering,
    DistributionEvaporation,
    DistributionMaxwellian,
    DistributionKalbachMann,
    DistributionTableEnergyAngle,
    DistributionNBody,
)

# ======================================================================================
# File identity and interpolation
# ======================================================================================


def text(value):
    """Decode an HDF5 string regardless of the h5py string representation."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def check_format(file):
    """Reject files whose format identity or schema version is unsupported."""
    name = text(file.attrs.get("format", ""))
    version = text(file.attrs.get("format-version", ""))
    if name != "mcdc-data" or version != "0.1.0":
        raise ValueError(
            f"{file.filename}: expected mcdc-data 0.1.0, found {name!r} {version!r}"
        )


def read_interpolation(group):
    """Return packed laws and exclusive boundaries from an interpolation group."""
    interpolation = group["interpolation"]
    laws = np.array(
        [encode_interpolation(text(law)) for law in interpolation["laws"][()]],
        dtype=np.int64,
    )
    boundaries = np.asarray(interpolation["boundaries"][()], dtype=np.int64)
    coordinate = next(name for name in ("value", "grid", "energy") if name in group)
    size = len(group[coordinate])
    if (
        laws.ndim != 1
        or boundaries.ndim != 1
        or len(laws) != len(boundaries)
        or not len(laws)
    ):
        raise ValueError(
            f"{group.name}: interpolation laws and boundaries must be nonempty vectors of equal length"
        )
    if boundaries[0] < 2 or boundaries[-1] != size or np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"{group.name}: invalid interpolation region boundaries")
    return laws, boundaries


def numbered_groups(group, prefix=""):
    """Read contiguous zero-based child indices in numeric order."""
    names = [name for name in group if name.startswith(prefix)]
    names.sort(key=lambda name: int(name[len(prefix) :]))
    if names != [f"{prefix}{i}" for i in range(len(names))]:
        raise ValueError(
            f"{group.name}: expected contiguous zero-based {prefix!r} indices"
        )
    return [group[name] for name in names]


# ======================================================================================
# Shared-grid cross sections and reaction fields
# ======================================================================================


def read_xs_energy_grid(group):
    """Read a projectile's eV grid, retaining repeated coordinates at jumps."""
    grid = np.asarray(group["xs-energy-grid"][()], dtype=np.float64)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) < 0)
        or grid[-1] <= grid[0]
    ):
        raise ValueError(
            f"{group.name}: cross-section energies must be finite and nondecreasing with at least two distinct values"
        )
    return grid


def read_reaction_properties(group):
    """Read source MT, cross-section segment, offset, and reference frame.

    Reaction category determines transport dispatch; ENDF-MT is optional source
    metadata. Cross-product data must not be silently discarded by a reader.
    """
    if len(group.get("cross-products", ())):
        raise ValueError(f"{group.name}: cross-product transport is unsupported")
    MT = int(group.attrs.get("ENDF-MT", 0))
    xs = np.asarray(group["xs/value"][()], dtype=np.float64)
    offset = int(group["xs/offset"][()])
    frame = text(group["reference-frame"][()])
    frames = {"LAB": REFERENCE_FRAME_LAB, "COM": REFERENCE_FRAME_COM}
    if frame not in frames:
        raise ValueError(f"{group.name}: unknown reference frame {frame!r}")
    return MT, xs, offset, frames[frame]


def set_xs_interpolation(reaction, group):
    """Validate shared-grid cross sections and attach their interpolation metadata.

    Neutron and photon reactions retain their own interpolation regions while
    sharing the projectile energy grid. Region boundaries are exclusive
    point indices local to the stored cross-section segment.
    """
    laws, boundaries = read_interpolation(group["xs"])
    if (
        reaction.xs.ndim != 1
        or reaction.xs_offset_ < 0
        or np.any(reaction.xs < 0)
        or not np.all(np.isfinite(reaction.xs))
    ):
        raise ValueError(f"{group.name}: cross sections must be finite and nonnegative")
    reaction.xs_interpolations = laws
    reaction.xs_interpolation_boundaries = boundaries
    return reaction


def xs_is_linear(reaction):
    """Whether summing grid values before interpolation preserves this reaction.

    A nonzero first value at a positive offset opens abruptly at its threshold.
    Such a segment must be evaluated separately to avoid interpolating the
    summed array into the interval below that threshold.
    """
    return bool(
        np.all(reaction.xs_interpolations == INTERPOLATION_LINEAR)
        and (reaction.xs_offset_ == 0 or reaction.xs[0] == 0.0)
    )


# ======================================================================================
# Scalar data
# ======================================================================================


def read_data(group):
    """Load constant, polynomial, or tabulated scalar data in canonical units."""
    kind = text(group.attrs["type"])
    if kind == "polynomial":
        return DataPolynomial(group["coefficients"][()])
    if kind == "table":
        return DataTable(
            group["grid"][()], group["value"][()], *read_interpolation(group)
        )
    raise ValueError(f"{group.name}: unsupported Data type {kind!r}")


# ======================================================================================
# Probability distributions
# ======================================================================================


def read_distribution(group):
    """Load an angle, energy, or correlated energy-angle distribution.

    Nested tables retain their own density interpolation. Incident-energy
    interpolation is carried separately, so angular sampling can interpolate
    probabilities while outgoing energies additionally use unit-base scaling.
    """
    kind = text(group.attrs["type"])
    if kind == "isotropic":
        return DistributionIsotropic()
    if kind == "table":
        arguments = {}
        if "value" in group or "pdf" in group:
            arguments.update(value=group["value"][()], pdf=group["pdf"][()])
            arguments["interpolations"], arguments["interpolation_boundaries"] = (
                read_interpolation(group)
            )
        if "discrete-value" in group or "pmf" in group:
            arguments.update(
                discrete_value=group["discrete-value"][()], pmf=group["pmf"][()]
            )
        return DistributionTable(**arguments)
    if kind == "multi-table":
        laws, boundaries = read_interpolation(group)
        return read_pdf_collection(
            group["tables"],
            grid=group["grid"][()],
            laws=laws,
            boundaries=boundaries,
            allow_discrete=True,
        )
    if kind == "level-scattering":
        return DistributionLevelScattering(
            float(group["C1"][()]), float(group["C2"][()])
        )
    if kind in ("evaporation", "maxwellian"):
        cls = (
            DistributionEvaporation if kind == "evaporation" else DistributionMaxwellian
        )
        return cls(
            read_data(group["temperature"]),
            restriction_energy=float(group["restriction-energy"][()]),
        )
    if kind == "n-body":
        density = read_distribution_density(group)
        return DistributionNBody(
            group["number-particles"][()],
            float(group["total-mass-ratio"][()]),
            density.x,
            density.y,
            interpolations=density.interpolations,
            interpolation_boundaries=density.interpolation_boundaries,
        )
    if kind in ("kalbach-mann", "energy-angle-table"):
        return read_correlated_distribution(group, kind)
    raise ValueError(f"{group.name}: unsupported Distribution type {kind!r}")


def read_distribution_density(group):
    """Return a normalized continuous density and its cumulative probabilities."""
    if "pmf" in group or "discrete-value" in group:
        raise ValueError(f"{group.name}: this density must be continuous only")
    laws, boundaries = read_interpolation(group)
    return DistributionTable(
        group["value"][()],
        pdf=group["pdf"][()],
        interpolations=laws,
        interpolation_boundaries=boundaries,
    ).pdf


def read_pdf_collection(
    group,
    grid=None,
    laws=INTERPOLATION_LINEAR,
    boundaries=None,
    *,
    allow_discrete=False,
):
    """Read packed probability tables without creating per-grid child objects.

    Correlated laws require continuous marginal/conditional densities.
    MultiTable additionally accepts aligned discrete components and pure PMFs.
    """
    discrete = any(
        name in group for name in ("pmf", "discrete-value", "discrete-offsets")
    )
    if discrete and not allow_discrete:
        raise ValueError(f"{group.name}: packed PDF collections are continuous only")
    arguments = {}
    if any(name in group for name in ("value", "pdf", "offsets", "interpolation")):
        if not all(
            name in group for name in ("value", "pdf", "offsets", "interpolation")
        ):
            raise ValueError(f"{group.name}: incomplete packed PDF fields")
        arguments.update(
            value=group["value"][()], pdf=group["pdf"][()], offsets=group["offsets"][()]
        )
        rule = group["interpolation"]
        arguments.update(
            pdf_interpolations=[
                encode_interpolation(text(law)) for law in rule["laws"][()]
            ],
            pdf_interpolation_boundaries=rule["boundaries"][()],
            pdf_interpolation_offsets=rule["offsets"][()],
        )
    if discrete:
        if not all(
            name in group for name in ("pmf", "discrete-value", "discrete-offsets")
        ):
            raise ValueError(f"{group.name}: incomplete packed discrete fields")
        arguments.update(
            discrete_value=group["discrete-value"][()],
            pmf=group["pmf"][()],
            discrete_offsets=group["discrete-offsets"][()],
        )
    if not arguments:
        raise ValueError(f"{group.name}: probability tables are missing")
    if grid is None:
        offsets = arguments.get("offsets", arguments.get("discrete_offsets"))
        grid = np.arange(len(offsets) - 1, dtype=float)
    return DistributionMultiTable(
        grid, interpolations=laws, interpolation_boundaries=boundaries, **arguments
    )


def packed_segment_laws(tables):
    """Expand local PDF regions for the correlated samplers' flat point arrays."""
    result = np.empty(len(tables.value), dtype=np.int64)
    for i, offset in enumerate(tables.offsets[:-1]):
        a, b = tables.pdf_interpolation_offsets[i : i + 2]
        start = 0
        for law, end in zip(
            tables.pdf_interpolations[a:b], tables.pdf_interpolation_boundaries[a:b]
        ):
            result[offset + start : offset + end] = law
            start = end - 1
    return result


def read_correlated_distribution(group, kind):
    """Load packed marginal PDFs and conditional angle data for a joint law."""
    tables = group["tables"]
    outgoing = read_pdf_collection(tables["outgoing-energy"])
    energy = group["energy"][()]
    if len(outgoing.grid) != len(energy):
        raise ValueError(
            f"{group.name}: one outgoing PDF is required per incident energy"
        )
    if kind == "kalbach-mann":
        factors = [
            read_data(child) for child in numbered_groups(tables["precompound-factor"])
        ]
        slopes = [
            read_data(child) for child in numbered_groups(tables["angular-slope"])
        ]
        if len(factors) != len(energy) or len(slopes) != len(energy):
            raise ValueError(
                f"{group.name}: one parameter Data record is required per incident energy"
            )
        result = DistributionKalbachMann(
            energy,
            outgoing.offsets[:-1],
            outgoing.value,
            outgoing.pdf,
            np.zeros(len(outgoing.value)),
            np.zeros(len(outgoing.value)),
        )
        result.precompound_tables, result.angular_slope_tables = factors, slopes
    else:
        angular = read_pdf_collection(tables["cosine-distributions"])
        if len(angular.grid) != len(outgoing.value):
            raise ValueError(
                f"{group.name}: each outgoing-energy point requires a cosine PDF"
            )
        result = DistributionTableEnergyAngle(
            energy,
            outgoing.offsets[:-1],
            outgoing.value,
            outgoing.pdf,
            angular.offsets[:-1],
            angular.value,
            angular.pdf,
        )
        result.cosine_pdf, result.cosine_cdf = angular.pdf, angular.cdf
        result.cosine_interpolations = packed_segment_laws(angular)
    result.pdf, result.cdf = outgoing.pdf, outgoing.cdf
    result.interpolations = packed_segment_laws(outgoing)
    return result
