import numpy as np
import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.rng as rng

from mcdc.constant import (
    DISTRIBUTION_ISOTROPIC,
    DISTRIBUTION_NONE,
    DISTRIBUTION_EVAPORATION,
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_LEVEL_SCATTERING,
    DISTRIBUTION_MAXWELLIAN,
    DISTRIBUTION_MULTITABLE,
    DISTRIBUTION_N_BODY,
    DISTRIBUTION_TABLE,
    DISTRIBUTION_TABLE_ENERGY_ANGLE,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
    PI,
)
from mcdc.transport.data import (
    evaluate_data,
    evaluate_data_clamped,
    get_table_interpolation_law,
)
from mcdc.transport.util import find_bin, integrate_segment, interpolate
from mcdc.transport.linalg import make_direction_basis

# ======================================================================================
# General distribution samplers
# ======================================================================================


@njit
def sample_normal(rng_state):
    """Sample a standard normal using the particle's reproducible RNG stream."""
    # Box-Muller transform; 1 - xi keeps the logarithm's argument positive.
    radius = math.sqrt(-2.0 * math.log(1.0 - rng.lcg(rng_state)))
    angle = 2.0 * math.pi * rng.lcg(rng_state)
    return radius * math.cos(angle)


@njit
def sample_distribution(E, distribution, rng_state, simulation, data):
    return _sample_distribution(E, distribution, rng_state, simulation, data, False)


@njit
def sample_distribution_with_scale(E, distribution, rng_state, simulation, data):
    return _sample_distribution(E, distribution, rng_state, simulation, data, True)


@njit
def _sample_distribution(E, distribution, rng_state, simulation, data, scale):
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_NONE:
        raise ValueError("Cannot sample an absent Distribution")

    elif distribution_type == DISTRIBUTION_ISOTROPIC:
        return sample_isotropic_cosine(rng_state)

    elif distribution_type == DISTRIBUTION_TABLE:
        table = simulation["table_distributions"][ID]
        return sample_table(table, rng_state, simulation, data)

    elif distribution_type == DISTRIBUTION_MULTITABLE:
        multi_table = simulation["multi_table_distributions"][ID]
        return _sample_multi_table(E, rng_state, multi_table, simulation, data, scale)

    elif distribution_type == DISTRIBUTION_LEVEL_SCATTERING:
        level_scattering = simulation["level_scattering_distributions"][ID]
        return sample_level_scattering(E, level_scattering)

    elif distribution_type == DISTRIBUTION_EVAPORATION:
        evaporation = simulation["evaporation_distributions"][ID]
        return sample_evaporation(E, rng_state, evaporation, simulation, data)

    elif distribution_type == DISTRIBUTION_MAXWELLIAN:
        maxwellian = simulation["maxwellian_distributions"][ID]
        return sample_maxwellian(E, rng_state, maxwellian, simulation, data)

    # TODO: Should not get here
    else:
        return -1.0


@njit
def sample_correlated_distribution(
    E, distribution, rng_state, simulation, data, atomic_weight_ratio=None, q_value=None
):
    """Sample outgoing energy and cosine in the distribution's reference frame.

    N-body phase space requires the target-to-projectile mass ratio and the
    reaction Q-value in eV to set its maximum outgoing COM energy.
    """
    return _sample_correlated_distribution(
        E,
        distribution,
        rng_state,
        simulation,
        data,
        False,
        atomic_weight_ratio,
        q_value,
    )


@njit
def sample_correlated_distribution_with_scale(
    E, distribution, rng_state, simulation, data, atomic_weight_ratio=None, q_value=None
):
    """Sample a joint law with outgoing-energy support scaling.

    N-body's physical scale uses the supplied target mass ratio and Q-value;
    its tabulated coordinate is always a dimensionless energy fraction.
    """
    return _sample_correlated_distribution(
        E, distribution, rng_state, simulation, data, True, atomic_weight_ratio, q_value
    )


@njit
def _sample_correlated_distribution(
    E, distribution, rng_state, simulation, data, scale, atomic_weight_ratio, q_value
):
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_KALBACH_MANN:
        kalbach_mann = simulation["kalbach_mann_distributions"][ID]
        return sample_kalbach_mann(E, rng_state, kalbach_mann, data, simulation)

    elif distribution_type == DISTRIBUTION_TABLE_ENERGY_ANGLE:
        table = simulation["table_energy_angle_distributions"][ID]
        return sample_table_energy_angle(E, rng_state, table, data)

    elif distribution_type == DISTRIBUTION_N_BODY:
        if atomic_weight_ratio is None or q_value is None:
            raise ValueError(
                "N-body sampling requires target mass ratio and reaction Q-value"
            )
        if (
            not math.isfinite(atomic_weight_ratio)
            or atomic_weight_ratio <= 0
            or not math.isfinite(q_value)
        ):
            raise ValueError("Invalid N-body target mass ratio or reaction Q-value")
        nbody = simulation["nbody_distributions"][ID]
        fraction, _ = sample_density_with_interval(
            nbody["pdf_ID"], rng.lcg(rng_state), simulation, data
        )
        available = atomic_weight_ratio / (atomic_weight_ratio + 1.0) * E + q_value
        maximum = (1.0 - 1.0 / nbody["total_mass_ratio"]) * max(0.0, available)
        E_out = fraction * maximum
        mu = sample_isotropic_cosine(rng_state)
        return E_out, mu

    # TODO: Should not get here
    else:
        return -1.0, -1.0


# ======================================================================================
# Distribution samplers
# ======================================================================================


@njit
def sample_uniform(low, high, rng_state):
    return low + rng.lcg(rng_state) * (high - low)


@njit
def sample_isotropic_cosine(rng_state):
    return 2.0 * rng.lcg(rng_state) - 1.0


@njit
def sample_isotropic_direction(rng_state):
    # Sample polar cosine and azimuthal angle uniformly
    mu = sample_isotropic_cosine(rng_state)
    azi = 2.0 * PI * rng.lcg(rng_state)

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    y = math.cos(azi) * c
    z = math.sin(azi) * c
    x = mu
    return x, y, z


@njit
def sample_table(table, rng_state, simulation, data):
    """Sample a continuous value or an exact discrete outcome from a table."""

    value, _ = sample_table_with_interval(table, rng_state, simulation, data)
    return value


@njit
def sample_table_with_interval(table, rng_state, simulation, data):
    """Sample a table and return its PDF interval, or -1 for a discrete outcome."""

    xi = rng.lcg(rng_state)
    masses = mcdc_get.table_distribution.cmf_all(table, data)
    discrete_probability = masses[-1]
    if xi < discrete_probability or table["continuous_probability"] == 0.0:
        pmf = mcdc_get.table_distribution.pmf_all(table, data)
        index = sample_discrete_index(xi, masses[1:], pmf)
        return mcdc_get.table_distribution.discrete_value(index, table, data), -1
    return sample_density_with_interval(
        table["pdf_ID"], xi - discrete_probability, simulation, data
    )


@njit
def sample_density_with_interval(pdf_ID, xi, simulation, data):
    """Invert a continuous DataTable density at a supplied cumulative mass."""
    pdf_data = simulation["data"][pdf_ID]
    pdf_table = simulation["table_data"][pdf_data["sub_ID"]]

    cdf = mcdc_get.table_data.aux_vector(0, pdf_table, data)

    # find_bin returns -1 at the first CDF point; avoid reading before the table.
    if xi <= cdf[0]:
        return mcdc_get.table_data.x(0, pdf_table, data), 0
    if xi >= cdf[-1]:
        return mcdc_get.table_data.x_last(pdf_table, data), len(cdf) - 2
    idx = min(max(find_bin(xi, cdf), 0), len(cdf) - 2)

    c0 = mcdc_get.table_data.aux(0, idx, pdf_table, data)

    p0 = mcdc_get.table_data.y(idx, pdf_table, data)
    p1 = mcdc_get.table_data.y(idx + 1, pdf_table, data)

    v0 = mcdc_get.table_data.x(idx, pdf_table, data)
    v1 = mcdc_get.table_data.x(idx + 1, pdf_table, data)

    interpolation = get_table_interpolation_law(idx, pdf_table, data)

    value = invert_table_segment(xi, c0, v0, v1, p0, p1, interpolation)
    return value, idx


@njit
def invert_table_segment(xi, c0, v0, v1, p0, p1, interpolation):
    """Invert a density segment, using bounded inversion for logarithmic laws."""

    s = xi - c0

    if interpolation == INTERPOLATION_HISTOGRAM:
        if p0 <= 0.0:
            return v0
        return v0 + s / p0

    elif interpolation == INTERPOLATION_LINEAR:
        dv = v1 - v0
        dp = p1 - p0

        if abs(dp) <= 1.0e-12 * max(abs(p0), abs(p1)):
            if p0 <= 0.0:
                return v0
            return v0 + s / p0

        m = dp / dv
        disc = p0 * p0 + 2.0 * m * s

        if disc < 0.0:
            disc = 0.0

        # Rationalized form avoids cancellation for almost constant densities.
        return v0 + 2.0 * s / (math.sqrt(disc) + p0)

    else:
        low, high = v0, v1
        for _ in range(52):
            middle = 0.5 * (low + high)
            if integrate_segment(middle, v0, v1, p0, p1, interpolation) < s:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)


@njit
def sample_white_direction(nx, ny, nz, rng_state):
    """Sample a cosine-weighted hemisphere about the normalized reference."""
    # Sample polar cosine
    mu = math.sqrt(rng.lcg(rng_state))

    # Sample azimuthal direction
    azi = 2.0 * PI * rng.lcg(rng_state)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    sin_polar = math.sqrt(max(0.0, 1.0 - mu * mu))
    u1x, u1y, u1z, u2x, u2y, u2z = make_direction_basis(nx, ny, nz)

    x = sin_polar * cos_azi * u1x + sin_polar * sin_azi * u2x + mu * nx
    y = sin_polar * cos_azi * u1y + sin_polar * sin_azi * u2y + mu * ny
    z = sin_polar * cos_azi * u1z + sin_polar * sin_azi * u2z + mu * nz
    return x, y, z


@njit
def sample_multi_table(E, rng_state, multi_table, simulation, data):
    return _sample_multi_table(E, rng_state, multi_table, simulation, data, False)


@njit
def sample_discrete_index(xi, cmf, pmf):
    """Select a point mass from upper cumulative bounds, skipping zero masses."""
    index = min(int(np.searchsorted(cmf, xi, side="right")), len(pmf) - 1)
    while index > 0 and pmf[index] == 0.0:
        index -= 1
    return index


@njit
def _sample_multi_table(E, rng_state, multi_table, simulation, data, scale):
    """Sample corresponding discrete lines or a unit-base continuous spectrum.

    Line probabilities and coordinates interpolate between incident tables.
    Continuous endpoint selection is conditional on reaching the continuous
    component; its weights include each endpoint's continuous probability.
    """
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)
    interior = False
    fraction = 0.0
    if E <= grid[0]:
        low = int(np.searchsorted(grid, E, side="right")) - 1 if E == grid[0] else 0
        high = low
        scale = False
    elif E >= grid[-1]:
        low = len(grid) - 1
        high = low
        scale = False
    else:
        interior = True
        low = int(np.searchsorted(grid, E, side="right")) - 1
        high = low + 1
        fraction = (E - grid[low]) / (grid[high] - grid[low])
        boundaries = mcdc_get.multi_table_distribution.interpolation_boundaries_all(
            multi_table, data
        )
        region = 0
        while region < len(boundaries) - 1 and low + 1 >= boundaries[region]:
            region += 1
        if (
            mcdc_get.multi_table_distribution.interpolations(region, multi_table, data)
            == INTERPOLATION_HISTOGRAM
        ):
            fraction = 0.0
            high = low
            scale = False

    discrete_offsets = mcdc_get.multi_table_distribution.discrete_offsets_all(
        multi_table, data
    )
    begin, end = int(discrete_offsets[low]), int(discrete_offsets[low + 1])
    if end > begin:
        pmf = mcdc_get.multi_table_distribution.pmf_all(multi_table, data)
        cmf = mcdc_get.multi_table_distribution.cmf_all(multi_table, data)
        values = mcdc_get.multi_table_distribution.discrete_value_all(multi_table, data)
        other = int(discrete_offsets[high])
        xi = rng.lcg(rng_state)
        cumulative = 0.0
        last_positive = 0
        for j in range(end - begin):
            probability = (1.0 - fraction) * pmf[begin + j] + fraction * pmf[other + j]
            cumulative = (1.0 - fraction) * cmf[begin + j] + fraction * cmf[other + j]
            if probability > 0.0:
                last_positive = j
                if xi < cumulative:
                    return (1.0 - fraction) * values[begin + j] + fraction * values[
                        other + j
                    ]
        probabilities = mcdc_get.multi_table_distribution.continuous_probability_all(
            multi_table, data
        )
        continuous = (1.0 - fraction) * probabilities[low] + fraction * probabilities[
            high
        ]
        if continuous <= 0.0:
            return (1.0 - fraction) * values[begin + last_positive] + fraction * values[
                other + last_positive
            ]
        # Rebase the remaining draw onto the normalized continuous component.
        xi = min(1.0, max(0.0, (xi - cumulative) / continuous))
        upper_weight = fraction * probabilities[high] / continuous
        selected = high if low != high and rng.lcg(rng_state) < upper_weight else low
        xi *= probabilities[selected]
    else:
        # A continuous-only table retains its ordinary stochastic table selection.
        selected = high if interior and rng.lcg(rng_state) < fraction else low
        xi = rng.lcg(rng_state)

    sampled = sample_multi_table_at(selected, xi, multi_table, data)
    if not scale:
        return sampled
    offsets = mcdc_get.multi_table_distribution.offsets_all(multi_table, data)
    # If the continuum exists at only one endpoint, retain that support.
    if offsets[low + 1] == offsets[low] or offsets[high + 1] == offsets[high]:
        return sampled
    values = mcdc_get.multi_table_distribution.value_all(multi_table, data)
    minimum = values[int(offsets[low])] + fraction * (
        values[int(offsets[high])] - values[int(offsets[low])]
    )
    maximum = values[int(offsets[low + 1]) - 1] + fraction * (
        values[int(offsets[high + 1]) - 1] - values[int(offsets[low + 1]) - 1]
    )
    first = values[int(offsets[selected])]
    last = values[int(offsets[selected + 1]) - 1]
    return minimum + (sampled - first) / (last - first) * (maximum - minimum)


@njit
def multi_table_arrays(index, table, data):
    """Return one PDF's coordinates, densities, CDF, and local region arrays."""
    offsets = mcdc_get.multi_table_distribution.offsets_all(table, data)
    start, end = int(offsets[index]), int(offsets[index + 1])
    regions = mcdc_get.multi_table_distribution.pdf_interpolation_offsets_all(
        table, data
    )
    first, last = int(regions[index]), int(regions[index + 1])
    return (
        mcdc_get.multi_table_distribution.value_all(table, data)[start:end],
        mcdc_get.multi_table_distribution.pdf_all(table, data)[start:end],
        mcdc_get.multi_table_distribution.cdf_all(table, data)[start:end],
        mcdc_get.multi_table_distribution.pdf_interpolations_all(table, data)[
            first:last
        ],
        mcdc_get.multi_table_distribution.pdf_interpolation_boundaries_all(table, data)[
            first:last
        ],
    )


@njit
def sample_multi_table_at(index, xi, table, data):
    """Invert one packed conditional PDF with its own interpolation regions."""
    value, pdf, cdf, laws, boundaries = multi_table_arrays(index, table, data)
    if xi <= cdf[0]:
        return value[0]
    if xi >= cdf[-1]:
        return value[-1]
    interval = min(max(find_bin(xi, cdf), 0), len(cdf) - 2)
    region = 0
    while region < len(laws) - 1 and interval + 1 >= boundaries[region]:
        region += 1
    return invert_table_segment(
        xi,
        cdf[interval],
        value[interval],
        value[interval + 1],
        pdf[interval],
        pdf[interval + 1],
        laws[region],
    )


@njit
def sample_maxwellian(E, rng_state, maxwellian, simulation, data):
    # Get nuclear temperature
    table = simulation["data"][maxwellian["nuclear_temperature_ID"]]
    nuclear_temperature = evaluate_data_clamped(E, table, simulation, data)
    restriction_energy = maxwellian["restriction_energy"]
    if nuclear_temperature <= 0.0 or E <= restriction_energy:
        return 0.0

    # Rejection sampling
    while True:
        xi1 = rng.lcg(rng_state)
        xi2 = rng.lcg(rng_state)
        xi3 = rng.lcg(rng_state)
        cos = math.cos(0.5 * PI * xi3)
        cos_square = cos * cos
        sample = -nuclear_temperature * (math.log(xi1) + math.log(xi2) * cos_square)

        # Accept sample?
        if 0.0 <= sample and sample <= E - restriction_energy:
            break

    return sample


@njit
def sample_level_scattering(E, level_scattering):
    C1 = level_scattering["C1"]
    C2 = level_scattering["C2"]
    return C2 * (E - C1)


@njit
def sample_evaporation(E, rng_state, evaporation, simulation, data):
    # Get nuclear temperature
    table = simulation["data"][evaporation["nuclear_temperature_ID"]]
    nuclear_temperature = evaluate_data_clamped(E, table, simulation, data)
    restriction_energy = evaporation["restriction_energy"]
    if nuclear_temperature <= 0.0 or E <= restriction_energy:
        return 0.0

    w = (E - restriction_energy) / nuclear_temperature
    g = 1.0 - math.exp(-w)

    # Rejection sampling
    while True:
        xi1 = rng.lcg(rng_state)
        xi2 = rng.lcg(rng_state)
        sample = -nuclear_temperature * math.log((1.0 - g * xi1) * (1.0 - g * xi2))

        # Accept sample?
        if 0.0 <= sample and sample <= E - restriction_energy:
            break

    return sample


@njit
def correlated_table_indices(E, grid, xi):
    """Select bracketing incident tables with constant endpoint extrapolation."""
    if len(grid) == 1 or E <= grid[0]:
        return 0, 0, 0, 0.0
    if E >= grid[-1]:
        last = len(grid) - 1
        return last, last, last, 0.0
    low = max(0, find_bin(E, grid))
    high = low + 1
    fraction = (E - grid[low]) / (grid[high] - grid[low])
    selected = high if xi < fraction else low
    return low, high, selected, fraction


@njit
def sample_flat_density(xi, values, pdf, cdf, laws):
    """Sample a normalized flattened-table slice, returning the local interval."""
    idx = min(max(find_bin(xi, cdf), 0), len(values) - 2)
    law = INTERPOLATION_LINEAR if len(laws) == 0 else laws[idx]
    value = invert_table_segment(
        xi, cdf[idx], values[idx], values[idx + 1], pdf[idx], pdf[idx + 1], law
    )
    return value, idx, law


@njit
def sample_kalbach_mann(E, rng_state, kalbach_mann, data, simulation=None):
    """Sample unit-base outgoing energy and its conditional Kalbach-Mann angle."""
    grid = mcdc_get.kalbach_mann_distribution.energy_all(kalbach_mann, data)
    offsets = mcdc_get.kalbach_mann_distribution.offset_all(kalbach_mann, data)
    values = mcdc_get.kalbach_mann_distribution.energy_out_all(kalbach_mann, data)
    pdf = mcdc_get.kalbach_mann_distribution.pdf_all(kalbach_mann, data)
    cdf = mcdc_get.kalbach_mann_distribution.cdf_all(kalbach_mann, data)
    laws = mcdc_get.kalbach_mann_distribution.interpolations_all(kalbach_mann, data)
    xi1, xi2 = rng.lcg(rng_state), rng.lcg(rng_state)
    xi3, xi4 = rng.lcg(rng_state), rng.lcg(rng_state)
    low, high, selected, fraction = correlated_table_indices(E, grid, xi1)
    low_end = int(offsets[low + 1]) if low + 1 < len(grid) else len(values)
    high_end = int(offsets[high + 1]) if high + 1 < len(grid) else len(values)
    start = int(offsets[selected])
    end = int(offsets[selected + 1]) if selected + 1 < len(grid) else len(values)
    local_laws = laws[start:end] if len(laws) else laws
    sampled, interval, law = sample_flat_density(
        xi2, values[start:end], pdf[start:end], cdf[start:end], local_laws
    )
    minimum = values[int(offsets[low])] + fraction * (
        values[int(offsets[high])] - values[int(offsets[low])]
    )
    maximum = values[low_end - 1] + fraction * (
        values[high_end - 1] - values[low_end - 1]
    )
    E_new = minimum + (sampled - values[start]) / (values[end - 1] - values[start]) * (
        maximum - minimum
    )

    # Parameter Data can be constant, polynomial, or independently tabulated.
    if simulation is not None and kalbach_mann["N_precompound_table"] > 0:
        ID = mcdc_get.kalbach_mann_distribution.precompound_table_IDs(
            selected, kalbach_mann, data
        )
        R = evaluate_data_clamped(sampled, simulation["data"][ID], simulation, data)
        ID = mcdc_get.kalbach_mann_distribution.angular_slope_table_IDs(
            selected, kalbach_mann, data
        )
        A = evaluate_data_clamped(sampled, simulation["data"][ID], simulation, data)
    else:
        index = start + interval
        R = interpolate(
            sampled,
            values[index],
            values[index + 1],
            mcdc_get.kalbach_mann_distribution.precompound_factor(
                index, kalbach_mann, data
            ),
            mcdc_get.kalbach_mann_distribution.precompound_factor(
                index + 1, kalbach_mann, data
            ),
            law,
        )
        A = interpolate(
            sampled,
            values[index],
            values[index + 1],
            mcdc_get.kalbach_mann_distribution.angular_slope(index, kalbach_mann, data),
            mcdc_get.kalbach_mann_distribution.angular_slope(
                index + 1, kalbach_mann, data
            ),
            law,
        )
    if abs(A) < 1.0e-12:
        mu = 2.0 * xi4 - 1.0
    elif xi3 > R:
        mu = math.asinh((2.0 * xi4 - 1.0) * math.sinh(A)) / A
    else:
        mu = 1.0 + math.log(xi4 + (1.0 - xi4) * math.exp(-2.0 * A)) / A
    return E_new, max(-1.0, min(1.0, mu))


@njit
def sample_table_energy_angle(E, rng_state, table, data):
    """Sample outgoing energy and the corresponding conditional cosine table."""
    grid = mcdc_get.table_energy_angle_distribution.energy_all(table, data)
    offsets = mcdc_get.table_energy_angle_distribution.offset_all(table, data)
    values = mcdc_get.table_energy_angle_distribution.energy_out_all(table, data)
    pdf = mcdc_get.table_energy_angle_distribution.pdf_all(table, data)
    cdf = mcdc_get.table_energy_angle_distribution.cdf_all(table, data)
    laws = mcdc_get.table_energy_angle_distribution.interpolations_all(table, data)
    xi1, xi2, xi3 = rng.lcg(rng_state), rng.lcg(rng_state), rng.lcg(rng_state)
    low, high, selected, fraction = correlated_table_indices(E, grid, xi1)
    low_end = int(offsets[low + 1]) if low + 1 < len(grid) else len(values)
    high_end = int(offsets[high + 1]) if high + 1 < len(grid) else len(values)
    start = int(offsets[selected])
    end = int(offsets[selected + 1]) if selected + 1 < len(grid) else len(values)
    local_laws = laws[start:end] if len(laws) else laws
    sampled, interval, law = sample_flat_density(
        xi2, values[start:end], pdf[start:end], cdf[start:end], local_laws
    )
    minimum = values[int(offsets[low])] + fraction * (
        values[int(offsets[high])] - values[int(offsets[low])]
    )
    maximum = values[low_end - 1] + fraction * (
        values[high_end - 1] - values[low_end - 1]
    )
    E_new = minimum + (sampled - values[start]) / (values[end - 1] - values[start]) * (
        maximum - minimum
    )

    # Histogram laws use the lower angular table. Linear laws use the nearest
    # cumulative-probability endpoint, measured within the selected incident table.
    index = start + interval
    if law != INTERPOLATION_HISTOGRAM and xi2 - cdf[index] > cdf[index + 1] - xi2:
        index += 1
    cosine_offsets = mcdc_get.table_energy_angle_distribution.cosine_offset__all(
        table, data
    )
    cosine = mcdc_get.table_energy_angle_distribution.cosine_all(table, data)
    cosine_pdf = mcdc_get.table_energy_angle_distribution.cosine_pdf_all(table, data)
    cosine_cdf = mcdc_get.table_energy_angle_distribution.cosine_cdf_all(table, data)
    cosine_laws = mcdc_get.table_energy_angle_distribution.cosine_interpolations_all(
        table, data
    )
    start = int(cosine_offsets[index])
    end = (
        int(cosine_offsets[index + 1])
        if index + 1 < len(cosine_offsets)
        else len(cosine)
    )
    local_laws = cosine_laws[start:end] if len(cosine_laws) else cosine_laws
    mu, _, _ = sample_flat_density(
        xi3, cosine[start:end], cosine_pdf[start:end], cosine_cdf[start:end], local_laws
    )
    return E_new, max(-1.0, min(1.0, mu))
