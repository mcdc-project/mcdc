import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.rng as rng

from mcdc.constant import (
    DISTRIBUTION_EVAPORATION,
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_LEVEL_SCATTERING,
    DISTRIBUTION_MAXWELLIAN,
    DISTRIBUTION_MULTITABLE,
    DISTRIBUTION_N_BODY,
    DISTRIBUTION_TABULATED,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOG,
    INTERPOLATION_SEMILOGX,
    INTERPOLATION_SEMILOGY,
    MAX_BISECTION_ITERATIONS,
    PI,
)
from mcdc.transport.data import evaluate_table, get_table_interpolation_law
from mcdc.transport.util import find_bin, linear_interpolation

# ======================================================================================
# General distribution samplers
# ======================================================================================


@njit
def sample_distribution(E, distribution, rng_state, simulation, data):
    return _sample_distribution(E, distribution, rng_state, simulation, data, False)


@njit
def sample_distribution_with_scale(E, distribution, rng_state, simulation, data):
    return _sample_distribution(E, distribution, rng_state, simulation, data, True)


@njit
def _sample_distribution(E, distribution, rng_state, simulation, data, scale):
    distribution_type = distribution["child_type"]
    ID = distribution["child_ID"]

    if distribution_type == DISTRIBUTION_TABULATED:
        table = simulation["tabulated_distributions"][ID]
        return sample_tabulated(table, rng_state, simulation, data)

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
def sample_correlated_distribution(E, distribution, rng_state, simulation, data):
    return _sample_correlated_distribution(
        E, distribution, rng_state, simulation, data, False
    )


@njit
def sample_correlated_distribution_with_scale(
    E, distribution, rng_state, simulation, data
):
    return _sample_correlated_distribution(
        E, distribution, rng_state, simulation, data, True
    )


@njit
def _sample_correlated_distribution(
    E, distribution, rng_state, simulation, data, scale
):
    distribution_type = distribution["child_type"]
    ID = distribution["child_ID"]

    if distribution_type == DISTRIBUTION_KALBACH_MANN:
        kalbach_mann = simulation["kalbach_mann_distributions"][ID]
        return sample_kalbach_mann(E, rng_state, kalbach_mann, data)

    elif distribution_type == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
        table = simulation["tabulated_energy_angle_distributions"][ID]
        return sample_tabulated_energy_angle(E, rng_state, table, data)

    elif distribution_type == DISTRIBUTION_N_BODY:
        nbody = simulation["nbody_distributions"][ID]
        E_out = sample_tabulated(nbody, rng_state, simulation, data)
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
def sample_direction(polar_cosine, azimuthal, polar_coordinate, rng_state):
    # Sample polar cosine and azimuthal angle
    mu = sample_uniform(polar_cosine[0], polar_cosine[1], rng_state)
    azi = sample_uniform(azimuthal[0], azimuthal[1], rng_state)

    # Apply polar coordinate
    wx = polar_coordinate[0]
    wy = polar_coordinate[1]
    wz = polar_coordinate[2]
    if abs(wz) >= 0.999:
        inv = 1.0 / math.sqrt(wx * wx + wy * wy)

        ux = -wy * inv
        uy = wx * inv
        uz = 0.0

        vx = -wz * wx * inv
        vy = -wz * wy * inv
        vz = math.sqrt(wx * wx + wy * wy)
    else:
        # Axis nearly parallel to z
        ux, uy, uz = 1.0, 0.0, 0.0
        vx, vy, vz = 0.0, 1.0, 0.0

    # Rotate into lab frame
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    cphi = math.cos(azi)
    sphi = math.sin(azi)
    dx = s * cphi * ux + s * sphi * vx + mu * wx
    dy = s * cphi * uy + s * sphi * vy + mu * wy
    dz = s * cphi * uz + s * sphi * vz + mu * wz

    return dx, dy, dz


@njit
def sample_tabulated(table, rng_state, simulation, data):
    """
    Sample a value from a tabulated distribution.
    """

    pdf_table = simulation["table_data"][table["pdf_ID"]]

    cdf = mcdc_get.table_data.aux_vector(0, pdf_table, data)

    xi = rng.lcg(rng_state)
    idx = find_bin(xi, cdf)

    c0 = mcdc_get.table_data.aux(0, idx, pdf_table, data)

    p0 = mcdc_get.table_data.y(idx, pdf_table, data)
    p1 = mcdc_get.table_data.y(idx + 1, pdf_table, data)

    v0 = mcdc_get.table_data.x(idx, pdf_table, data)
    v1 = mcdc_get.table_data.x(idx + 1, pdf_table, data)

    # Tabulated pdfs are either histogram or linear.
    interpolation = mcdc_get.table_data.interpolations(0, pdf_table, data)

    return invert_tabulated_segment(
        xi,
        c0,
        v0,
        v1,
        p0,
        p1,
        interpolation,
    )


@njit
def invert_tabulated_segment(xi, c0, v0, v1, p0, p1, interpolation):
    """Invert one histogram or linear PDF segment."""

    s = xi - c0

    if interpolation == INTERPOLATION_HISTOGRAM:
        if p0 <= 0.0:
            return v0
        return v0 + s / p0

    elif interpolation == INTERPOLATION_LINEAR:
        dv = v1 - v0
        dp = p1 - p0

        if abs(dp) <= 1.0e-12 * max(abs(p0), abs(p1), 1.0):
            if p0 <= 0.0:
                return v0
            return v0 + s / p0

        m = dp / dv
        disc = p0 * p0 + 2.0 * m * s

        if disc < 0.0:
            disc = 0.0

        return v0 + (math.sqrt(disc) - p0) / m

    else:
        # Distribution tables should only use histogram or linear interpolation.
        return v0


@njit
def sample_pmf(pmf, rng_state, data):
    xi = rng.lcg(rng_state)

    offset = pmf["cmf_offset"]
    length = pmf["cmf_length"]
    cmf = data[offset : offset + length]
    # Above is equivalent to: cmf = mcdc_get.pmf_distribution.cmf_all(pmf, data)

    idx = find_bin(xi, cmf)
    return mcdc_get.pmf_distribution.value(idx, pmf, data)


@njit
def sample_white_direction(nx, ny, nz, rng_state):
    # Sample polar cosine
    mu = math.sqrt(rng.lcg(rng_state))

    # Sample azimuthal direction
    azi = 2.0 * PI * rng.lcg(rng_state)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu**2) ** 0.5

    if nz != 1.0:
        B = (1.0 - nz**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * nz * cos_azi - ny * sin_azi) * C
        y = ny * mu + (ny * nz * cos_azi + nx * sin_azi) * C
        z = nz * mu - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the formula
    else:
        B = (1.0 - ny**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * ny * cos_azi - nz * sin_azi) * C
        z = nz * mu + (nz * ny * cos_azi + nx * sin_azi) * C
        y = ny * mu - cos_azi * Ac * B
    return x, y, z


@njit
def sample_multi_table(E, rng_state, multi_table, simulation, data):
    return _sample_multi_table(E, rng_state, multi_table, simulation, data, False)


@njit
def _sample_multi_table(E, rng_state, multi_table, simulation, data, scale):
    """Sample from a multi-table distribution."""

    # Get the grid
    offset = multi_table["grid_offset"]
    length = multi_table["grid_length"]
    grid = data[offset : offset + length]
    # Above is equivalent to: grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)

    # Helper flag for scaling later
    use_next_table = False

    # Default interpolation factor
    f = 0.0

    # Below grid: use first table without unit-base scaling.
    if E < grid[0]:
        idx = 0
        scale = False

    # Above grid: use last table without unit-base scaling.
    elif E > grid[-1]:
        idx = len(grid) - 1
        scale = False

    # Pick the table
    else:
        # Calculate Interpolation factor
        idx = find_bin(E, grid)
        E0 = grid[idx]
        E1 = grid[idx + 1]
        f = (E - E0) / (E1 - E0)

        # Sample which table to choose
        if rng.lcg(rng_state) < f:
            idx += 1
            use_next_table = True  # For scaling later if needed

    # Sample from the selected table
    ID = int(mcdc_get.multi_table_distribution.table_IDs(idx, multi_table, data))
    table_distribution = simulation["tabulated_distributions"][ID]
    sample = sample_tabulated(table_distribution, rng_state, simulation, data)

    # No scaling needed?
    if not scale:
        return sample

    # Retrieve the interval index
    if use_next_table:
        idx -= 1

    # PDF table indices
    ID0 = int(mcdc_get.multi_table_distribution.table_IDs(idx, multi_table, data))
    ID1 = int(mcdc_get.multi_table_distribution.table_IDs(idx + 1, multi_table, data))
    #
    pdf_ID = table_distribution["pdf_ID"]
    pdf_ID0 = simulation["tabulated_distributions"][ID0]["pdf_ID"]
    pdf_ID1 = simulation["tabulated_distributions"][ID1]["pdf_ID"]

    # The tables
    table = simulation["table_data"][pdf_ID]
    table0 = simulation["table_data"][pdf_ID0]
    table1 = simulation["table_data"][pdf_ID1]

    # Table's min
    val_min0 = mcdc_get.table_data.x(0, table0, data)
    val_min1 = mcdc_get.table_data.x(0, table1, data)

    # Table's max
    val_max0 = mcdc_get.table_data.x_last(table0, data)
    val_max1 = mcdc_get.table_data.x_last(table1, data)

    # The effective min and max
    val_min = val_min0 + f * (val_min1 - val_min0)
    val_max = val_max0 + f * (val_max1 - val_max0)

    # The sampled table's min and max
    val_low = mcdc_get.table_data.x(0, table, data)
    val_high = mcdc_get.table_data.x_last(table, data)
    return val_min + (sample - val_low) / (val_high - val_low) * (val_max - val_min)


@njit
def sample_maxwellian(E, rng_state, maxwellian, simulation, data):
    # Get nuclear temperature
    table = simulation["table_data"][maxwellian["nuclear_temperature_ID"]]
    nuclear_temperature = evaluate_table(E, table, data)
    restriction_energy = maxwellian["restriction_energy"]

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
    table = simulation["table_data"][evaporation["nuclear_temperature_ID"]]
    nuclear_temperature = evaluate_table(E, table, data)
    restriction_energy = evaporation["restriction_energy"]

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
def sample_kalbach_mann(E, rng_state, kalbach_mann, data):
    offset = kalbach_mann["energy_offset"]
    length = kalbach_mann["energy_length"]
    grid = data[offset : offset + length]
    # Above is equivalent to: grid = mcdc_get.kalbach_mann_distribution.energy_all(kalbach_mann, data)

    # Random numbers
    xi1 = rng.lcg(rng_state)
    xi2 = rng.lcg(rng_state)
    xi3 = rng.lcg(rng_state)
    xi4 = rng.lcg(rng_state)

    # Interpolation factor
    idx = find_bin(E, grid)
    E0 = grid[idx]
    E1 = grid[idx + 1]
    f = (E - E0) / (E1 - E0)

    # ==================================================================================
    # Min and max energy values for scaling
    # ==================================================================================

    # First table
    start = int(mcdc_get.kalbach_mann_distribution.offset(idx, kalbach_mann, data))
    end = int(mcdc_get.kalbach_mann_distribution.offset(idx + 1, kalbach_mann, data))
    E0_min = mcdc_get.kalbach_mann_distribution.energy_out(start, kalbach_mann, data)
    E0_max = mcdc_get.kalbach_mann_distribution.energy_out(end - 1, kalbach_mann, data)

    # Second table
    start = end
    if idx + 2 == len(grid):
        end = kalbach_mann["energy_length"]
    else:
        end = int(
            mcdc_get.kalbach_mann_distribution.offset(idx + 2, kalbach_mann, data)
        )
    E1_min = mcdc_get.kalbach_mann_distribution.energy_out(start, kalbach_mann, data)
    E1_max = mcdc_get.kalbach_mann_distribution.energy_out(end - 1, kalbach_mann, data)

    # The combination of the two tables
    E_min = E0_min + f * (E1_min - E0_min)
    E_max = E0_max + f * (E1_max - E0_max)

    # Sample which table to choose
    if xi1 < f:
        idx += 1

    # Get the table range
    start = int(mcdc_get.kalbach_mann_distribution.offset(idx, kalbach_mann, data))
    if idx + 1 == len(grid):
        end = kalbach_mann["energy_length"]
    else:
        end = int(
            mcdc_get.kalbach_mann_distribution.offset(idx + 1, kalbach_mann, data)
        )
    size = end - start

    # The CDF
    offset = kalbach_mann["cdf_offset"]
    cdf = data[start + offset : start + offset + size]
    # Above is equivalent to: cdf = mcdc_get.kalbach_mann_distribution.cdf_chunk(start, size, kalbach_mann, data)

    # Sample bin index
    idx = find_bin(xi2, cdf)
    c = cdf[idx]

    # Get the other values
    idx += start  # Apply the offset as these are not chunk-extracted like the cdf
    p0 = mcdc_get.kalbach_mann_distribution.pdf(idx, kalbach_mann, data)
    p1 = mcdc_get.kalbach_mann_distribution.pdf(idx + 1, kalbach_mann, data)
    E0 = mcdc_get.kalbach_mann_distribution.energy_out(idx, kalbach_mann, data)
    E1 = mcdc_get.kalbach_mann_distribution.energy_out(idx + 1, kalbach_mann, data)

    # Calculate the outgoing energy (not-scaled)
    m = (p1 - p0) / (E1 - E0)
    if m == 0.0:
        E_hat = E0 + (xi2 - c) / p0
    else:
        E_hat = E0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi2 - c)) - p0)

    # Scale against the bounds
    E_low = mcdc_get.kalbach_mann_distribution.energy_out(start, kalbach_mann, data)
    E_high = mcdc_get.kalbach_mann_distribution.energy_out(end - 1, kalbach_mann, data)
    E_new = E_min + (E_hat - E_low) / (E_high - E_low) * (E_max - E_min)

    # Precompound factor and angular slope
    R0 = mcdc_get.kalbach_mann_distribution.precompound_factor(idx, kalbach_mann, data)
    R1 = mcdc_get.kalbach_mann_distribution.precompound_factor(
        idx + 1, kalbach_mann, data
    )
    A0 = mcdc_get.kalbach_mann_distribution.angular_slope(idx, kalbach_mann, data)
    A1 = mcdc_get.kalbach_mann_distribution.angular_slope(idx + 1, kalbach_mann, data)
    #
    mE = (E_hat - E0) / (E1 - E0)
    R = R0 + mE * (R1 - R0)
    A = A0 + mE * (A1 - A0)

    # Calculate the angular coine
    T = (2.0 * xi4 - 1.0) * math.sinh(A)
    if xi3 > R:
        mu = math.log(T + math.sqrt(T**2 + 1.0)) / A
    else:
        mu = math.log(xi4 * math.exp(A) + (1.0 - xi4) * math.exp(-A)) / A

    return E_new, mu


@njit
def sample_tabulated_energy_angle(E, rng_state, table, data):
    offset = table["energy_offset"]
    length = table["energy_length"]
    grid = data[offset : offset + length]
    # Above is equivalent to: grid = mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)

    # Random numbers
    xi1 = rng.lcg(rng_state)
    xi2 = rng.lcg(rng_state)
    xi3 = rng.lcg(rng_state)

    # Interpolation factor
    idx = find_bin(E, grid)
    E0 = grid[idx]
    E1 = grid[idx + 1]
    f = (E - E0) / (E1 - E0)

    # ==================================================================================
    # Min and max energy values for scaling
    # ==================================================================================

    # First table
    start = int(mcdc_get.tabulated_energy_angle_distribution.offset(idx, table, data))
    end = int(mcdc_get.tabulated_energy_angle_distribution.offset(idx + 1, table, data))
    E0_min = mcdc_get.tabulated_energy_angle_distribution.energy_out(start, table, data)
    E0_max = mcdc_get.tabulated_energy_angle_distribution.energy_out(
        end - 1, table, data
    )

    # Second table
    start = end
    if idx + 2 == len(grid):
        end = table["energy_length"]
    else:
        end = int(
            mcdc_get.tabulated_energy_angle_distribution.offset(idx + 2, table, data)
        )
    E1_min = mcdc_get.tabulated_energy_angle_distribution.energy_out(start, table, data)
    E1_max = mcdc_get.tabulated_energy_angle_distribution.energy_out(
        end - 1, table, data
    )

    # The combination of the two tables
    E_min = E0_min + f * (E1_min - E0_min)
    E_max = E0_max + f * (E1_max - E0_max)

    # Sample which table to choose
    if xi1 < f:
        idx += 1

    # Get the table range
    start = int(mcdc_get.tabulated_energy_angle_distribution.offset(idx, table, data))
    if idx + 1 == len(grid):
        end = table["energy_length"]
    else:
        end = int(
            mcdc_get.tabulated_energy_angle_distribution.offset(idx + 1, table, data)
        )
    size = end - start

    # The CDF
    offset = table["cdf_offset"]
    cdf = data[start + offset : start + offset + size]
    # Above is equivalent to:
    # cdf = mcdc_get.tabulated_energy_angle_distribution.cdf_chunk(
    #     start, size, table, data
    # )

    # Sample bin index
    idx = find_bin(xi2, cdf)
    c = cdf[idx]

    # Get the other values
    idx_local = (
        idx + start
    )  # Apply the offset as these are not chunk-extracted like the cdf
    p0 = mcdc_get.tabulated_energy_angle_distribution.pdf(idx_local, table, data)
    p1 = mcdc_get.tabulated_energy_angle_distribution.pdf(idx_local + 1, table, data)
    E0 = mcdc_get.tabulated_energy_angle_distribution.energy_out(idx_local, table, data)
    E1 = mcdc_get.tabulated_energy_angle_distribution.energy_out(
        idx_local + 1, table, data
    )

    # Calculate the outgoing energy (not-scaled)
    m = (p1 - p0) / (E1 - E0)
    if m == 0.0:
        E_hat = E0 + (xi2 - c) / p0
    else:
        E_hat = E0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi2 - c)) - p0)

    # Scale against the bounds
    E_low = mcdc_get.tabulated_energy_angle_distribution.energy_out(start, table, data)
    E_high = mcdc_get.tabulated_energy_angle_distribution.energy_out(
        end - 1, table, data
    )
    E_new = E_min + (E_hat - E_low) / (E_high - E_low) * (E_max - E_min)

    # Determine angular table index
    if xi2 - cdf[idx] > cdf[idx + 1] - xi2:
        idx += 1

    # Get the angular table range
    start = int(
        mcdc_get.tabulated_energy_angle_distribution.cosine_offset_(idx, table, data)
    )
    if idx + 1 == len(grid):
        end = table["cosine_length"]
    else:
        end = int(
            mcdc_get.tabulated_energy_angle_distribution.cosine_offset_(
                idx + 1, table, data
            )
        )
    size = end - start

    # The CDF
    offset = table["cosine_cdf_offset"]
    cdf = data[start + offset : start + offset + size]
    # Above is equivalent to:
    # cdf = mcdc_get.tabulated_energy_angle_distribution.cosine_cdf_chunk(
    #     start, size, table, data
    # )

    # Sample bin index
    idx = find_bin(xi3, cdf)
    c = cdf[idx]

    # Get the other values
    idx += start  # Apply the offset as these are not chunk-extracted like the cdf
    p0 = mcdc_get.tabulated_energy_angle_distribution.cosine_pdf(idx, table, data)
    p1 = mcdc_get.tabulated_energy_angle_distribution.cosine_pdf(idx + 1, table, data)
    mu0 = mcdc_get.tabulated_energy_angle_distribution.cosine(idx, table, data)
    mu1 = mcdc_get.tabulated_energy_angle_distribution.cosine(idx + 1, table, data)

    m = (p1 - p0) / (mu1 - mu0)
    if m == 0.0:
        mu = mu0 + (xi3 - c) / p0
    else:
        mu = mu0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi3 - c)) - p0)

    return E_new, mu
