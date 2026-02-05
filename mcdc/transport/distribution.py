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
    PI,
)
from mcdc.transport.data import evaluate_table
from mcdc.transport.util import find_bin, linear_interpolation

# ======================================================================================
# General distribution samplers
# ======================================================================================


@njit
def sample_distribution(E, distribution, rng_state, mcdc, data, scale=False):
    distribution_type = distribution["child_type"]
    ID = distribution["child_ID"]

    if distribution_type == DISTRIBUTION_TABULATED:
        table = mcdc["tabulated_distributions"][ID]
        return sample_tabulated(table, rng_state, data)

    elif distribution_type == DISTRIBUTION_MULTITABLE:
        multi_table = mcdc["multi_table_distributions"][ID]
        return sample_multi_table(E, rng_state, multi_table, data, scale)

    elif distribution_type == DISTRIBUTION_LEVEL_SCATTERING:
        level_scattering = mcdc["level_scattering_distributions"][ID]
        return sample_level_scattering(E, level_scattering)

    elif distribution_type == DISTRIBUTION_EVAPORATION:
        evaporation = mcdc["evaporation_distributions"][ID]
        return sample_evaporation(E, rng_state, evaporation, mcdc, data)

    elif distribution_type == DISTRIBUTION_MAXWELLIAN:
        maxwellian = mcdc["maxwellian_distributions"][ID]
        return sample_maxwellian(E, rng_state, maxwellian, mcdc, data)

    # TODO: Should not get here
    else:
        return -1.0


@njit
def sample_correlated_distribution(E, distribution, rng_state, mcdc, data, scale=False):
    distribution_type = distribution["child_type"]
    ID = distribution["child_ID"]

    if distribution_type == DISTRIBUTION_KALBACH_MANN:
        kalbach_mann = mcdc["kalbach_mann_distributions"][ID]
        return sample_kalbach_mann(E, rng_state, kalbach_mann, data)

    elif distribution_type == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
        table = mcdc["tabulated_energy_angle_distributions"][ID]
        return sample_tabulated_energy_angle(E, rng_state, table, data)

    elif distribution_type == DISTRIBUTION_N_BODY:
        nbody = mcdc["nbody_distributions"][ID]
        E_out = sample_tabulated(nbody, rng_state, data)
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
def sample_direction(polar, azimuthal, polar_coordinate, rng_state):
    # Sample polar cosine and azimuthal angle
    mu = sample_uniform(polar[0], polar[1], rng_state)
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
def sample_tabulated(table, rng_state, data):
    xi = rng.lcg(rng_state)
    idx = find_bin(xi, mcdc_get.tabulated_distribution.cdf_all(table, data))
    cdf_low = mcdc_get.tabulated_distribution.cdf(idx, table, data)
    cdf_high = mcdc_get.tabulated_distribution.cdf(idx + 1, table, data)
    value_low = mcdc_get.tabulated_distribution.value(idx, table, data)
    value_high = mcdc_get.tabulated_distribution.value(idx + 1, table, data)
    return linear_interpolation(xi, cdf_low, cdf_high, value_low, value_high)


@njit
def sample_pmf(pmf, rng_state, data):
    xi = rng.lcg(rng_state)
    idx = find_bin(xi, mcdc_get.pmf_distribution.cmf_all(pmf, data))
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
def sample_multi_table(E, rng_state, multi_table, data, scale=False):
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)

    # Edge cases
    if E < grid[0]:
        idx = 0
        scale = False
    elif E > grid[-1]:
        idx = len(grid) - 1
        scale = False
    else:
        # Interpolation factor
        idx = find_bin(E, grid)
        E0 = grid[idx]
        E1 = grid[idx + 1]
        f = (E - E0) / (E1 - E0)

        # Min and max values for scaling
        val_min = 0.0
        val_max = 1.0
        if scale:
            # First table
            start = int(
                mcdc_get.multi_table_distribution.offset(idx, multi_table, data)
            )
            end = int(
                mcdc_get.multi_table_distribution.offset(idx + 1, multi_table, data)
            )
            val0_min = mcdc_get.multi_table_distribution.value(start, multi_table, data)
            val0_max = mcdc_get.multi_table_distribution.value(
                end - 1, multi_table, data
            )

            # Second table
            start = end
            if idx + 2 == len(grid):
                end = multi_table["value_length"]
            else:
                end = int(
                    mcdc_get.multi_table_distribution.offset(idx + 2, multi_table, data)
                )
            val1_min = mcdc_get.multi_table_distribution.value(start, multi_table, data)
            val1_max = mcdc_get.multi_table_distribution.value(
                end - 1, multi_table, data
            )

            # Both
            val_min = val0_min + f * (val1_min - val0_min)
            val_max = val0_max + f * (val1_max - val0_max)

        # Sample which table to choose
        if rng.lcg(rng_state) > f:
            idx += 1

    # Get the table range
    start = int(mcdc_get.multi_table_distribution.offset(idx, multi_table, data))
    if idx + 1 == len(grid):
        end = multi_table["value_length"]
    else:
        end = int(mcdc_get.multi_table_distribution.offset(idx + 1, multi_table, data))
    size = end - start

    # The CDF
    cdf = mcdc_get.multi_table_distribution.cdf_chunk(start, size, multi_table, data)

    # Generate random numbers
    xi = rng.lcg(rng_state)

    # Sample bin index
    idx = find_bin(xi, cdf)
    c = cdf[idx]

    # Get the other values
    idx += start  # Apply the offset as these are not chunk-extracted like the cdf
    p0 = mcdc_get.multi_table_distribution.pdf(idx, multi_table, data)
    p1 = mcdc_get.multi_table_distribution.pdf(idx + 1, multi_table, data)
    val0 = mcdc_get.multi_table_distribution.value(idx, multi_table, data)
    val1 = mcdc_get.multi_table_distribution.value(idx + 1, multi_table, data)

    m = (p1 - p0) / (val1 - val0)
    if m == 0.0:
        sample = val0 + (xi - c) / p0
    else:
        sample = val0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi - c)) - p0)

    if not scale:
        return sample

    # Scale against the bounds
    val_low = mcdc_get.multi_table_distribution.value(start, multi_table, data)
    val_high = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)
    return val_min + (sample - val_low) / (val_high - val_low) * (val_max - val_min)


@njit
def sample_maxwellian(E, rng_state, maxwellian, mcdc, data):
    # Get nuclear temperature
    table = mcdc["table_data"][maxwellian["nuclear_temperature_ID"]]
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
def sample_evaporation(E, rng_state, evaporation, mcdc, data):
    # Get nuclear temperature
    table = mcdc["table_data"][evaporation["nuclear_temperature_ID"]]
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
    grid = mcdc_get.kalbach_mann_distribution.energy_all(kalbach_mann, data)

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
    if xi1 > f:
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
    cdf = mcdc_get.kalbach_mann_distribution.cdf_chunk(start, size, kalbach_mann, data)

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
    grid = mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)

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
    if xi1 > f:
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
    cdf = mcdc_get.tabulated_energy_angle_distribution.cdf_chunk(
        start, size, table, data
    )

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
    cdf = mcdc_get.tabulated_energy_angle_distribution.cosine_cdf_chunk(
        start, size, table, data
    )

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
