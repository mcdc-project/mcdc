import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import LIGHT_SPEED
from mcdc.transport.util import (
    find_bin,
    find_bin_with_rules,
    interpolate,
    linear_interpolation,
)
from mcdc.transport.linalg import make_direction_basis


@njit
def relativistic_energy_from_speed(speed, mass):
    """Return kinetic energy without cancellation in ``gamma - 1``."""
    beta_squared = (speed / LIGHT_SPEED) ** 2
    inverse_gamma = math.sqrt(1.0 - beta_squared)
    gamma_minus_one = beta_squared / (inverse_gamma * (1.0 + inverse_gamma))
    return mass * gamma_minus_one


# ======================================================================================
# Shared-grid cross-section evaluation
# ======================================================================================


@njit
def evaluate_xs_energy_grid(E, grid):
    """Locate a right-side interval, bounding lookup to the available grid."""
    E = min(max(E, grid[0]), grid[-1])
    idx = find_bin_with_rules(E, grid, 0.0, False)
    if idx < 0:
        idx = len(grid) - 2
    return idx, grid[idx], grid[idx + 1]


@njit
def interpolate_xs(E, E0, E1, xs0, xs1):
    """Interpolate a linear cross-section segment using the upper value at a jump."""
    if E1 <= E0:
        return xs1
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def evaluate_reaction_xs(E, grid, offset, values, laws, boundaries):
    """Evaluate a shared-grid segment using its own interpolation regions.

    Values are zero below a positive-offset threshold and constant beyond the
    shared grid. Repeated energies select the right-hand limit of a jump.
    Interpolation boundaries are exclusive point indices local to the segment.
    """
    E = max(E, grid[0])
    if E < grid[offset]:
        return 0.0
    if len(values) == 1 or E >= grid[-1]:
        return values[-1]
    idx, E0, E1 = evaluate_xs_energy_grid(E, grid)
    local = idx - offset
    law = laws[-1]
    for i in range(len(boundaries)):
        if local + 1 < boundaries[i]:
            law = laws[i]
            break
    return interpolate(E, E0, E1, values[local], values[local + 1], law)


@njit
def evaluate_neutron_xs_energy_grid(e, nuclide, data):
    energy_grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)

    return evaluate_xs_energy_grid(e, energy_grid)


@njit
def evaluate_electron_xs_energy_grid(e, element, data):
    energy_grid = mcdc_get.element.electron_xs_energy_grid_all(element, data)
    idx = find_bin(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1


@njit
def evaluate_photon_xs_energy_grid(e, element, data):
    energy_grid = mcdc_get.element.photon_xs_energy_grid_all(element, data)
    return evaluate_xs_energy_grid(e, energy_grid)


@njit
def evaluate_proton_xs_energy_grid(e, nuclide, data):
    offset = nuclide["proton_xs_energy_grid_offset"]
    length = nuclide["proton_xs_energy_grid_length"]
    energy_grid = data[offset : offset + length]

    idx = find_bin(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1


@njit
def scatter_direction(ux, uy, uz, mu0, azi):
    """Rotate a unit direction by polar cosine mu0 and azimuth azi in radians."""
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    sin_polar = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    u1x, u1y, u1z, u2x, u2y, u2z = make_direction_basis(ux, uy, uz)

    ux_new = sin_polar * cos_azi * u1x + sin_polar * sin_azi * u2x + mu0 * ux
    uy_new = sin_polar * cos_azi * u1y + sin_polar * sin_azi * u2y + mu0 * uy
    uz_new = sin_polar * cos_azi * u1z + sin_polar * sin_azi * u2z + mu0 * uz

    return ux_new, uy_new, uz_new
