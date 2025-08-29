import math

from numba import njit

####

import mcdc.kernel as kernel
import mcdc.mcdc_get as mcdc_get

from mcdc.constant import PI
from mcdc.util import binary_search


@njit
def evaluate_xs_energy_grid(e, nuclide, data):
    energy_grid = mcdc_get.nuclide.xs_energy_grid_all(nuclide, data)
    idx = binary_search(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1


@njit
def sample_isotropic_direction(particle_container):
    # Sample polar cosine and azimuthal angle uniformly
    mu = 2.0 * kernel.rng(particle_container) - 1.0
    azi = 2.0 * PI * kernel.rng(particle_container)

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    y = math.cos(azi) * c
    z = math.sin(azi) * c
    x = mu
    return x, y, z


@njit
def scatter_direction(ux, uy, uz, mu0, azi):
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu0**2) ** 0.5

    if uz != 1.0:
        B = (1.0 - uz**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uz * cos_azi - uy * sin_azi) * C
        uy_new = uy * mu0 + (uy * uz * cos_azi + ux * sin_azi) * C
        uz_new = uz * mu0 - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the scattering formula
    else:
        B = (1.0 - uy**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uy * cos_azi - uz * sin_azi) * C
        uz_new = uz * mu0 + (uz * uy * cos_azi + ux * sin_azi) * C
        uy_new = uy * mu0 - cos_azi * Ac * B

    return ux_new, uy_new, uz_new
