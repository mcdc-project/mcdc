import math

####

import mcdc.mcdc_get as mcdc_get

from mcdc.src.algorithm import binary_search


def sample_multipdf(x, xi1, xi2, multipdf, data):
    # Interpolation factor
    grid = mcdc_get.multipdf.grid_all(multipdf, data)
    idx = binary_search(x, grid)
    x0 = grid[idx]
    x1 = grid[idx + 1]
    f = (x - x0) - (x1 - x0)

    # Sample which table to choose
    if xi1 > f:
        idx += 1

    # Get the table range
    start = int(mcdc_get.multipdf.offset(idx, multipdf, data))
    if idx + 1 == len(grid):
        end = len(grid)
    else:
        end = int(mcdc_get.multipdf.offset(idx + 1, multipdf, data))
    size = end - start

    # The CDF
    cdf = mcdc_get.multipdf.cdf_chunk(start, size, multipdf, data)

    # Sample bin index
    idx = binary_search(xi2, cdf)
    p0 = mcdc_get.multipdf.pdf(idx, multipdf, data)
    p1 = mcdc_get.multipdf.pdf(idx + 1, multipdf, data)
    val0 = mcdc_get.multipdf.value(idx, multipdf, data)
    val1 = mcdc_get.multipdf.value(idx + 1, multipdf, data)

    m = (p1 - p0) / (val1 - val0)
    c = cdf[idx]
    if m == 0.0:
        sample = val0 + (xi2 - c) / p0
    else:
        sample = val0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi2 - c)) - p0)

    return sample
