from numba import njit

####

from mcdc.constant import COINCIDENCE_TOLERANCE, COINCIDENCE_TOLERANCE_TIME
import mcdc.mcdc_get as mcdc_get
import mcdc.transport.rng as rng

from mcdc.transport.distribution import (
    sample_uniform,
    sample_tabulated,
    sample_pmf,
    sample_white_direction,
    sample_isotropic_direction,
    sample_direction,
)
from mcdc.transport.util import find_bin_with_rules


@njit
def source_particle(particle_container, seed, simulation, data):
    particle = particle_container[0]
    particle["rng_seed"] = seed

    # Sample source
    # TODO: use cdf and binary search instead
    xi = rng.lcg(particle_container)
    tot = 0.0
    source = simulation["sources"][0]
    for source in simulation["sources"]:
        tot += source["probability"]
        if tot >= xi:
            break

    # Position
    if source["point_source"]:
        x = source["point"][0]
        y = source["point"][1]
        z = source["point"][2]
    else:
        x = sample_position_axis(
            source["uniform_x"],
            source["x"],
            source["x_pdf_ID"],
            particle_container,
            simulation,
            data,
        )
        y = sample_position_axis(
            source["uniform_y"],
            source["y"],
            source["y_pdf_ID"],
            particle_container,
            simulation,
            data,
        )
        z = sample_position_axis(
            source["uniform_z"],
            source["z"],
            source["z_pdf_ID"],
            particle_container,
            simulation,
            data,
        )

    # Direction
    if source["isotropic_direction"]:
        ux, uy, uz = sample_isotropic_direction(particle_container)
    elif source["white_direction"]:
        rx = source["direction"][0]
        ry = source["direction"][1]
        rz = source["direction"][2]
        ux, uy, uz = sample_white_direction(rx, ry, rz, particle_container)
    elif source["mono_direction"]:
        ux = source["direction"][0]
        uy = source["direction"][1]
        uz = source["direction"][2]
    else:
        ux, uy, uz = sample_direction(
            source["polar_cosine"],
            source["azimuthal"],
            source["direction"],
            particle_container,
        )

    # Energy
    if source["mono_energetic"]:
        E = source["energy"]
    elif source["discrete_energy"]:
        ID = source["energy_pmf_ID"]
        sub_ID = simulation["distributions"][ID]["sub_ID"]
        pmf = simulation["pmf_distributions"][sub_ID]
        E = sample_pmf(pmf, particle_container, data)
    else:
        ID = source["energy_pdf_ID"]
        sub_ID = simulation["distributions"][ID]["sub_ID"]
        table = simulation["tabulated_distributions"][sub_ID]
        E = sample_tabulated(table, particle_container, simulation, data)

    # Time
    if source["discrete_time"]:
        t = source["time"]
    else:
        t = sample_uniform(
            source["time_range"][0], source["time_range"][1], particle_container
        )

    # Motion translation
    if source["moving"]:
        # Get moving interval index wrt the given time
        time_grid = mcdc_get.source.move_time_grid_all(source, data)

        tolerance = COINCIDENCE_TOLERANCE_TIME
        go_lower = False
        idx = find_bin_with_rules(t, time_grid, tolerance, go_lower)

        # Coinciding cases
        if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
            idx += 1

        # Source move translations
        trans_0 = mcdc_get.source.move_translations_vector(idx, source, data)

        # Source move velocities
        V = mcdc_get.source.move_velocities_vector(idx, source, data)

        # Source move time grid
        time_0 = mcdc_get.source.move_time_grid(idx, source, data)

        # Translate the particle
        t_local = t - time_0
        x += trans_0[0] + V[0] * t_local
        y += trans_0[1] + V[1] * t_local
        z += trans_0[2] + V[2] * t_local

    # Make and return particle
    particle["x"] = x
    particle["y"] = y
    particle["z"] = z
    particle["t"] = t
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz
    particle["E"] = E
    particle["w"] = 1.0
    particle["particle_type"] = source["particle_type"]


@njit
def sample_position_axis(uniform, bounds, pdf_ID, rng_state, simulation, data):
    """Sample one independent source-position coordinate."""
    if uniform:
        return sample_uniform(bounds[0], bounds[1], rng_state)

    sub_ID = simulation["distributions"][pdf_ID]["sub_ID"]
    table = simulation["tabulated_distributions"][sub_ID]
    return sample_tabulated(table, rng_state, simulation, data)
