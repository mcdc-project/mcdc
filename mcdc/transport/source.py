from numba import njit

####

import mcdc.transport.rng as rng

from mcdc.transport.distribution import (
    sample_uniform,
    sample_tabulated,
    sample_pmf,
    sample_white_direction,
    sample_isotropic_direction,
)

import mcdc.code_factory.adapt as adapt


@adapt.toggle("sensitivity")
def _init_resp_cum(P_rec_arr):
    """Initialize per-particle response accumulator for a new source history."""
    P_rec_arr[0]["resp_cum"][:] = 0.0


@njit
def source_particle(P_rec_arr, seed, mcdc, data):
    P_rec = P_rec_arr[0]
    P_rec["rng_seed"] = seed

    # Sample source
    # TODO: use cdf and binary search instead
    xi = rng.lcg(P_rec_arr)
    tot = 0.0
    for source in mcdc["sources"]:
        tot += source["probability"]
        if tot >= xi:
            break

    # Position
    if source["point_source"]:
        x = source["point"][0]
        y = source["point"][1]
        z = source["point"][2]
    else:
        x = sample_uniform(source["x"][0], source["x"][1], P_rec_arr)
        y = sample_uniform(source["y"][0], source["y"][1], P_rec_arr)
        z = sample_uniform(source["z"][0], source["z"][1], P_rec_arr)

    # Direction
    if source["isotropic_direction"]:
        ux, uy, uz = sample_isotropic_direction(P_rec_arr)
    elif source["white_direction"]:
        rx = source["direction"][0]
        ry = source["direction"][1]
        rz = source["direction"][2]
        ux, uy, uz = sample_white_direction(rx, ry, rz, P_rec_arr)
    elif source["mono_direction"]:
        ux = source["direction"][0]
        uy = source["direction"][1]
        uz = source["direction"][2]

    # Energy
    if mcdc["settings"]["multigroup_mode"]:
        E = 0.0
        if source["mono_energetic"]:
            g = source["energy_group"]
        else:
            ID = source["energy_group_pmf_ID"]
            pmf = mcdc["pmf_distributions"][ID]
            g = sample_pmf(pmf, P_rec_arr, data)
    else:
        g = 0
        if source["mono_energetic"]:
            E = source["energy"]
        else:
            ID = source["energy_pdf_ID"]
            table = mcdc["tabulated_distributions"][ID]
            E = sample_tabulated(table, P_rec_arr, data)

    # Time
    if source["discrete_time"]:
        t = source["time"]
    else:
        t = sample_uniform(source["time_range"][0], source["time_range"][1], P_rec_arr)

    # Make and return particle
    P_rec["x"] = x
    P_rec["y"] = y
    P_rec["z"] = z
    P_rec["t"] = t
    P_rec["ux"] = ux
    P_rec["uy"] = uy
    P_rec["uz"] = uz
    P_rec["g"] = g
    P_rec["E"] = E
    P_rec["w"] = 1.0
    P_rec["particle_type"] = source["particle_type"]

    _init_resp_cum(P_rec_arr)
