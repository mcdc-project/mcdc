from numba import njit

####

import mcdc.code_factory.gpu.adapt as adapt
import mcdc.mcdc_get as mcdc_get
import mcdc.transport.mesh as mesh_module
import mcdc.transport.physics as physics

from mcdc.constant import (
    AXIS_T,
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    COINCIDENCE_TOLERANCE,
    COINCIDENCE_TOLERANCE_TIME,
    INF,
    MULTIPLIER_ENERGY,
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_FISSION,
    REACTION_TOTAL,
    SCORE_FLUX,
    SCORE_DENSITY,
    SCORE_COLLISION,
    SCORE_CAPTURE,
    SCORE_FISSION,
    SCORE_NET_CURRENT,
)
from mcdc.transport.geometry.surface import get_normal_component
from mcdc.transport.tally.filter import get_filter_indices
from mcdc.transport.util import atomic_add


@njit
def make_scores(particle_container, flux, tally, idx_base, mcdc, data):
    particle = particle_container[0]
    speed = physics.particle_speed(particle_container, mcdc, data)

    multiplier = 1.0
    for i_multiplier in range(tally["multipliers_length"]):
        multiplier_type = mcdc_get.tally.multipliers(i_multiplier, tally, data)
        if multiplier_type == MULTIPLIER_ENERGY:
            multiplier *= particle["E"]

    for i_score in range(tally["scores_length"]):
        score_type = mcdc_get.tally.scores(i_score, tally, data)
        score = 0.0
        if score_type == SCORE_FLUX:
            score = flux
        elif score_type == SCORE_DENSITY:
            score = flux / speed
        elif score_type == SCORE_COLLISION:
            score = flux * physics.macro_xs(
                REACTION_TOTAL, particle_container, mcdc, data
            )
        elif score_type == SCORE_CAPTURE:
            score = flux * physics.macro_xs(
                REACTION_NEUTRON_CAPTURE, particle_container, mcdc, data
            )
        elif score_type == SCORE_FISSION:
            score = flux * physics.macro_xs(
                REACTION_NEUTRON_FISSION, particle_container, mcdc, data
            )
        elif score_type == SCORE_NET_CURRENT:
            surface = mcdc["surfaces"][particle["surface_ID"]]
            mu = get_normal_component(particle_container, speed, surface, data)
            score = flux * mu
        atomic_add(data, idx_base + i_score, score * multiplier)


@njit
def tracklength_tally(particle_container, distance, tally, mcdc, data):
    particle = particle_container[0]
    tally_base = mcdc["tallies"][tally["parent_ID"]]

    # Get filter indices
    MG_mode = mcdc["settings"]["multigroup_mode"]
    i_mu, i_azi, i_energy, i_time = get_filter_indices(
        particle_container, tally_base, data, MG_mode
    )

    # No score if outside non-changing phase-space bins
    if i_mu == -1 or i_azi == -1 or i_energy == -1:
        return

    # Particle/track properties
    ut = 1.0 / physics.particle_speed(particle_container, mcdc, data)
    t = particle["t"]
    t_final = t + ut * distance

    # No score if particle does not cross the time bins
    t_min = mcdc_get.tally.time(0, tally_base, data)
    t_max = mcdc_get.tally.time_last(tally_base, data)
    if (
        t_final < t_min + COINCIDENCE_TOLERANCE_TIME
        or t > t_max - COINCIDENCE_TOLERANCE_TIME
    ):
        return

    # Get the appropriate time index if needed
    if t < t_min + COINCIDENCE_TOLERANCE_TIME:
        i_time = 0

    # Tally base index
    idx_base = (
        tally_base["bin_offset"]
        + i_mu * tally_base["stride_mu"]
        + i_azi * tally_base["stride_azi"]
        + i_energy * tally_base["stride_energy"]
        + i_time * tally_base["stride_time"]
    )

    # Sweep through the distance
    distance_swept = 0.0
    while distance_swept < distance - COINCIDENCE_TOLERANCE:
        # The next time grid
        t_next = mcdc_get.tally.time(i_time + 1, tally_base, data)

        # Get the distance to score in this segment
        if t_final < t_next - COINCIDENCE_TOLERANCE_TIME:
            distance_scored = distance - distance_swept
        else:
            distance_scored = (t_next - t) / ut

        # Score
        flux = distance_scored * particle["w"]
        make_scores(particle_container, flux, tally_base, idx_base, mcdc, data)

        # Accumulate distance swept
        distance_swept += distance_scored

        # Increment the time
        t += distance_scored * ut

        # Increment index
        i_time += 1
        idx_base += tally_base["stride_time"]

        # Check if it is the last segment
        #   The rest of the distance is not scored
        if i_time == tally_base["time_length"]:
            return


@njit
def surface_tally(particle_container, surface, tally, mcdc, data):
    particle = particle_container[0]
    tally_base = mcdc["tallies"][tally["parent_ID"]]

    # Get filter indices
    MG_mode = mcdc["settings"]["multigroup_mode"]
    i_mu, i_azi, i_energy, i_time = get_filter_indices(
        particle_container, tally_base, data, MG_mode
    )

    # No score if outside non-changing phase-space bins
    if i_mu == -1 or i_azi == -1 or i_energy == -1 or i_time == -1:
        return

    # Tally index
    idx_base = (
        tally_base["bin_offset"]
        + i_mu * tally_base["stride_mu"]
        + i_azi * tally_base["stride_azi"]
        + i_energy * tally_base["stride_energy"]
        + i_time * tally_base["stride_time"]
    )

    # Flux
    speed = physics.particle_speed(particle_container, mcdc, data)
    mu = get_normal_component(particle_container, speed, surface, data)
    flux = particle["w"] / abs(mu)

    # Score
    make_scores(particle_container, flux, tally_base, idx_base, mcdc, data)


@njit
def mesh_tally(particle_container, distance, tally, mcdc, data):
    particle = particle_container[0]
    tally_base = mcdc["tallies"][tally["parent_ID"]]

    # Get filter indices
    MG_mode = mcdc["settings"]["multigroup_mode"]
    i_mu, i_azi, i_energy, i_time = get_filter_indices(
        particle_container, tally_base, data, MG_mode
    )

    # No score if outside non-changing phase-space bins
    if i_mu == -1 or i_azi == -1 or i_energy == -1:
        return

    # Get the mesh
    mesh = mcdc["meshes"][tally["mesh_ID"]]

    # Particle/track properties
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    ut = 1.0 / physics.particle_speed(particle_container, mcdc, data)
    x_final = x + ux * distance
    y_final = y + uy * distance
    z_final = z + uz * distance
    t_final = t + ut * distance

    # No score if particle does not cross the time bins
    t_min = mcdc_get.tally.time(0, tally_base, data)
    t_max = mcdc_get.tally.time_last(tally_base, data)
    if (
        t_final < t_min + COINCIDENCE_TOLERANCE_TIME
        or t > t_max - COINCIDENCE_TOLERANCE_TIME
    ):
        return

    # Get the appropriate time index if needed
    if t < t_min:
        i_time = 0

    # Get mesh bin indices
    i_x, i_y, i_z = mesh_module.get_indices(particle_container, mesh, mcdc, data)

    # No score if particle does not cross the mesh bins
    # Also get the appropriate index if needed
    x_min = mesh_module.get_x(0, mesh, mcdc, data)
    x_max = mesh_module.get_x(mesh["Nx"], mesh, mcdc, data)
    if ux > 0.0:
        if x_final < x_min + COINCIDENCE_TOLERANCE or x > x_max - COINCIDENCE_TOLERANCE:
            return
        if x < x_min + COINCIDENCE_TOLERANCE:
            i_x = 0
    else:
        if x < x_min + COINCIDENCE_TOLERANCE or x_final > x_max - COINCIDENCE_TOLERANCE:
            return
        if x > x_max - COINCIDENCE_TOLERANCE:
            i_x = mesh["Nx"]
    y_min = mesh_module.get_y(0, mesh, mcdc, data)
    y_max = mesh_module.get_y(mesh["Ny"], mesh, mcdc, data)
    if uy > 0.0:
        if y_final < y_min + COINCIDENCE_TOLERANCE or y > y_max - COINCIDENCE_TOLERANCE:
            return
        if y < y_min + COINCIDENCE_TOLERANCE:
            i_y = 0
    else:
        if y < y_min + COINCIDENCE_TOLERANCE or y_final > y_max - COINCIDENCE_TOLERANCE:
            return
        if y > y_max - COINCIDENCE_TOLERANCE:
            i_y = mesh["Ny"]
    z_min = mesh_module.get_z(0, mesh, mcdc, data)
    z_max = mesh_module.get_z(mesh["Nz"], mesh, mcdc, data)
    if uz > 0.0:
        if z_final < z_min + COINCIDENCE_TOLERANCE or z > z_max - COINCIDENCE_TOLERANCE:
            return
        if z < z_min + COINCIDENCE_TOLERANCE:
            i_z = 0
    else:
        if z < z_min + COINCIDENCE_TOLERANCE or z_final > z_max - COINCIDENCE_TOLERANCE:
            return
        if z > z_max - COINCIDENCE_TOLERANCE:
            i_z = mesh["Nz"]

    # Tally base index
    idx_base = (
        tally_base["bin_offset"]
        + i_mu * tally_base["stride_mu"]
        + i_azi * tally_base["stride_azi"]
        + i_energy * tally_base["stride_energy"]
        + i_time * tally_base["stride_time"]
        + i_x * tally["stride_x"]
        + i_y * tally["stride_y"]
        + i_z * tally["stride_z"]
    )

    # Sweep through the distance
    distance_swept = 0.0
    while distance_swept < distance - COINCIDENCE_TOLERANCE:
        # ==============================================================================
        # Find distances to the mesh grids
        # ==============================================================================

        # x-direction
        if ux == 0.0:
            dx = INF
        else:
            if ux > 0.0:
                x_next = mesh_module.get_x(i_x + 1, mesh, mcdc, data)
                x_next = min(x_next, x_final)
            else:
                x_next = mesh_module.get_x(i_x, mesh, mcdc, data)
                x_next = max(x_next, x_final)
            dx = (x_next - x) / ux

        # y-direction
        if uy == 0.0:
            dy = INF
        else:
            if uy > 0.0:
                y_next = mesh_module.get_y(i_y + 1, mesh, mcdc, data)
                y_next = min(y_next, y_final)
            else:
                y_next = mesh_module.get_y(i_y, mesh, mcdc, data)
                y_next = max(y_next, y_final)
            dy = (y_next - y) / uy

        # z-direction
        if uz == 0.0:
            dz = INF
        else:
            if uz > 0.0:
                z_next = mesh_module.get_z(i_z + 1, mesh, mcdc, data)
                z_next = min(z_next, z_final)
            else:
                z_next = mesh_module.get_z(i_z, mesh, mcdc, data)
                z_next = max(z_next, z_final)
            dz = (z_next - z) / uz

        # t-direction
        t_next = mcdc_get.tally.time(i_time + 1, tally_base, data)
        dt = (min(t_next, t_final) - t) / ut

        # ==============================================================================
        # Evaluate grid crossings
        # ==============================================================================

        distance_scored = INF
        axis_crossed = -1
        if dx <= distance_scored:
            axis_crossed = AXIS_X
            distance_scored = dx
        if dy <= distance_scored:
            axis_crossed = AXIS_Y
            distance_scored = dy
        if dz <= distance_scored:
            axis_crossed = AXIS_Z
            distance_scored = dz
        if dt <= distance_scored:
            axis_crossed = AXIS_T
            distance_scored = dt

        # Score
        flux = distance_scored * particle["w"]
        make_scores(particle_container, flux, tally_base, idx_base, mcdc, data)

        # Accumulate distance swept
        distance_swept += distance_scored

        # Move the 4D position
        x += distance_scored * ux
        y += distance_scored * uy
        z += distance_scored * uz
        t += distance_scored * ut

        # Increment index and check if out of bounds
        if axis_crossed == AXIS_X:
            if ux > 0.0:
                i_x += 1
                if i_x == mesh["Nx"]:
                    break
                idx_base += tally["stride_x"]
            else:
                i_x -= 1
                if i_x == -1:
                    break
                idx_base -= tally["stride_x"]
        elif axis_crossed == AXIS_Y:
            if uy > 0.0:
                i_y += 1
                if i_y == mesh["Ny"]:
                    break
                idx_base += tally["stride_y"]
            else:
                i_y -= 1
                if i_y == -1:
                    break
                idx_base -= tally["stride_y"]
        elif axis_crossed == AXIS_Z:
            if uz > 0.0:
                i_z += 1
                if i_z == mesh["Nz"]:
                    break
                idx_base += tally["stride_z"]
            else:
                i_z -= 1
                if i_z == -1:
                    break
                idx_base -= tally["stride_z"]
        elif axis_crossed == AXIS_T:
            i_time += 1
            if i_time == tally_base["time_length"] - 1:
                break
            idx_base += tally_base["stride_time"]


# =============================================================================
# Eigenvalue tally
# =============================================================================


@njit
def eigenvalue_tally(particle_container, distance, mcdc, data):
    particle = particle_container[0]
    flux = distance * particle["w"]

    # Get nu-fission
    nuSigmaF = physics.neutron_production_xs(
        REACTION_NEUTRON_FISSION, particle_container, mcdc, data
    )

    # Fission production (needed even during inactive cycle)
    atomic_add(mcdc["eigenvalue_tally_nuSigmaF"], 0, flux * nuSigmaF)

    # Done, if inactive
    if not mcdc["cycle_active"]:
        return

    # ==================================================================================
    # Neutron density
    # ==================================================================================

    v = physics.particle_speed(particle_container, mcdc, data)
    n_density = flux / v
    atomic_add(mcdc["eigenvalue_tally_n"], 0, n_density)

    # Maximum neutron density
    if mcdc["n_max"] < n_density:
        mcdc["n_max"] = n_density

    # ==================================================================================
    # TODO: Delayed neutron precursor density
    # ==================================================================================
    return
    # Get the decay-wighted multiplicity
    total = 0.0
    if mcdc["settings"]["multigroup_mode"]:
        g = particle["g"]
        for j in range(J):
            nu_d = mcdc_get.material.mgxs_nu_d(g, j, material, data)
            decay = mcdc_get.material.mgxs_decay_rate(j, material, data)
            total += nu_d / decay
    else:
        E = P["E"]
        for i in range(material["N_nuclide"]):
            ID_nuclide = material["nuclide_IDs"][i]
            nuclide = mcdc["nuclides"][ID_nuclide]
            if not nuclide["fissionable"]:
                continue
            for j in range(J):
                nu_d = get_nu_group(NU_FISSION_DELAYED, nuclide, E, j)
                decay = nuclide["ce_decay"][j]
                total += nu_d / decay

    SigmaF = physics.macro_xs(REACTION_NEUTRON_FISSION, particle_container, mcdc, data)
    C_density = flux * total * SigmaF / mcdc["k_eff"]
    atomic_add(mcdc["eigenvalue_tally_C"], 0, C_density)

    # Maximum precursor density
    if mcdc["C_max"] < C_density:
        mcdc["C_max"] = C_density
