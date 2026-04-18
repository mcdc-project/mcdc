import math
import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
import mcdc.transport.util as util

from mcdc.constant import (
    ELECTRON_CUTOFF_ENERGY,
    ELECTRON_MASS,
    ELECTRON_REACTION_BREMSSTRAHLUNG,
    ELECTRON_REACTION_EXCITATION,
    ELECTRON_REACTION_ELASTIC_SCATTERING,
    ELECTRON_REACTION_IONIZATION,
    ELECTRON_REACTION_TOTAL,
    LIGHT_SPEED,
    PI,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_distribution,
)
from mcdc.transport.physics.util import (
    evaluate_electron_xs_energy_grid,
    scatter_direction,
)
from mcdc.transport.util import find_bin, linear_interpolation

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    particle = particle_container[0]
    E = particle["E"]
    mass = ELECTRON_MASS
    return LIGHT_SPEED * math.sqrt(E * (E + 2.0 * mass)) / (E + mass)


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, simulation, data):
    particle = particle_container[0]
    material = simulation["native_materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0
    for i in range(material["N_element"]):
        element_ID = int(mcdc_get.native_material.element_IDs(i, material, data))
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.native_material.element_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, element, data)
        total += element_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, element, data):
    idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)
    if reaction_type == ELECTRON_REACTION_TOTAL:
        xs0 = mcdc_get.element.electron_total_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_total_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_IONIZATION:
        xs0 = mcdc_get.element.electron_ionization_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_ionization_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_ELASTIC_SCATTERING:
        xs0 = mcdc_get.element.electron_elastic_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_elastic_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_EXCITATION:
        xs0 = mcdc_get.element.electron_excitation_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_excitation_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_BREMSSTRAHLUNG:
        xs0 = mcdc_get.element.electron_bremsstrahlung_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_bremsstrahlung_xs(idx + 1, element, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction_base, element, data):
    idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)

    # Apply offset
    offset = reaction_base["xs_offset_"]
    if idx < offset:
        return 0.0
    else:
        idx -= offset

    xs0 = mcdc_get.electron_reaction.xs(idx, reaction_base, data)
    xs1 = mcdc_get.electron_reaction.xs(idx + 1, reaction_base, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    collision_data = collision_data_container[0]
    material = simulation["native_materials"][particle["material_ID"]]

    # Particle properties
    E = particle["E"]

    # Check for cutoff energy
    if E <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # ==================================================================================
    # Sample colliding element
    # ==================================================================================

    SigmaT = macro_xs(ELECTRON_REACTION_TOTAL, particle_container, simulation, data)

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_element"]):
        element_ID = int(mcdc_get.native_material.element_IDs(i, material, data))
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.native_material.element_densities(i, material, data)
        sigmaT = total_micro_xs(ELECTRON_REACTION_TOTAL, E, element, data)

        total += element_density * sigmaT

        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    sigma_ionization = total_micro_xs(ELECTRON_REACTION_IONIZATION, E, element, data)
    sigma_elastic = total_micro_xs(
        ELECTRON_REACTION_ELASTIC_SCATTERING, E, element, data
    )
    sigma_bremsstrahlung = total_micro_xs(
        ELECTRON_REACTION_BREMSSTRAHLUNG, E, element, data
    )
    sigma_excitation = total_micro_xs(ELECTRON_REACTION_EXCITATION, E, element, data)

    xi = rng.lcg(particle_container) * sigmaT
    total = 0.0

    # Ionization
    total += sigma_ionization
    if xi < total:
        total -= sigma_ionization
        for i in range(element["N_electron_ionization_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_ionization_reaction_IDs(i, element, data)
            )
            reaction = simulation["electron_ionization_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = simulation["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                ionization(
                    reaction,
                    particle_container,
                    collision_data_container,
                    element,
                    program,
                    data,
                )
                return

    # Elastic scattering
    total += sigma_elastic
    if xi < total:
        total -= sigma_elastic
        for i in range(element["N_electron_elastic_scattering_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_elastic_scattering_reaction_IDs(
                    i, element, data
                )
            )
            reaction = simulation["electron_elastic_scattering_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = simulation["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                elastic_scattering(
                    reaction, particle_container, element, simulation, data
                )
                return

    # Bremsstrahlung
    total += sigma_bremsstrahlung
    if xi < total:
        total -= sigma_bremsstrahlung
        for i in range(element["N_electron_bremsstrahlung_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_bremsstrahlung_reaction_IDs(i, element, data)
            )
            reaction = simulation["electron_bremsstrahlung_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = simulation["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                bremsstrahlung(
                    reaction,
                    particle_container,
                    collision_data_container,
                    simulation,
                    data,
                )
                return

    # Excitation
    total += sigma_excitation
    if xi < total:
        total -= sigma_excitation
        for i in range(element["N_electron_excitation_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_excitation_reaction_IDs(i, element, data)
            )
            reaction = simulation["electron_excitation_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = simulation["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                excitation(
                    reaction,
                    particle_container,
                    collision_data_container,
                    simulation,
                    data,
                )
                return


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def elastic_scattering(reaction, particle_container, element, simulation, data):
    particle = particle_container[0]

    # Current energy
    E = particle["E"]
    Z = int(element["atomic_number"])
    multi_table = simulation["multi_table_distributions"][reaction["mu_ID"]]
    mu0 = sample_coupled_elastic_mu(E, Z, particle_container, multi_table, data)

    # Update direction
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    particle["ux"] = ux_new
    particle["uy"] = uy_new
    particle["uz"] = uz_new


@njit
def sample_coupled_elastic_mu(E, Z, rng_state, multi_table, data):
    idx0, idx1, frac = elastic_table_energy_interval(E, multi_table, data)
    fu0, mu_n0, eta0 = elastic_forward_branch_probability(idx0, Z, multi_table, data)

    if idx1 == idx0:
        fu = fu0
    else:
        fu1, mu_n1, eta1 = elastic_forward_branch_probability(
            idx1, Z, multi_table, data
        )
        fu = fu0 + frac * (fu1 - fu0)

    if fu < 0.0:
        fu = 0.0
    elif fu > 1.0:
        fu = 1.0

    if rng.lcg(rng_state) < fu:
        u = rng.lcg(rng_state)
        if idx1 == idx0:
            mu = sample_forward_peak_mu(eta0, mu_n0, u)
        else:
            mu0 = sample_forward_peak_mu(eta0, mu_n0, u)
            mu1 = sample_forward_peak_mu(eta1, mu_n1, u)
            mu = mu0 + frac * (mu1 - mu0)
    else:
        u = rng.lcg(rng_state)
        if idx1 == idx0:
            mu = sample_multi_table_fixed_index(idx0, u, multi_table, data)
        else:
            mu = sample_multi_table_log_cdf(E, idx0, idx1, u, multi_table, data)

    if mu < -1.0:
        return -1.0
    if mu > 1.0:
        return 1.0
    return mu


@njit
def elastic_table_energy_interval(E, multi_table, data):
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)
    if E <= grid[0]:
        return 0, 0, 0.0
    if E >= grid[-1]:
        idx = len(grid) - 1
        return idx, idx, 0.0

    idx0 = find_bin(E, grid)
    idx1 = idx0 + 1
    E0 = grid[idx0]
    E1 = grid[idx1]
    frac = (E - E0) / (E1 - E0)
    return idx0, idx1, frac


@njit
def elastic_table_bounds(index, multi_table, data):
    start = int(mcdc_get.multi_table_distribution.offset(index, multi_table, data))
    if index + 1 == multi_table["grid_length"]:
        end = multi_table["value_length"]
    else:
        end = int(mcdc_get.multi_table_distribution.offset(index + 1, multi_table, data))
    return start, end


@njit
def elastic_table_area(index, multi_table, data):
    start, end = elastic_table_bounds(index, multi_table, data)
    size = end - start
    if size < 2:
        return 0.0

    if multi_table["pdf_length"] == 0:
        c0 = mcdc_get.multi_table_distribution.cdf(start, multi_table, data)
        c1 = mcdc_get.multi_table_distribution.cdf(end - 1, multi_table, data)
        return c1 - c0

    total = 0.0
    for i in range(start, end - 1):
        mu0 = mcdc_get.multi_table_distribution.value(i, multi_table, data)
        mu1 = mcdc_get.multi_table_distribution.value(i + 1, multi_table, data)
        p0 = mcdc_get.multi_table_distribution.pdf(i, multi_table, data)
        p1 = mcdc_get.multi_table_distribution.pdf(i + 1, multi_table, data)
        total += 0.5 * (p0 + p1) * (mu1 - mu0)
    return total


@njit
def elastic_table_tail_density(index, multi_table, data):
    start, end = elastic_table_bounds(index, multi_table, data)
    size = end - start
    if size < 2:
        return 0.0

    mu_prev = mcdc_get.multi_table_distribution.value(end - 2, multi_table, data)
    mu_n = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)
    dmu = mu_n - mu_prev
    if dmu <= 0.0:
        return 0.0

    if multi_table["pdf_length"] == 0:
        c_prev = mcdc_get.multi_table_distribution.cdf(end - 2, multi_table, data)
        c_last = mcdc_get.multi_table_distribution.cdf(end - 1, multi_table, data)
        return (c_last - c_prev) / dmu

    return mcdc_get.multi_table_distribution.pdf(end - 1, multi_table, data)


@njit
def elastic_forward_branch_probability(index, Z, multi_table, data):
    energy = mcdc_get.multi_table_distribution.grid(index, multi_table, data)
    eta = compute_scattering_eta(energy, Z)

    start, end = elastic_table_bounds(index, multi_table, data)
    size = end - start
    if size < 2:
        mu_n = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)
        return 0.0, mu_n, eta

    mu_n = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)
    p_n = elastic_table_tail_density(index, multi_table, data)
    t_t = elastic_table_area(index, multi_table, data)

    x_n = eta + 1.0 - mu_n
    if p_n <= 0.0 or x_n <= 0.0 or eta <= 0.0:
        return 0.0, mu_n, eta

    A = p_n * x_n * x_n
    t_fu = A * ((1.0 / eta) - (1.0 / x_n))
    denom = t_fu + t_t
    if denom <= 0.0:
        return 0.0, mu_n, eta

    return t_fu / denom, mu_n, eta


@njit
def sample_forward_peak_mu(eta, mu_n, u):
    x_n = eta + 1.0 - mu_n
    if x_n <= 0.0 or eta <= 0.0:
        return mu_n

    inv_n = 1.0 / x_n
    inv = inv_n + u * ((1.0 / eta) - inv_n)
    if inv <= 0.0:
        return mu_n

    return 1.0 + eta - (1.0 / inv)


@njit
def sample_multi_table_fixed_index(index, xi, multi_table, data):
    start, end = elastic_table_bounds(index, multi_table, data)
    size = end - start
    if size <= 1:
        return mcdc_get.multi_table_distribution.value(start, multi_table, data)

    cdf_offset = multi_table["cdf_offset"]
    cdf = data[start + cdf_offset : start + cdf_offset + size]
    idx = find_bin(xi, cdf)
    c0 = cdf[idx]
    c1 = cdf[idx + 1]
    if c1 <= c0:
        return mcdc_get.multi_table_distribution.value(start + idx, multi_table, data)

    val0 = mcdc_get.multi_table_distribution.value(start + idx, multi_table, data)
    val1 = mcdc_get.multi_table_distribution.value(start + idx + 1, multi_table, data)
    frac = (xi - c0) / (c1 - c0)
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return val0 + frac * (val1 - val0)


@njit
def evaluate_multi_table_cdf_at_mu(index, mu, multi_table, data):
    start, end = elastic_table_bounds(index, multi_table, data)
    size = end - start
    if size <= 0:
        return 0.0

    value_offset = multi_table["value_offset"]
    values = data[start + value_offset : start + value_offset + size]
    if mu <= values[0]:
        return 0.0
    if mu >= values[-1]:
        return 1.0

    cdf_offset = multi_table["cdf_offset"]
    cdf = data[start + cdf_offset : start + cdf_offset + size]
    idx = find_bin(mu, values)
    mu0 = values[idx]
    mu1 = values[idx + 1]
    c0 = cdf[idx]
    c1 = cdf[idx + 1]

    if mu1 <= mu0:
        return c0
    return c0 + (mu - mu0) * (c1 - c0) / (mu1 - mu0)


@njit
def elastic_log_energy_fraction(E, E0, E1):
    if E <= 0.0 or E0 <= 0.0 or E1 <= 0.0:
        return (E - E0) / (E1 - E0)
    return (math.log(E) - math.log(E0)) / (math.log(E1) - math.log(E0))


@njit
def log_interpolate_cdf_value(E, E0, E1, c0, c1):
    eps = 1.0e-300
    if c0 <= 0.0 and c1 <= 0.0:
        return 0.0
    if c0 >= 1.0 and c1 >= 1.0:
        return 1.0

    y0 = max(min(c0, 1.0), eps)
    y1 = max(min(c1, 1.0), eps)
    frac = elastic_log_energy_fraction(E, E0, E1)
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return math.exp(math.log(y0) + frac * (math.log(y1) - math.log(y0)))


@njit
def sample_multi_table_log_cdf(E, idx0, idx1, xi, multi_table, data):
    if idx0 == idx1:
        return sample_multi_table_fixed_index(idx0, xi, multi_table, data)

    E0 = mcdc_get.multi_table_distribution.grid(idx0, multi_table, data)
    E1 = mcdc_get.multi_table_distribution.grid(idx1, multi_table, data)

    start0, end0 = elastic_table_bounds(idx0, multi_table, data)
    start1, end1 = elastic_table_bounds(idx1, multi_table, data)
    lo0 = mcdc_get.multi_table_distribution.value(start0, multi_table, data)
    lo1 = mcdc_get.multi_table_distribution.value(start1, multi_table, data)
    hi0 = mcdc_get.multi_table_distribution.value(end0 - 1, multi_table, data)
    hi1 = mcdc_get.multi_table_distribution.value(end1 - 1, multi_table, data)

    lo = min(lo0, lo1)
    hi = max(hi0, hi1)
    if xi <= 0.0:
        return lo
    if xi >= 1.0:
        return hi

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        c0 = evaluate_multi_table_cdf_at_mu(idx0, mid, multi_table, data)
        c1 = evaluate_multi_table_cdf_at_mu(idx1, mid, multi_table, data)
        c = log_interpolate_cdf_value(E, E0, E1, c0, c1)
        if c < xi:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


@njit
def compute_scattering_eta(E, Z):
    pc = math.sqrt(E * (E + 2.0 * ELECTRON_MASS))
    beta = pc / (E + ELECTRON_MASS)
    tau = E / ELECTRON_MASS
    FINE_STRUCTURE_CONSTANT = 7.2973525693e-3

    r = (FINE_STRUCTURE_CONSTANT * ELECTRON_MASS) / (0.885 * pc)
    z_sq = float(Z) ** (2.0 / 3.0)
    bracket = 1.13 + 3.76 * ((FINE_STRUCTURE_CONSTANT * float(Z)) / beta) ** 2
    rel = math.sqrt(tau / (tau + 1.0))

    return 0.25 * (r * r) * z_sq * bracket * rel


@njit
def sample_small_angle_mu_coulomb(E, Z, rng_state, mu_cut):
    eta = compute_scattering_eta(E, Z)

    x_cut = 1.0 - mu_cut
    u = rng.lcg(rng_state)

    denom = (1.0 / eta) - (1.0 / (eta + x_cut))
    inv = (1.0 / eta) - u * denom
    x = (1.0 / inv) - eta

    return 1.0 - x


# ======================================================================================
# Excitation
# ======================================================================================


@njit
def excitation(
    reaction, particle_container, collision_data_container, simulation, data
):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Current energy
    E = particle["E"]

    dE = evaluate_eloss(E, reaction, simulation, data)

    # Calculate outgoing energy
    E_out = E - dE

    # Check for cutoff
    if E_out <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["E"] = 0.0
        particle["alive"] = False
        return

    # If above cutoff, just deposit dE
    particle["E"] = E_out
    collision_data["energy_deposition"] += dE * particle["w"]


@njit
def evaluate_eloss(E, reaction, simulation, data):
    data_base = simulation["data"][int(reaction["eloss_ID"])]
    return evaluate_data(E, data_base, simulation, data)


# ======================================================================================
# Bremsstrahlung
# ======================================================================================


@njit
def bremsstrahlung(
    reaction, particle_container, collision_data_container, simulation, data
):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Current energy
    E = particle["E"]

    dE = evaluate_eloss(E, reaction, simulation, data)
    E_out = E - dE

    # Check for cutoff
    if E_out <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E_out * particle["w"]
        particle["E"] = 0.0
        particle["alive"] = False
        return

    # If above cutoff, allow photon to escape and update electron energy
    particle["E"] = E_out


# ======================================================================================
# Ionization
# ======================================================================================


@njit
def ionization(
    reaction, particle_container, collision_data_container, element, program, data
):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Current energy
    E = particle["E"]

    # Sample subshell
    N = int(reaction["N_subshell"])
    xs_vals = np.empty(N, dtype=np.float64)

    total = 0.0
    for i in range(N):
        xs_sub_ID = int(
            mcdc_get.electron_ionization_reaction.subshell_x_IDs(i, reaction, data)
        )
        xs_sub_table = simulation["data"][xs_sub_ID]
        xs_sub_i = evaluate_data(E, xs_sub_table, simulation, data)
        xs_vals[i] = xs_sub_i
        total += xs_sub_i

    xi = rng.lcg(particle_container) * total
    total_acc = 0.0
    chosen = 0
    for i in range(N):
        total_acc += xs_vals[i]
        if total_acc >= xi:
            chosen = i
            break

    # Binding energy
    B = mcdc_get.element.electron_ionization_subshell_binding_energy(
        chosen, element, data
    )
    if E <= B:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # Sample secondary energy
    dist_ID = int(
        mcdc_get.electron_ionization_reaction.subshell_product_IDs(
            chosen, reaction, data
        )
    )
    dist_base = simulation["distributions"][dist_ID]
    T_delta = sample_distribution(E, dist_base, particle_container, simulation, data)

    # Primary outgoing energy
    E_out = E - B - T_delta
    particle["E"] = E_out

    collision_data["energy_deposition"] += B * particle["w"]

    primary_alive_after = True
    if E_out <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E_out * particle["w"]
        particle["E"] = 0.0
        particle["alive"] = False
        primary_alive_after = False

    if T_delta <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += T_delta * particle["w"]
        return

    # Sample delta direction
    ux_delta, uy_delta, uz_delta = sample_delta_direction(
        T_delta, E, particle_container
    )

    # Momentum conservation if primary survives
    if primary_alive_after:
        p_before = math.sqrt(E * (E + 2.0 * ELECTRON_MASS))
        p_delta = math.sqrt(T_delta * (T_delta + 2.0 * ELECTRON_MASS))

        ux_before = particle["ux"]
        uy_before = particle["uy"]
        uz_before = particle["uz"]

        # Momentum vectors after collision
        px_after = p_before * ux_before - p_delta * ux_delta
        py_after = p_before * uy_before - p_delta * uy_delta
        pz_after = p_before * uz_before - p_delta * uz_delta

        # Normalize and set primary's new direction
        norm_sq = px_after * px_after + py_after * py_after + pz_after * pz_after
        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            particle["ux"] = px_after / norm
            particle["uy"] = py_after / norm
            particle["uz"] = pz_after / norm

    # Add secondary particle to bank
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]
    particle_module.copy_as_child(particle_container_new, particle_container)

    particle_new["E"] = T_delta
    particle_new["ux"] = ux_delta
    particle_new["uy"] = uy_delta
    particle_new["uz"] = uz_delta
    particle_new["w"] = particle["w"]

    particle_bank_module.bank_active_particle(particle_container_new, program)


@njit
def compute_mu_delta(T_delta, T_prim):
    pd = math.sqrt(T_delta * (T_delta + 2.0 * ELECTRON_MASS))
    pp = math.sqrt(T_prim * (T_prim + 2.0 * ELECTRON_MASS))
    mu = (T_delta * (T_prim + 2.0 * ELECTRON_MASS)) / (pd * pp)

    # Check in case of numerical issues
    if mu < -1.0:
        mu = -1.0
    if mu > 1.0:
        mu = 1.0

    return mu


@njit
def sample_delta_direction(T_delta, T_prim, particle_container):
    particle = particle_container[0]
    mu = compute_mu_delta(T_delta, T_prim)
    azi = 2.0 * PI * rng.lcg(particle_container)
    return scatter_direction(particle["ux"], particle["uy"], particle["uz"], mu, azi)
