import math

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
    INTERPOLATION_LINEAR,
    LIGHT_SPEED,
    PI,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import invert_tabulated_segment, sample_distribution
from mcdc.transport.physics.util import (
    evaluate_electron_xs_energy_grid,
    scatter_direction,
)
from mcdc.transport.util import (
    find_bin,
    linear_interpolation,
    log_interpolation,
    semilogx_interpolation,
)

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
    material = simulation["materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0
    for i in range(material["N_element"]):
        element_ID = mcdc_get.material.element_IDs(i, material, data)
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.material.element_densities(i, material, data)
        xs = total_micro_xs(reaction_type, E, element, data)
        total += element_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, element, data):
    idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)
    return _total_micro_xs(reaction_type, E, idx, E0, E1, element, data)


@njit
def _total_micro_xs(reaction_type, E, idx, E0, E1, element, data):
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
def reaction_micro_xs(E, reaction, element, data):
    idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)
    return _reaction_micro_xs(E, idx, E0, E1, reaction, data)


@njit
def _reaction_micro_xs(E, idx, E0, E1, reaction, data):
    # Apply offset
    offset = reaction["xs_offset_"]
    if idx < offset:
        return 0.0
    else:
        idx -= offset

    xs0 = mcdc_get.electron_reaction.xs(idx, reaction, data)
    xs1 = mcdc_get.electron_reaction.xs(idx + 1, reaction, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    collision_data = collision_data_container[0]
    material = simulation["materials"][particle["material_ID"]]

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
        element_ID = mcdc_get.material.element_IDs(i, material, data)
        element = simulation["elements"][element_ID]

        element_density = mcdc_get.material.element_densities(i, material, data)
        idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)
        sigmaT = _total_micro_xs(ELECTRON_REACTION_TOTAL, E, idx, E0, E1, element, data)

        total += element_density * sigmaT

        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    sigma_ionization = _total_micro_xs(
        ELECTRON_REACTION_IONIZATION, E, idx, E0, E1, element, data
    )
    sigma_elastic = _total_micro_xs(
        ELECTRON_REACTION_ELASTIC_SCATTERING, E, idx, E0, E1, element, data
    )
    sigma_bremsstrahlung = _total_micro_xs(
        ELECTRON_REACTION_BREMSSTRAHLUNG, E, idx, E0, E1, element, data
    )
    sigma_excitation = _total_micro_xs(
        ELECTRON_REACTION_EXCITATION, E, idx, E0, E1, element, data
    )

    xi = rng.lcg(particle_container) * sigmaT
    total = 0.0

    # Ionization
    total += sigma_ionization
    if xi < total:
        total -= sigma_ionization
        for i in range(element["N_electron_ionization_reaction"]):
            reaction_ID = mcdc_get.element.electron_ionization_reaction_IDs(
                i, element, data
            )
            reaction = simulation["electron_reactions"][reaction_ID]
            total += _reaction_micro_xs(E, idx, E0, E1, reaction, data)

            if xi < total:
                sample_ionization(
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
            reaction_ID = mcdc_get.element.electron_elastic_scattering_reaction_IDs(
                i, element, data
            )
            reaction = simulation["electron_reactions"][reaction_ID]
            reaction_xs = _reaction_micro_xs(E, idx, E0, E1, reaction, data)
            total += reaction_xs

            if xi < total:
                sample_elastic_scattering(
                    reaction,
                    reaction_xs,
                    particle_container,
                    element,
                    simulation,
                    data,
                )
                return

    # Bremsstrahlung
    total += sigma_bremsstrahlung
    if xi < total:
        total -= sigma_bremsstrahlung
        for i in range(element["N_electron_bremsstrahlung_reaction"]):
            reaction_ID = mcdc_get.element.electron_bremsstrahlung_reaction_IDs(
                i, element, data
            )
            reaction = simulation["electron_reactions"][reaction_ID]
            total += _reaction_micro_xs(E, idx, E0, E1, reaction, data)

            if xi < total:
                sample_bremsstrahlung(
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
            reaction_ID = mcdc_get.element.electron_excitation_reaction_IDs(
                i, element, data
            )
            reaction = simulation["electron_reactions"][reaction_ID]
            total += _reaction_micro_xs(E, idx, E0, E1, reaction, data)

            if xi < total:
                sample_excitation(
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
def sample_elastic_scattering(
    reaction, xs_total, particle_container, element, simulation, data
):
    particle = particle_container[0]

    sub_ID = reaction["sub_ID"]
    elastic_scattering = simulation["electron_elastic_scattering_reactions"][sub_ID]

    # Current energy
    E = particle["E"]

    Z = int(element["atomic_number"])
    mu_cut = float(elastic_scattering["mu_cut"])

    # Screened Rutherford CDF sets the large- vs small-angle split
    eta = compute_scattering_eta(E, Z)
    prob_large = sr_cdf(mu_cut, eta)

    xi = rng.lcg(particle_container)

    if xi < prob_large:
        # ---------------------------------------------------------------------
        # Large-angle elastic scattering
        # ---------------------------------------------------------------------

        mu_distribution = simulation["distributions"][elastic_scattering["mu_ID"]]
        mu0 = sample_distribution(
            E, mu_distribution, particle_container, simulation, data
        )

    else:
        # ---------------------------------------------------------------------
        # Small-angle elastic scattering (Coulomb tail sampling)
        # ---------------------------------------------------------------------

        Z = int(element["atomic_number"])
        mu0 = sample_small_angle_mu_coulomb(E, Z, particle_container, mu_cut)

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
def sr_cdf(mu, eta):
    """Screened Rutherford CDF: fraction of events with scattering cosine < mu."""
    num = (1.0 / eta) - 1.0 / (1.0 - mu + 2.0 * eta)
    denom = (1.0 / eta) - 1.0 / (2.0 + 2.0 * eta)
    return num / denom


@njit
def sample_small_angle_mu_coulomb(E, Z, rng_state, mu_cut):
    eta = compute_scattering_eta(E, Z)

    x_cut = 1.0 - mu_cut
    u = rng.lcg(rng_state)

    denom = (1.0 / eta) - (1.0 / (eta + x_cut))
    inv = (1.0 / eta) - u * denom
    x = (1.0 / inv) - eta

    return 1.0 - x


@njit
def elastic_large_xs(E, elastic_scattering, simulation, data):
    reaction_data = simulation["data"][int(elastic_scattering["xs_large_ID"])]
    return evaluate_data(E, reaction_data, simulation, data)


# ======================================================================================
# Excitation
# ======================================================================================


@njit
def sample_excitation(
    reaction, particle_container, collision_data_container, simulation, data
):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    sub_ID = reaction["sub_ID"]
    excitation = simulation["electron_excitation_reactions"][sub_ID]

    # Current energy
    E = particle["E"]

    dE = evaluate_eloss(E, excitation, simulation, data)

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
    reaction_data = simulation["data"][int(reaction["eloss_ID"])]
    return evaluate_data(E, reaction_data, simulation, data)


# ======================================================================================
# Bremsstrahlung
# ======================================================================================


@njit
def sample_bremsstrahlung(
    reaction, particle_container, collision_data_container, simulation, data
):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    sub_ID = reaction["sub_ID"]
    bremsstrahlung = simulation["electron_bremsstrahlung_reactions"][sub_ID]

    # Current energy
    E = particle["E"]

    dE = evaluate_eloss(E, bremsstrahlung, simulation, data)
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
def sample_ionization(
    reaction, particle_container, collision_data_container, element, program, data
):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    sub_ID = reaction["sub_ID"]
    ionization = simulation["electron_ionization_reactions"][sub_ID]

    # Current energy
    E = particle["E"]

    # Sample subshell
    N = int(ionization["N_subshell"])
    # ENDF subshells K through Q3 (MT 534--572); fixed size for GPU local storage.
    cumulative_xs = util.local_array(39, type_.float64)
    if N < 1 or N > len(cumulative_xs):
        raise ValueError("Unsupported number of electron ionization subshells")
    total = 0.0
    for i in range(N):
        xs_sub_ID = mcdc_get.electron_ionization_reaction.subshell_x_IDs(
            i, ionization, data
        )
        xs_sub_table = simulation["data"][xs_sub_ID]
        total += evaluate_data(E, xs_sub_table, simulation, data)
        cumulative_xs[i] = total

    if total <= 0.0:
        raise ValueError("Cannot sample ionization with zero subshell cross section")

    xi = rng.lcg(particle_container) * total
    chosen = 0
    for i in range(N):
        if xi < cumulative_xs[i]:
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
    dist_ID = mcdc_get.electron_ionization_reaction.subshell_product_IDs(
        chosen, ionization, data
    )
    T_dist = simulation["distributions"][dist_ID]
    T_delta = sample_delta_energy(E, B, T_dist, particle_container, simulation, data)

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

    # Construct the secondary particle
    particle_container_new = util.local_array(1, type_.particle_data)
    particle_new = particle_container_new[0]
    particle_module.copy_as_child(particle_container_new, particle_container)

    particle_new["E"] = T_delta
    particle_new["ux"] = ux_delta
    particle_new["uy"] = uy_delta
    particle_new["uz"] = uz_delta
    particle_new["w"] = particle["w"]

    # Continue the lower-energy electron when requested; bank the other one.
    if (
        simulation["settings"]["electron_transport"]["prioritize_low_energy"]
        and particle["alive"]
        and particle_new["E"] < particle["E"]
    ):
        particle_bank_module.bank_active_particle(particle_container, program)
        particle_module.copy(particle_container, particle_container_new)
    else:
        particle_bank_module.bank_active_particle(particle_container_new, program)


@njit
def sample_delta_energy(E, B, distribution, rng_state, simulation, data):
    """Sample the knock-on electron energy from the subshell spectrum tables.

    One random number inverts the two tables bounding E, and the two energies
    are interpolated log-log in incident energy, following FRENSIE's EPRDATA14
    policy. The artificial table at the binding energy is skipped; below the
    first table above B, the sampled energy is scaled to reach zero at E = B.
    """

    multi_table = simulation["multi_table_distributions"][distribution["sub_ID"]]
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)

    # Skip the EPRDATA14 table at E = B, which tabulates positive knock-on
    # energies although no energy is available. The library reader guarantees
    # a table above the binding energy.
    first = 0
    while grid[first] <= B:
        first += 1

    xi = rng.lcg(rng_state)

    # Below the first table above B: scale its sample to reach zero at E = B
    if E <= grid[first]:
        T_delta = _invert_delta_table(first, xi, multi_table, simulation, data)
        T_delta *= (E - B) / (grid[first] - B)

    # Above the grid: use the last table
    elif E >= grid[-1]:
        T_delta = _invert_delta_table(len(grid) - 1, xi, multi_table, simulation, data)

    else:
        idx = find_bin(E, grid)
        E0 = grid[idx]
        E1 = grid[idx + 1]
        T0 = _invert_delta_table(idx, xi, multi_table, simulation, data)
        T1 = _invert_delta_table(idx + 1, xi, multi_table, simulation, data)

        if T0 > 0.0 and T1 > 0.0:
            T_delta = log_interpolation(E, E0, E1, T0, T1)
        else:
            # Log-log interpolation is undefined at a zero sample
            T_delta = semilogx_interpolation(E, E0, E1, T0, T1)

    # Rounded tabulated energies can slightly exceed the knock-on limit
    return min(T_delta, 0.5 * (E - B))


@njit
def _invert_delta_table(idx, xi, multi_table, simulation, data):
    """Invert the spectrum table at grid index idx with a given random number."""

    ID = mcdc_get.multi_table_distribution.table_IDs(idx, multi_table, data)
    sub_ID = simulation["distributions"][ID]["sub_ID"]
    table = simulation["tabulated_distributions"][sub_ID]

    pdf_data = simulation["data"][table["pdf_ID"]]
    pdf_table = simulation["table_data"][pdf_data["sub_ID"]]

    cdf = mcdc_get.table_data.aux_vector(0, pdf_table, data)

    # find_bin returns -1 at the first CDF point; avoid reading before the table.
    if xi <= cdf[0]:
        return mcdc_get.table_data.x(0, pdf_table, data)
    idx = find_bin(xi, cdf)

    c0 = cdf[idx]
    c1 = cdf[idx + 1]
    v0 = mcdc_get.table_data.x(idx, pdf_table, data)
    v1 = mcdc_get.table_data.x(idx + 1, pdf_table, data)

    # A repeated CDF value has a zero-width segment; take its upper value
    if xi == c1:
        return v1

    # PDF input is piecewise linear; invert the PDF segment exactly
    interpolation = mcdc_get.table_data.interpolations(0, pdf_table, data)
    if interpolation == INTERPOLATION_LINEAR:
        p0 = mcdc_get.table_data.y(idx, pdf_table, data)
        p1 = mcdc_get.table_data.y(idx + 1, pdf_table, data)
        return invert_tabulated_segment(xi, c0, v0, v1, p0, p1, interpolation)

    # CDF input is stored as a histogram PDF (a piecewise-linear CDF), but the
    # EPRDATA14 CDF is log-log in (CDF, energy); the segment starting at
    # CDF = 0 is linear
    if c0 == 0.0 or v0 == 0.0:
        return linear_interpolation(xi, c0, c1, v0, v1)
    return log_interpolation(xi, c0, c1, v0, v1)


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
