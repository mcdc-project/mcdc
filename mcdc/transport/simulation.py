from numba import njit, objmode, uint64

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.output as output_module
import mcdc.transport.geometry as geometry
import mcdc.transport.geometry.surface as surface_module
import mcdc.transport.mpi as mpi
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.physics as physics
import mcdc.transport.rng as rng
import mcdc.transport.tally as tally_module
import mcdc.transport.technique as technique
import mcdc.transport.util as util

from mcdc.constant import *
from mcdc.print_ import (
    print_header_batch,
    print_progress,
    print_progress_eigenvalue,
)
from mcdc.transport.source import source_particle

# ======================================================================================
# Main simulations
# ======================================================================================


def fixed_source_simulation(simulation_container, data):
    # Ensure `mcdc` exist for the lifetime of the program by intentionally leaking their memory
    # adapt.leak(simulation_container)
    simulation = simulation_container[0]

    # Get some settings
    settings = simulation["settings"]
    N_batch = settings["N_batch"]
    N_particle = settings["N_particle"]
    N_census = settings["N_census"]
    use_census_based_tally = settings["use_census_based_tally"]

    # Loop over batches
    for idx_batch in range(N_batch):
        simulation["idx_batch"] = idx_batch
        seed_batch = rng.split_seed(uint64(idx_batch), settings["rng_seed"])

        # Distribute work
        mpi.distribute_work(N_particle, simulation)

        # Print multi-batch header
        if N_batch > 1:
            with objmode():
                print_header_batch(idx_batch, N_batch)

        # Loop over time censuses
        for idx_census in range(N_census):
            simulation["idx_census"] = idx_census
            seed_census = rng.split_seed(uint64(seed_batch), rng.SEED_SPLIT_CENSUS)

            # Reset tally time filters if census-based tally is used
            if use_census_based_tally:
                tally_module.filter.set_census_based_time_grid(simulation, data)

            # Accordingly promote future particles to censused particles
            if particle_bank_module.get_bank_size(simulation["bank_future"]) > 0:
                particle_bank_module.promote_future_particles(simulation, data)

            # Loop over source particles
            seed_source = rng.split_seed(uint64(seed_census), rng.SEED_SPLIT_SOURCE)
            source_loop(uint64(seed_source), simulation, data)

            # Manage particle banks: population control and work rebalance
            particle_bank_module.manage_particle_banks(simulation)

            # Time census-based tally closeout
            if use_census_based_tally:
                tally_module.closeout.reduce(simulation, data)
                if simulation["mpi_master"]:
                    tally_module.closeout.accumulate_statistics_and_reset_scores(
                        simulation, data
                    )
                    with objmode():
                        output_module.generate_census_based_tally(simulation, data)
                    tally_module.closeout.reset_statistics(simulation, data)
                else:
                    tally_module.closeout.reset_scores(simulation, data)

            # Terminate census loop if all banks are empty
            if (
                idx_census > 0
                and particle_bank_module.total_size(simulation["bank_source"]) == 0
                and particle_bank_module.total_size(simulation["bank_census"]) == 0
                and particle_bank_module.total_size(simulation["bank_future"]) == 0
            ):
                break

        # Multi-batch closeout
        if N_batch > 1:
            # Reset banks
            particle_bank_module.set_bank_size(simulation["bank_active"], 0)
            particle_bank_module.set_bank_size(simulation["bank_census"], 0)
            particle_bank_module.set_bank_size(simulation["bank_source"], 0)
            particle_bank_module.set_bank_size(simulation["bank_future"], 0)

            if not use_census_based_tally:
                # Tally history closeout
                tally_module.closeout.reduce(simulation, data)
                if simulation["mpi_master"]:
                    tally_module.closeout.accumulate_statistics_and_reset_scores(
                        simulation, data
                    )
                else:
                    tally_module.closeout.reset_scores(simulation, data)

    # Tally closeout
    if not use_census_based_tally:
        tally_module.closeout.finalize(simulation, data)
    else:
        tally_module.closeout.finalize_census(simulation, data)


def eigenvalue_simulation(simulation_container, data):
    # Ensure `mcdc` exist for the lifetime of the program
    # by intentionally leaking their memory
    # adapt.leak(simulation_container)
    simulation = simulation_container[0]

    # Get some settings
    settings = simulation["settings"]
    N_inactive = settings["N_inactive"]
    N_cycle = settings["N_cycle"]
    N_particle = settings["N_particle"]

    # Distribute work
    mpi.distribute_work(N_particle, simulation)

    # Loop over power iteration cycles
    for idx_cycle in range(N_cycle):
        simulation["idx_cycle"] = idx_cycle
        seed_cycle = rng.split_seed(uint64(idx_cycle), settings["rng_seed"])

        # Loop over source particles
        source_loop(uint64(seed_cycle), simulation, data)

        # Tally "history" closeout
        tally_module.closeout.eigenvalue_cycle(simulation, data)
        if simulation["cycle_active"]:
            tally_module.closeout.reduce(simulation, data)
            if simulation["mpi_master"]:
                tally_module.closeout.accumulate_statistics_and_reset_scores(
                    simulation, data
                )
            else:
                tally_module.closeout.reset_scores(simulation, data)

        # Manage particle banks: population control and work rebalance
        particle_bank_module.manage_particle_banks(simulation)

        # Print progress
        with objmode():
            print_progress_eigenvalue(simulation, data)

        # Entering active cycle?
        simulation["idx_cycle"] += 1
        if simulation["idx_cycle"] >= N_inactive:
            simulation["cycle_active"] = True

    # Tally closeout
    tally_module.closeout.finalize(simulation, data)
    tally_module.closeout.eigenvalue_simulation(simulation)


# =============================================================================
# Source loop
# =============================================================================


@njit
def source_loop(seed, simulation, data):
    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    work_start = simulation["mpi_work_start"]
    work_size = simulation["mpi_work_size"]

    for idx_work in range(work_size):
        simulation["idx_work"] = work_start + idx_work
        generate_source_particle(work_start, idx_work, seed, simulation, data)

        # Run the source particle and its secondaries
        exhaust_active_bank(simulation, data)

        N_prog = source_closeout(simulation, idx_work, N_prog, data)


@njit
def generate_source_particle(work_start, idx_work, seed, program, data):
    """Get a source particle and put into one of the banks"""
    simulation = util.access_simulation(program)
    settings = simulation["settings"]

    # Get from fixed-source?
    if particle_bank_module.get_bank_size(simulation["bank_source"]) == 0:
        particle_container = util.local_array(1, type_.particle_data)
        particle = particle_container[0]

        # Sample source
        seed_work = rng.split_seed(work_start + idx_work, seed)
        source_particle(particle_container, seed_work, simulation, data)

    # Get from source bank
    else:
        particle_container = simulation["bank_source"]["particle_data"][
            idx_work : (idx_work + 1)
        ]
        particle = particle_container[0]

    # Skip if beyond time boundary
    if particle["t"] > settings["time_boundary"]:
        return

    # Check if it is beyond current or next census times
    hit_census = False
    hit_next_census = False
    idx_census = simulation["idx_census"]

    if idx_census < settings["N_census"] - 1:
        if particle["t"] > mcdc_get.settings.census_time(
            idx_census + 1, settings, data
        ):
            hit_census = True
            hit_next_census = True
        elif particle["t"] > mcdc_get.settings.census_time(idx_census, settings, data):
            hit_census = True

    # Put into the right bank
    if not hit_census:
        particle_bank_module.bank_active_particle(particle_container, program)
    elif not hit_next_census:
        # Particle will participate after the current census
        particle_bank_module.bank_census_particle(particle_container, program)
    else:
        # Particle will participate in the future
        particle_bank_module.bank_future_particle(particle_container, program)


@njit
def exhaust_active_bank(simulation, data):
    particle_container = util.local_array(1, type_.particle)
    particle = particle_container[0]

    # Loop until active bank is exhausted
    while particle_bank_module.get_bank_size(simulation["bank_active"]) > 0:
        # Get particle from active bank
        particle_bank_module.pop_particle(particle_container, simulation["bank_active"])

        # Particle loop
        particle_loop(particle_container, simulation, data)


@njit
def source_closeout(simulation, idx_work, N_prog, data):
    # Tally closeout for history-based statistics
    if simulation["history_based_statistics"]:
        tally_module.closeout.accumulate_statistics_and_reset_scores(simulation, data)

    # Progress printout
    percent = (idx_work + 1.0) / simulation["mpi_work_size"]
    if simulation["settings"]["use_progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, simulation)
    return N_prog


# ======================================================================================
# Particle loop
# ======================================================================================


@njit
def particle_loop(particle_container, simulation, data):
    particle = particle_container[0]

    # Apply techniques
    apply_techniques(particle_container, simulation, data)
    if not particle["alive"]:
        return

    while particle["alive"]:
        step_particle(particle_container, simulation, data)


@njit
def step_particle(particle_container, program, data):
    simulation = util.access_simulation(program)
    settings = simulation["settings"]
    particle = particle_container[0]

    # ==================================================================================
    # Preparation
    # ==================================================================================

    # Reset particle event flags
    particle["event"] = EVENT_NONE

    # Inspect geometry
    d_geometry = inspect_geometry(particle_container, simulation, data)
    if particle["event"] == EVENT_LOST:
        return

    # Determine distance to next events
    distance = determine_next_events(particle_container, d_geometry, simulation, data)
    if particle["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Advance
    # ==================================================================================

    # Move particle
    move_particle(particle_container, distance, simulation, data)

    # Condensed interactions
    if use_condensed_interactions(particle_container, settings):
        condensed_interactions(particle_container, distance, simulation, data)
        if not particle["alive"]:
            return

    # ==================================================================================
    # Main event
    # ==================================================================================

    # Time boundary crossing
    if particle["event"] == EVENT_TIME_BOUNDARY:
        particle["alive"] = False
        return

    # Collision
    if particle["event"] & EVENT_COLLISION:
        collision(particle_container, program, data)
        if not particle["alive"]:
            return

    # Geometry crossing
    elif particle["event"] & EVENT_GEOMETRY_CROSSING:
        geometry_crossing(particle_container, simulation, data)
        if not particle["alive"]:
            return

    # ==================================================================================
    # Post treatments
    # ==================================================================================

    # Apply techniques
    apply_techniques(particle_container, program, data)
    if not particle["alive"]:
        return

    # Census time crossing
    if particle["event"] & EVENT_TIME_CENSUS:
        particle_bank_module.bank_census_particle(particle_container, program)
        particle["alive"] = False


@njit
def inspect_geometry(particle_container, simulation, data):
    """
    Locate the particle and record its next geometry crossing.

    - Set particle top cell and material IDs (if not lost)
    - Set surface ID for surface actions, or -1 for a lattice-only crossing
    - Set EVENT_LOST if lost; otherwise retain EVENT_NONE until event determination
    - Return distance to the geometry crossing, or zero if lost
    """
    particle = particle_container[0]

    # Locate the material before evaluating material-dependent transport data.
    if particle["material_ID"] == -1:
        if not geometry.locate_particle(particle_container, simulation, data):
            particle["event"] = EVENT_LOST
            return 0.0

    distance = geometry.inspect_geometry(particle_container, simulation, data)
    if particle["event"] == EVENT_LOST:
        return 0.0

    return distance


@njit
def determine_next_events(particle_container, d_geometry, simulation, data):
    """Determine next events based on their respective distances."""
    settings = simulation["settings"]
    particle = particle_container[0]
    speed = physics.particle_speed(particle_container, simulation, data)

    # ==================================================================================
    # Compute the distances
    # ==================================================================================

    # Distance to time boundary
    d_time_boundary = speed * (settings["time_boundary"] - particle["t"])

    # Distance to census time
    idx = simulation["idx_census"]
    d_time_census = speed * (
        mcdc_get.settings.census_time(idx, settings, data) - particle["t"]
    )

    # Distance to next collision
    d_collision = physics.collision_distance(particle_container, simulation, data)

    # Maximum distance for condensed interactions
    d_condensed = INF
    if use_condensed_interactions(particle_container, settings):
        d_condensed = physics.max_condensed_step_distance(
            particle_container, simulation, data
        )

    # ==================================================================================
    # Select event distance and resolve coincident events
    # ==================================================================================
    #   Time boundary is exclusive and overrides all coincident events.
    #   Otherwise, select at most one physical event, possibly with census.
    #   Geometry crossing takes precedence over collision.
    #   A condensed-step limit alone results in EVENT_NONE.

    # Get minimum distance
    distance = min(d_geometry, d_collision, d_time_census, d_time_boundary, d_condensed)

    # To infinity and beyond, no event, particle is lost
    if distance >= INF:
        particle["event"] = EVENT_LOST
        particle["surface_ID"] = -1
        particle["alive"] = False
        return 0.0

    # Final time overrides all coincident events.
    if geometry.check_coincidence(d_time_boundary, distance):
        particle["event"] = EVENT_TIME_BOUNDARY
        particle["surface_ID"] = -1
        return d_time_boundary

    # Geometry crossing
    if geometry.check_coincidence(d_geometry, distance):
        particle["event"] = EVENT_GEOMETRY_CROSSING
    # Collision
    elif geometry.check_coincidence(d_collision, distance):
        particle["event"] = EVENT_COLLISION
    else:
        particle["event"] = EVENT_NONE

    # Time census
    if geometry.check_coincidence(d_time_census, distance):
        particle["event"] |= EVENT_TIME_CENSUS

    # Reset surface_ID if not geometry crossing
    if not particle["event"] & EVENT_GEOMETRY_CROSSING:
        particle["surface_ID"] = -1

    return distance


@njit
def use_condensed_interactions(particle_container, settings):
    """Return whether condensed interactions is enabled for the particle."""
    particle = particle_container[0]
    condensed_interactions = settings["condensed_interactions"]
    return (
        particle["particle_type"] == PARTICLE_PROTON
        and condensed_interactions["proton"]
    )


@njit
def move_particle(particle_container, distance, simulation, data):
    """Score tracklength tallies and move particle by the selected distance."""
    score_tracklength_tallies(particle_container, distance, simulation, data)
    particle_module.move(particle_container, distance, simulation, data)


@njit
def condensed_interactions(particle_container, distance, simulation, data):
    """Condense interactions over the traveled distance and score their tallies."""
    particle = particle_container[0]

    incident_container = util.local_array(1, type_.particle)
    incident_container[0] = particle

    interaction_data_container = util.local_array(1, type_.interaction_data)
    interaction_data_container[0]["energy_deposition"] = 0.0

    # Preserve the state before condensed interactions for tally filters.
    particle_module.copy(
        interaction_data_container["incident_particle"], particle_container
    )

    physics.condensed_interactions(
        particle_container, interaction_data_container, distance, simulation, data
    )

    # Score the condensed contribution at the step endpoint.
    score_interaction_tallies(
        particle_container, interaction_data_container, simulation, data
    )

    # Deflection can turn the particle back into the incident region.
    if particle["alive"] and particle["surface_ID"] >= 0:
        if not geometry.surface_crossing_valid(
            incident_container, particle_container, distance, simulation, data
        ):
            # Keep the geometry event so coincident lattice crossings and the
            # outgoing cell are resolved even when this surface is not crossed.
            particle["surface_ID"] = -1


@njit
def collision(particle_container, program, data):
    """Perform a discrete collision and score its interaction contribution."""
    simulation = util.access_simulation(program)

    interaction_data_container = util.local_array(1, type_.interaction_data)
    interaction_data_container[0]["energy_deposition"] = 0.0

    # Preserve the incident state before the discrete collision.
    particle_module.copy(
        interaction_data_container["incident_particle"], particle_container
    )

    physics.collision(particle_container, interaction_data_container, program, data)

    score_interaction_tallies(
        particle_container, interaction_data_container, simulation, data
    )


@njit
def geometry_crossing(particle_container, simulation, data):
    particle = particle_container[0]

    # Surface crossing
    if particle["surface_ID"] >= 0:
        surface_crossing(particle_container, simulation, data)

    # Reset geometry IDs
    if particle["alive"]:
        particle["cell_ID"] = -1
        particle["material_ID"] = -1


@njit
def surface_crossing(particle_container, simulation, data):
    """Apply boundary condition and score surface crossing tallies"""
    particle = particle_container[0]
    crossed_surface_ID = particle["surface_ID"]

    surface = simulation["surfaces"][crossed_surface_ID]
    BC = surface["boundary_condition"]

    # Apply BC
    if BC == BC_VACUUM:
        particle["alive"] = False
    elif BC == BC_REFLECTIVE:
        surface_module.reflect(particle_container, surface)
        return  # No score

    # Score tally
    for i in range(surface["N_surface_crossing_tally"]):
        tally_ID = mcdc_get.surface.surface_crossing_tally_IDs(i, surface, data)
        tally = simulation["tallies"][tally_ID]
        tally_module.score.surface_crossing(
            particle_container, surface, tally, simulation, data
        )


@njit
def apply_techniques(particle_container, program, data):
    """Apply variance reduction techniques."""
    simulation = util.access_simulation(program)
    particle = particle_container[0]

    # Weight windows
    if technique.get_weight_window_object(particle_container, program)["active"]:
        technique.weight_windows(particle_container, program, data)
        if not particle["alive"]:
            return

    # Global weight roulette
    if simulation["technique"]["global_weight_roulette"]["active"]:
        technique.global_weight_roulette(particle_container, simulation)


@njit
def score_tracklength_tallies(particle_container, distance, simulation, data):
    """Score tracklength tallies using the distance and incident particle state."""
    particle = particle_container[0]

    if simulation["cycle_active"]:
        cell = simulation["cells"][particle["cell_ID"]]
        for i in range(cell["N_tracklength_tally"]):
            tally_ID = mcdc_get.cell.tracklength_tally_IDs(i, cell, data)
            tally = simulation["tallies"][tally_ID]
            tally_module.score.tracklength(
                particle_container, distance, tally, simulation, data
            )

    if simulation["settings"]["neutron_eigenvalue_mode"]:
        tally_module.score.eigenvalue_tally(
            particle_container, distance, simulation, data
        )


@njit
def score_interaction_tallies(
    particle_container, interaction_data_container, simulation, data
):
    """Score interaction contributions in the incident cell."""
    particle = particle_container[0]

    if simulation["cycle_active"]:
        cell = simulation["cells"][particle["cell_ID"]]
        for i in range(cell["N_interaction_tally"]):
            tally_ID = mcdc_get.cell.interaction_tally_IDs(i, cell, data)
            tally = simulation["tallies"][tally_ID]
            tally_module.score.interaction(
                interaction_data_container, tally, simulation, data
            )
