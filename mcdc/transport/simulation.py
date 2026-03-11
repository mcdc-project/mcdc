import numpy as np

from numba import njit, objmode, uint64

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.output as output_module
import mcdc.transport.geometry as geometry
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
                tally_module.closeout.accumulate(simulation, data)
                if simulation["mpi_master"]:
                    with objmode():
                        output_module.generate_census_based_tally(simulation, data)
                tally_module.closeout.reset_sum_bins(simulation, data)

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
                tally_module.closeout.accumulate(simulation, data)

    # Tally closeout
    if not use_census_based_tally:
        tally_module.closeout.finalize(simulation, data)


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
            tally_module.closeout.accumulate(simulation, data)

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

        source_closeout(simulation, idx_work, N_prog, data)


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
        particle_bank_module.bank_active_particle(particle_container, simulation)
    elif not hit_next_census:
        # Particle will participate after the current census
        particle_bank_module.bank_census_particle(particle_container, simulation)
    else:
        # Particle will participate in the future
        particle_bank_module.bank_future_particle(particle_container, simulation)


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
    # Tally history closeout for one-batch fixed-source simulation
    if (
        not simulation["settings"]["eigenvalue_mode"]
        and simulation["settings"]["N_batch"] == 1
    ):
        if not simulation["settings"]["use_census_based_tally"]:
            tally_module.closeout.accumulate(simulation, data)

    # Progress printout
    percent = (idx_work + 1.0) / simulation["mpi_work_size"]
    if simulation["settings"]["use_progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, simulation)


# ======================================================================================
# Particle loop
# ======================================================================================


@njit
def particle_loop(particle_container, simulation, data):
    particle = particle_container[0]

    while particle["alive"]:
        step_particle(particle_container, simulation, data)


@njit
def step_particle(particle_container, simulation, data):
    particle = particle_container[0]

    # Determine and move to event
    move_to_event(particle_container, simulation, data)

    # Execute events
    if particle["event"] == EVENT_LOST:
        return

    # Collision
    if particle["event"] & EVENT_COLLISION:
        physics.collision(particle_container, simulation, data)

    # Surface and domain crossing
    if particle["event"] & EVENT_SURFACE_CROSSING:
        geometry.surface_crossing(particle_container, simulation, data)

    # Census time crossing
    if particle["event"] & EVENT_TIME_CENSUS:
        particle_bank_module.bank_census_particle(particle_container, simulation)
        particle["alive"] = False

    # Time boundary crossing
    if particle["event"] & EVENT_TIME_BOUNDARY:
        particle["alive"] = False

    # Weight roulette
    if particle["alive"]:
        technique.weight_roulette(particle_container, simulation)


@njit
def move_to_event(particle_container, simulation, data):
    settings = simulation["settings"]

    # ==================================================================================
    # Preparation (as needed)
    # ==================================================================================
    particle = particle_container[0]

    # Multigroup preparation
    #   In MG mode, particle speed is material-dependent.
    if settings["multigroup_mode"]:
        # If material is not identified yet, locate the particle
        if particle["material_ID"] == -1:
            if not geometry.locate_particle(particle_container, simulation, data):
                # Particle is lost
                particle["event"] = EVENT_LOST
                return

    # ==================================================================================
    # Geometry inspection
    # ==================================================================================
    #   - Set particle top cell and material IDs (if not lost)
    #   - Set surface ID (if surface hit)
    #   - Set particle boundary event (surface or lattice crossing, or lost)
    #   - Return distance to boundary (surface or lattice)

    d_boundary = geometry.inspect_geometry(particle_container, simulation, data)

    # Particle is lost?
    if particle["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Get distances to other events
    # ==================================================================================

    # Distance to domain
    speed = physics.particle_speed(particle_container, simulation, data)

    # Distance to time boundary
    d_time_boundary = speed * (settings["time_boundary"] - particle["t"])

    # Distance to census time
    idx = simulation["idx_census"]
    d_time_census = speed * (
        mcdc_get.settings.census_time(idx, settings, data) - particle["t"]
    )

    # Distance to next collision
    d_collision = physics.collision_distance(particle_container, simulation, data)

    # ==================================================================================
    # Determine event(s)
    # ==================================================================================
    # TODO: Make a function to better maintain the repeating operation

    distance = d_boundary

    # Check distance to collision
    if d_collision < distance - COINCIDENCE_TOLERANCE:
        distance = d_collision
        particle["event"] = EVENT_COLLISION
        particle["surface_ID"] = -1
    elif geometry.check_coincidence(d_collision, distance):
        particle["event"] += EVENT_COLLISION

    # Check distance to time census
    if d_time_census < distance - COINCIDENCE_TOLERANCE:
        distance = d_time_census
        particle["event"] = EVENT_TIME_CENSUS
        particle["surface_ID"] = -1
    elif geometry.check_coincidence(d_time_census, distance):
        particle["event"] += EVENT_TIME_CENSUS

    # Check distance to time boundary (exclusive event)
    if d_time_boundary < distance + COINCIDENCE_TOLERANCE:
        distance = d_time_boundary
        particle["event"] = EVENT_TIME_BOUNDARY
        particle["surface_ID"] = -1

    # ==================================================================================
    # Move particle
    # ==================================================================================

    # Score tracklength tallies
    if simulation["cycle_active"]:
        # Cell tallies
        cell = simulation["cells"][particle["cell_ID"]]
        for i in range(cell["N_tally"]):
            tally_ID = int(mcdc_get.cell.tally_IDs(i, cell, data))
            tally = simulation["tracklength_tallies"][tally_ID]
            tally_module.score.tracklength_tally(
                particle_container, distance, tally, simulation, data
            )

        # Other tracklength tallies
        for i in range(simulation["N_tracklength_tally"]):
            tally = simulation["tracklength_tallies"][i]

            # Skip cell tallies
            if tally["spatial_filter_type"] == SPATIAL_FILTER_CELL:
                continue

            tally_module.score.tracklength_tally(
                particle_container, distance, tally, simulation, data
            )

    if settings["eigenvalue_mode"]:
        tally_module.score.eigenvalue_tally(
            particle_container, distance, simulation, data
        )

    # Move particle
    particle_module.move(particle_container, distance, simulation, data)
