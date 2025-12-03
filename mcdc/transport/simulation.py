import numpy as np

from numba import njit, objmode, uint64

####

import mcdc.code_factory.adapt as adapt
import mcdc.config as config
import mcdc.mcdc_get as mcdc_get
import mcdc.object_.numba_types as type_
import mcdc.output as output_module
import mcdc.transport.geometry as geometry
import mcdc.transport.mpi as mpi
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.physics as physics
import mcdc.transport.rng as rng
import mcdc.transport.tally as tally_module
import mcdc.transport.technique as technique

from mcdc.constant import *
from mcdc.print_ import (
    print_header_batch,
    print_progress,
    print_progress_eigenvalue,
)
from mcdc.transport.source import source_particle

caching = config.caching


# ======================================================================================
# Fixed-source simulation
# ======================================================================================


def fixed_source_simulation(mcdc_arr, data):
    # Ensure `mcdc` exist for the lifetime of the program by intentionally leaking their memory
    # adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Get some settings
    settings = mcdc["settings"]
    N_batch = settings["N_batch"]
    N_particle = settings["N_particle"]
    N_census = settings["N_census"]
    use_census_based_tally = settings["use_census_based_tally"]

    # Loop over batches
    for idx_batch in range(N_batch):
        mcdc["idx_batch"] = idx_batch
        seed_batch = rng.split_seed(uint64(idx_batch), settings["rng_seed"])

        # Distribute work
        mpi.distribute_work(N_particle, mcdc)

        # Print multi-batch header
        if N_batch > 1:
            with objmode():
                print_header_batch(idx_batch, N_batch)

        # Loop over time censuses
        for idx_census in range(N_census):
            mcdc["idx_census"] = idx_census
            seed_census = rng.split_seed(uint64(seed_batch), rng.SEED_SPLIT_CENSUS)

            # Reset tally time filters if census-based tally is used
            if use_census_based_tally:
                tally_module.filter.set_census_based_time_grid(mcdc, data)

            # Check and accordingly promote future particles to censused particles
            if particle_bank_module.get_bank_size(mcdc["bank_future"]) > 0:
                particle_bank_module.check_future_bank(mcdc, data)

            # Loop over source particles
            seed_source = rng.split_seed(uint64(seed_census), rng.SEED_SPLIT_SOURCE)
            loop_source(uint64(seed_source), mcdc, data)

            # Manage particle banks: population control and work rebalance
            particle_bank_module.manage_particle_banks(mcdc)

            # Time census-based tally closeout
            if use_census_based_tally:
                tally_module.closeout.reduce(mcdc, data)
                tally_module.closeout.accumulate(mcdc, data)
                if mcdc["mpi_master"]:
                    with objmode():
                        output_module.generate_census_based_tally(mcdc, data)
                tally_module.closeout.reset_sum_bins(mcdc, data)

            # Terminate census loop if all banks are empty
            if (
                idx_census > 0
                and particle_bank_module.total_size(mcdc["bank_source"]) == 0
                and particle_bank_module.total_size(mcdc["bank_census"]) == 0
                and particle_bank_module.total_size(mcdc["bank_future"]) == 0
            ):
                break

        # Multi-batch closeout
        if N_batch > 1:
            # Reset banks
            particle_bank_module.set_bank_size(mcdc["bank_active"], 0)
            particle_bank_module.set_bank_size(mcdc["bank_census"], 0)
            particle_bank_module.set_bank_size(mcdc["bank_source"], 0)
            particle_bank_module.set_bank_size(mcdc["bank_future"], 0)

            if not use_census_based_tally:
                # Tally history closeout
                tally_module.closeout.reduce(mcdc, data)
                tally_module.closeout.accumulate(mcdc, data)

    # Tally closeout
    if not use_census_based_tally:
        tally_module.closeout.finalize(mcdc, data)


# =========================================================================
# Eigenvalue simulation
# =========================================================================


def eigenvalue_simulation(mcdc_arr, data):
    # Ensure `mcdc` exist for the lifetime of the program
    # by intentionally leaking their memory
    # adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Get some settings
    settings = mcdc["settings"]
    N_inactive = settings["N_inactive"]
    N_cycle = settings["N_cycle"]
    N_particle = settings["N_particle"]

    # Distribute work
    mpi.distribute_work(N_particle, mcdc)

    # Loop over power iteration cycles
    for idx_cycle in range(N_cycle):
        mcdc["idx_cycle"] = idx_cycle
        seed_cycle = rng.split_seed(uint64(idx_cycle), settings["rng_seed"])

        # Loop over source particles
        loop_source(uint64(seed_cycle), mcdc, data)

        # Tally "history" closeout
        tally_module.closeout.eigenvalue_cycle(mcdc, data)
        if mcdc["cycle_active"]:
            tally_module.closeout.reduce(mcdc, data)
            tally_module.closeout.accumulate(mcdc, data)

        # Manage particle banks: population control and work rebalance
        particle_bank_module.manage_particle_banks(mcdc)

        # Print progress
        with objmode():
            print_progress_eigenvalue(mcdc, data)

        # Entering active cycle?
        mcdc["idx_cycle"] += 1
        if mcdc["idx_cycle"] >= N_inactive:
            mcdc["cycle_active"] = True

    # Tally closeout
    tally_module.closeout.finalize(mcdc, data)
    tally_module.closeout.eigenvalue_simulation(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def loop_source(seed, mcdc, data):
    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]

    for idx_work in range(work_size):
        mcdc["idx_work"] = work_start + idx_work
        generate_source_particle(work_start, idx_work, seed, mcdc, data)

        # Run the source particle and its secondaries
        exhaust_active_bank(mcdc, data)

        source_closeout(mcdc, idx_work, N_prog, data)


@njit
def generate_source_particle(work_start, idx_work, seed, prog, data):
    """Get a source particle and put into one of the banks"""
    mcdc = adapt.mcdc_global(prog)
    settings = mcdc["settings"]

    particle_container = np.zeros(1, type_.particle_data)
    particle = particle_container[0]

    # Get from fixed-source?
    if particle_bank_module.get_bank_size(mcdc["bank_source"]) == 0:
        # Sample source
        seed_work = rng.split_seed(work_start + idx_work, seed)
        source_particle(particle_container, seed_work, mcdc, data)

    # Get from source bank
    else:
        particle_container = mcdc["bank_source"]["particles"][idx_work : (idx_work + 1)]
        particle = particle_container[0]

    # Skip if beyond time boundary
    if particle["t"] > settings["time_boundary"]:
        return

    # Check if it is beyond current or next census times
    hit_census = False
    hit_next_census = False
    idx_census = mcdc["idx_census"]

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
        particle_bank_module.add_active(particle_container, prog)
    elif not hit_next_census:
        # Particle will participate after the current census
        particle_bank_module.add_census(particle_container, prog)
    else:
        # Particle will participate in the future
        particle_bank_module.add_future(particle_container, prog)


@njit
def exhaust_active_bank(prog, data):
    mcdc = adapt.mcdc_global(prog)
    particle_container = np.zeros(1, type_.particle)
    particle = particle_container[0]

    # Loop until active bank is exhausted
    while particle_bank_module.get_bank_size(mcdc["bank_active"]) > 0:
        # Get particle from active bank
        particle_bank_module.get_particle(particle_container, mcdc["bank_active"], mcdc)

        prep_particle(particle_container, prog)

        # Particle loop
        loop_particle(particle_container, mcdc, data)


@njit
def prep_particle(particle_container, prog):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)


@njit
def source_closeout(prog, idx_work, N_prog, data):
    mcdc = adapt.mcdc_global(prog)

    # Tally history closeout for one-batch fixed-source simulation
    if not mcdc["settings"]["eigenvalue_mode"] and mcdc["settings"]["N_batch"] == 1:
        if not mcdc["settings"]["use_census_based_tally"]:
            tally_module.closeout.accumulate(mcdc, data)

    # Progress printout
    percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
    if mcdc["settings"]["use_progress_bar"] and int(percent * 100.0) > N_prog:
        N_prog += 1
        with objmode():
            print_progress(percent, mcdc)


# ======================================================================================
# Particle loop
# ======================================================================================


@njit
def loop_particle(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)

    while particle["alive"]:
        step_particle(particle_container, prog, data)


@njit
def step_particle(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)

    # Determine and move to event
    move_to_event(particle_container, mcdc, data)

    # Execute events
    if particle["event"] == EVENT_LOST:
        return

    # Collision
    if particle["event"] & EVENT_COLLISION:
        physics.collision(particle_container, prog, data)

    # Surface and domain crossing
    if particle["event"] & EVENT_SURFACE_CROSSING:
        geometry.surface_crossing(particle_container, prog, data)

    # Census time crossing
    if particle["event"] & EVENT_TIME_CENSUS:
        particle_bank_module.add_census(particle_container, prog)
        particle["alive"] = False

    # Time boundary crossing
    if particle["event"] & EVENT_TIME_BOUNDARY:
        particle["alive"] = False

    # Weight roulette
    if particle["alive"]:
        technique.weight_roulette(particle_container, prog)


@njit
def move_to_event(particle_container, mcdc, data):
    settings = mcdc["settings"]

    # ==================================================================================
    # Preparation (as needed)
    # ==================================================================================
    particle = particle_container[0]

    # Multigroup preparation
    #   In MG mode, particle speed is material-dependent.
    if settings["multigroup_mode"]:
        # If material is not identified yet, locate the particle
        if particle["material_ID"] == -1:
            if not geometry.locate_particle(particle_container, mcdc, data):
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

    d_boundary = geometry.inspect_geometry(particle_container, mcdc, data)

    # Particle is lost?
    if particle["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Get distances to other events
    # ==================================================================================

    # Distance to domain
    speed = physics.particle_speed(particle_container, mcdc, data)

    # Distance to time boundary
    d_time_boundary = speed * (settings["time_boundary"] - particle["t"])

    # Distance to census time
    idx = mcdc["idx_census"]
    d_time_census = speed * (
        mcdc_get.settings.census_time(idx, settings, data) - particle["t"]
    )

    # Distance to next collision
    d_collision = physics.collision_distance(particle_container, mcdc, data)

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
    if mcdc["cycle_active"]:
        # Cell tallies
        cell = mcdc["cells"][particle["cell_ID"]]
        for i in range(cell["N_tally"]):
            tally_ID = int(mcdc_get.cell.tally_IDs(i, cell, data))
            tally = mcdc["cell_tallies"][tally_ID]
            tally_module.score.tracklength_tally(
                particle_container, distance, tally, mcdc, data
            )

        # Global tallies
        for i in range(mcdc["N_global_tally"]):
            tally = mcdc["global_tallies"][i]
            tally_module.score.tracklength_tally(
                particle_container, distance, tally, mcdc, data
            )

        # Mesh tallies
        for i in range(mcdc["N_mesh_tally"]):
            tally = mcdc["mesh_tallies"][i]
            tally_module.score.mesh_tally(
                particle_container, distance, tally, mcdc, data
            )

    if settings["eigenvalue_mode"]:
        tally_module.score.eigenvalue_tally(particle_container, distance, mcdc, data)

    # Move particle
    particle_module.move(particle_container, distance, mcdc, data)
