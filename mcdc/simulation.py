from mpi4py import MPI

####

import mcdc.objects as objects
import mcdc.rng as rng

from mcdc.constant import (
    SEED_SPLIT_BANK,
    SEED_SPLIT_CENSUS,
    SEED_SPLIT_SOURCE,
)
from mcdc.prints import print_batch_header


# ======================================================================================
# Fixed-Source Simulation
# ======================================================================================


def simulation_fixed_source(data, mcdc, data_new, mcdc_new):
    settings = objects.settings
    master = MPI.COMM_WORLD.Get_rank() == 0

    # TODO: Ensure `mcdc` exist for the lifetime of the program
    # by intentionally leaking their memory

    # ==================================================================================
    # Batch loop
    # ==================================================================================

    N_batch = settings.N_batch
    for i_batch in range(N_batch):
        seed_batch = rng.split_seed(i_batch, settings.rng_seed)

        # TODO: Domain decomposition

        # Print multi-batch header
        if settings.N_batch > 1 and master:
            print_batch_header(i_batch, N_batch)

        # TODO: UQ

        # ==============================================================================
        # Time-census loop
        # ==============================================================================

        N_census = settings.N_census
        for i_census in range(N_census):
            seed_census = rng.split_seed(seed_batch, SEED_SPLIT_CENSUS)

            # TODO: Set census-based tally time grids
            # TODO: Check and accordingly promote future particles to censused particle

            # ==========================================================================
            # Source loop
            # ==========================================================================

            seed_source = rng.split_seed(seed_census, SEED_SPLIT_SOURCE)

            # Progress bar indicator
            N_prog = 0

            # TODO: dd_check_in

            # Loop over particle sources
            work_start = mcdc["mpi_work_start"]
            work_size = mcdc["mpi_work_size"]
            work_end = work_start + work_size

            print(work_size)

            for idx_work in range(work_size):
                generate_source_particle(work_start, idx_work, seed_source, mcdc)

                # Run the source particle and its secondaries
                exhaust_active_bank(data_tally, mcdc, mcdc_new)

                source_closeout(mcdc, idx_work, N_prog, data_tally)
                # Tally history closeout for one-batch simulation
                if N_batch == 1:
                    if not settings.use_census_based_tally:
                        kernel.tally_accumulate(data_tally, mcdc)

                # TODO: Tally history closeout for multi-batch uq simulation

                # Progress printout
                percent = (idx_work + 1.0) / mcdc["mpi_work_size"]
                if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
                    N_prog += 1
                    with objmode():
                        print_progress(percent, mcdc)

            if mcdc["technique"]["domain_decomposition"]:
                source_dd_resolution(data_tally, mcdc)

            # Manage particle banks: population control and work rebalance
            seed_bank = kernel.split_seed(seed_census, SEED_SPLIT_BANK)
            kernel.manage_particle_banks(seed_bank, mcdc)

            # TODO: Time census-based tally closeout

        # Multi-batch closeout
        if N_batch > 1:
            # Reset banks
            kernel.set_bank_size(mcdc["bank_active"], 0)
            kernel.set_bank_size(mcdc["bank_census"], 0)
            kernel.set_bank_size(mcdc["bank_source"], 0)
            kernel.set_bank_size(mcdc["bank_future"], 0)

            # TODO: DD closeout

            # Tally history closeout for multi-batch simulation
            if not settings.use_census_based_tally:
                kernel.tally_reduce(data_tally, mcdc)
                kernel.tally_accumulate(data_tally, mcdc)

                # TODO: UQ closeout

    # Tally closeout
    if not settings.use_census_based_tally:
        # TODO: UQ closeout

        kernel.tally_closeout(data_tally, mcdc)


# ======================================================================================
# Eigenvalue Simulation
# ======================================================================================


def simulation_eigenvalue(mcdc):
    pass
