from mcdc.object_.simulation import Simulation

# ======================================================================================
# Run Simulation
# ======================================================================================


def run_simulation(simulationPy: Simulation):
    """Compile when needed, prepare runtime state, and execute a simulation."""
    import mcdc.print_ as print_module
    from mpi4py import MPI

    # TIMER: total
    time_total_start = MPI.Wtime()

    # Get settings and MPI master status
    settings = simulationPy.settings
    master = MPI.COMM_WORLD.Get_rank() == 0

    # ==================================================================================
    # Preparation
    # ==================================================================================

    # TIMER: preparation
    time_prep_start = MPI.Wtime()

    # Generate the program state:
    #   - `simulation`: the simulation structure, storing fixed side data and meta data
    #                   that describes arbitrarily-sized data
    #   - `data`: a long 1D array storing arbitrarily-sized data of the simulation
    # NOTE: The simulation structure is generated in a one-sized array container.
    #       The use of container is necessary to ensure proper mutability and tracking
    #       of the structure when running in different kinds of machines supported by
    #       the Numba-based compilation framework.
    simulation_container, data = prepare(simulationPy)
    simulation = simulation_container[0]

    # Prevent intermediate census tallies from a previous run from being recombined.
    import mcdc.output as output_module

    if settings.use_census_based_tally:
        if master:
            output_module.clear_census_based_tally_files(settings)
        MPI.COMM_WORLD.Barrier()

    # Print headers
    if master:
        print_module.print_banner()
        print_module.print_configuration()
        print(" Now running the particle transport...")
        if settings.neutron_eigenvalue_mode:
            print_module.print_eigenvalue_header(simulation)

    # TIMER: preparation
    time_prep_end = MPI.Wtime()

    # ==================================================================================
    # Running the simulation
    # ==================================================================================

    # TIMER: simulation
    time_simulation_start = MPI.Wtime()

    # Run simulation
    import mcdc.transport.simulation as simulation_module

    if settings.neutron_eigenvalue_mode:
        simulation_module.eigenvalue_simulation(simulation_container, data)
    else:
        simulation_module.fixed_source_simulation(simulation_container, data)

    # TIMER: simulation
    time_simulation_end = MPI.Wtime()

    # ==================================================================================
    # Working on the output
    # ==================================================================================

    # TIMER: output
    time_output_start = MPI.Wtime()

    # Generate hdf5 output file
    output_module.generate_output(simulation, data, simulationPy)

    # Combine per-batch, per-census tally files into the main output
    if settings.use_census_based_tally:
        output_module.recombine_tallies(simulationPy, simulation)

    # TIMER: output
    time_output_end = MPI.Wtime()

    # Final barrier
    MPI.COMM_WORLD.Barrier()

    # TIMER: total
    time_total_end = MPI.Wtime()

    # Manage timers
    simulation["runtime_total"] = time_total_end - time_total_start
    simulation["runtime_preparation"] = time_prep_end - time_prep_start
    simulation["runtime_simulation"] = time_simulation_end - time_simulation_start
    simulation["runtime_output"] = time_output_end - time_output_start
    output_module.create_runtime_datasets(simulation)
    if master:
        print_module.print_runtime(simulation)

    # ==================================================================================
    # Finalizing
    # ==================================================================================

    finalize(simulation)


# ======================================================================================
# Prepare
# ======================================================================================


def prepare(simulationPy: Simulation):
    """Create framework-owned runtime state for a compiled simulation.

    Model-specific finalization occurs during :meth:`mcdc.Simulation.compile`.
    This function packs that model, allocates execution resources, configures
    the selected backend, and loads any external source-particle state.
    """
    # ==================================================================================
    # Prepare problem-dependent runtime state
    # ==================================================================================

    from mcdc.code_factory.numba_layers_generator import generate_numba_layers
    from mcdc.code_factory.literals_generator import make_literals

    make_literals(simulationPy)

    simulation_container, data = generate_numba_layers(simulationPy)
    simulation = simulation_container[0]

    # Pick Python-version RNG if needed
    import mcdc.config as config
    import mcdc.transport.rng as rng

    if config.mode == "python":
        rng.wrapping_add = rng.wrapping_add_python
        rng.wrapping_mul = rng.wrapping_mul_python

    # TODO: Find out why the following is needed to avoid circular import
    import mcdc.transport.particle_bank as particle_bank_module

    # ==================================================================================
    # Source particles from file
    # ==================================================================================
    # TODO: Re-enable file-backed source initialization after its particle-bank
    # schema and MPI redistribution path are updated.
    #
    # import h5py
    # import mcdc.transport.mpi as mpi
    # from mpi4py import MPI
    #
    # for i in range(simulation["mpi_size"]):
    #     if simulation["mpi_rank"] == i and settings.use_source_file:
    #         with h5py.File(settings.source_file_name, "r") as f:
    #             N_particle = f["particles_size"][()]
    #             mpi.distribute_work(N_particle, simulation)
    #             N_local = simulation["mpi_work_size"]
    #             start = simulation["mpi_work_start"]
    #             end = start + N_local
    #             simulation["bank_source"]["particle_data"][:N_local] = f[
    #                 "particles"
    #             ][start:end]
    #             simulation["bank_source"]["size"] = N_local
    #     MPI.COMM_WORLD.Barrier()

    # ==================================================================================
    # Finalize
    # ==================================================================================

    return simulation_container, data


# ======================================================================================
# Misc.
# ======================================================================================


def finalize(simulation):
    import mcdc.config as config

    # GPU teardowns if needed
    if config.target == "gpu":
        from mcdc.code_factory.gpu.program_builder import teardown_gpu_program

        teardown_gpu_program(simulation)
