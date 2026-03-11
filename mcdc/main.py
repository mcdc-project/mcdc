# ======================================================================================
# Run
# ======================================================================================


def run():
    """
    Execute the MC/DC simulation.

    Runs the transport simulation defined by the current problem
    (materials, geometry, sources, tallies, and settings).
    Results are written to an HDF5 output file.

    Command-line arguments (``--N_particle``, ``--output``, etc.) override
    the corresponding settings when provided.
    """
    import mcdc.print_ as print_module
    from mpi4py import MPI

    # TIMER: total
    time_total_start = MPI.Wtime()

    # Get settings and MPI master status
    from mcdc.object_.simulation import simulation as simulationPy

    settings = simulationPy.settings
    master = MPI.COMM_WORLD.Get_rank() == 0

    # Override settings with command-line arguments
    import mcdc.config as config

    if config.args.N_particle is not None:
        settings.N_particle = config.args.N_particle
    if config.args.N_batch is not None:
        settings.N_batch = config.args.N_batch
    if config.args.output is not None:
        settings.output_name = config.args.output
    if config.args.progress_bar is not None:
        settings.use_progress_bar = config.args.progress_bar

    # ==================================================================================
    # Preparation
    # ==================================================================================

    # TIMER: preparation
    time_prep_start = MPI.Wtime()

    # Generate the program state:
    #   - `simulation`: the simulation, storing fixed side data and meta data that
    #                   describes arbitrarily-sized data
    #   - `data`: a long 1D array storing arbitrarily-sized data of the simulation
    # NOTE: The simulation structure to be generated in a container, which is a
    #       a one-sized array that stores the structure. The container is needed to
    #       ensure proper mutability and tracking of the structure when running in
    #       different kinds of machines supported by the Numba compilation framework.
    simulation_container, data = preparation()
    simulation = simulation_container[0]

    # Print headers
    if master:
        print_module.print_banner()
        print_module.print_configuration()
        print(" Now running the particle transport...")
        if settings.eigenvalue_mode:
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

    if settings.eigenvalue_mode:
        simulation_module.eigenvalue_simulation(simulation_container, data)
    else:
        simulation_module.fixed_source_simulation(simulation_container, data)

    # TIMER: simulation
    time_simulation_end = MPI.Wtime()

    # ==================================================================================
    # Working on the output
    # ==================================================================================

    import mcdc.output as output_module

    # TIMER: output
    time_output_start = MPI.Wtime()

    # Generate hdf5 output file
    output_module.generate_output(simulation, data)

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

    # GPU teardowns if needed
    if config.target == "gpu":
        from mcdc.code_factory.gpu.program_builder import teardown_gpu_program

        teardown_gpu_program(simulation)


# ======================================================================================
# Preparation
# ======================================================================================


def preparation():
    import math

    from mpi4py import MPI

    from mcdc.object_.simulation import simulation as simulationPy
    from mcdc.object_.material import MaterialMG

    # ==================================================================================
    # Adjust simulation settings as needed
    # ==================================================================================

    # Get settings
    settings = simulationPy.settings

    # Set nuclear and atomic data for transported particles
    if settings.neutron_transport:
        for nuclide in simulationPy.nuclides:
            nuclide.set_neutron_data()

    # Set physics mode
    if len(simulationPy.materials) == 0:
        # Default physics in dummy mode
        settings.multigroup_mode = True
    else:
        settings.multigroup_mode = isinstance(simulationPy.materials[0], MaterialMG)

    # Set appropriate time boundary
    settings.time_boundary = min(
        [settings.time_boundary] + [tally.time[-1] for tally in simulationPy.tallies]
    )

    # ==================================================================================
    # Adjust simulation parameters as needed
    # ==================================================================================

    # Reset time grid size of all tallies if census-based tally is desired
    if settings.use_census_based_tally:
        N_bin = settings.census_tally_frequency
        for tally in simulationPy.tallies:
            tally._use_census_based_tally(N_bin)

    # Normalize source probability
    norm = 0.0
    for source in simulationPy.sources:
        norm += source.probability
    for source in simulationPy.sources:
        source.probability /= norm

    # Create root universe if not defined
    if len(simulationPy.universes[0].cells) == 0:
        simulationPy.universes[0].cells = simulationPy.cells

    # Initial guess
    simulationPy.k_eff = settings.k_init

    # Activate tally scoring for fixed-source
    if not settings.eigenvalue_mode:
        simulationPy.cycle_active = True
    # All active eigenvalue cycle?
    elif settings.N_inactive == 0:
        simulationPy.cycle_active = True

    # ==================================================================================
    # Set particle bank sizes
    # ==================================================================================

    # Some sizes
    N_particle = settings.N_particle
    N_work = math.ceil(N_particle / MPI.COMM_WORLD.Get_size())
    N_census = settings.N_census

    # Determine bank size
    if settings.eigenvalue_mode or N_census == 1:
        settings.future_bank_buffer_ratio = 0.0
    if not settings.eigenvalue_mode and N_census == 1:
        settings.census_bank_buffer_ratio = 0.0
        settings.source_bank_buffer_ratio = 0.0
    size_active = settings.active_bank_buffer
    size_census = int((settings.census_bank_buffer_ratio) * N_work)
    size_source = int((settings.source_bank_buffer_ratio) * N_work)
    size_future = int((settings.future_bank_buffer_ratio) * N_work)

    # Set bank size
    simulationPy.bank_active.size[0] = size_active
    simulationPy.bank_census.size[0] = size_census
    simulationPy.bank_source.size[0] = size_source
    simulationPy.bank_future.size[0] = size_future

    # ==================================================================================
    # Generate Numba-supported "Objects"
    # ==================================================================================

    from mcdc.code_factory.numba_objects_generator import generate_numba_objects

    if MPI.COMM_WORLD.Get_rank() == 0:
        from mcdc.code_factory.literals_generator import make_literals

        make_literals(simulationPy)
    simulation_container, data = generate_numba_objects(simulationPy)
    simulation = simulation_container[0]

    # Reload mcdc getters and setters
    import importlib
    import mcdc.mcdc_get as mcdc_get
    import mcdc.mcdc_set as mcdc_set

    importlib.reload(mcdc_get)
    importlib.reload(mcdc_set)

    # ==================================================================================
    # Adapt functions as needed
    # ==================================================================================

    # Pick physics model
    import mcdc.transport.physics as physics

    if settings.multigroup_mode:
        physics.neutron.particle_speed = physics.neutron.multigroup.particle_speed
        physics.neutron.macro_xs = physics.neutron.multigroup.macro_xs
        physics.neutron.neutron_production_xs = (
            physics.neutron.multigroup.neutron_production_xs
        )
        physics.neutron.collision = physics.neutron.multigroup.collision

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
    # TODO: Use parallel h5py, may need to compile for speed

    import h5py

    # All ranks, take turn
    for i in range(simulation["mpi_size"]):
        if simulation["mpi_rank"] == i:
            if settings.use_source_file:
                with h5py.File(settings.source_file_name, "r") as f:
                    # Get source particle size
                    N_particle = f["particles_size"][()]

                    # Redistribute work
                    mpi.distribute_work(N_particle, simulation)
                    N_local = simulation["mpi_work_size"]
                    start = simulation["mpi_work_start"]
                    end = start + N_local

                    # Add particles to source bank
                    simulation["bank_source"]["particles"][:N_local] = f["particles"][
                        start:end
                    ]
                    simulation["bank_source"]["size"] = N_local
        MPI.COMM_WORLD.Barrier()

    # ==================================================================================
    # Platform targeting, adapters, and toggles for portability
    # ==================================================================================

    # Set up GPU if needed
    if config.target == "gpu":
        from mcdc.code_factory.gpu.program_builder import setup_gpu

        setup_gpu(mcdc_container, data)

    # ==================================================================================
    # Finalize
    # ==================================================================================

    return simulation_container, data
