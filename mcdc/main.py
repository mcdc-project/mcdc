# ======================================================================================
# Run
# ======================================================================================


from mcdc import mcdc_get
from mcdc.print_ import print_error, print_structure

# Cache for preparation() result - speeds up repeated visualizations
_visualize_cache = None


def run():
    import mcdc.print_ as print_module
    from mpi4py import MPI

    # Timer: total
    time_total_start = MPI.Wtime()

    from mcdc.object_.simulation import simulation

    settings = simulation.settings
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

    # Timer: preparation
    time_prep_start = MPI.Wtime()

    mcdc_arr, data = preparation()
    mcdc = mcdc_arr[0]

    # Print headers
    if master:
        print_module.print_banner()
        print_module.print_configuration()
        print(" Now running TNT...")
        if settings.eigenvalue_mode:
            print_module.print_eigenvalue_header(mcdc)

    # Timer: preparation
    time_prep_end = MPI.Wtime()

    # ==================================================================================
    # Running the simulation
    # ==================================================================================

    # Timer: simulation
    time_simulation_start = MPI.Wtime()

    # Run simulation
    import mcdc.transport.simulation as simulation_module

    if settings.eigenvalue_mode:
        simulation_module.eigenvalue_simulation(mcdc_arr, data)
    else:
        simulation_module.fixed_source_simulation(mcdc_arr, data)

    # Timer: simulation
    time_simulation_end = MPI.Wtime()

    # ==================================================================================
    # Working on the output
    # ==================================================================================

    import mcdc.output as output_module

    # Timer: output
    time_output_start = MPI.Wtime()

    # Generate hdf5 output file
    output_module.generate_output(mcdc, data)

    # Timer: output
    time_output_end = MPI.Wtime()

    # Final barrier
    MPI.COMM_WORLD.Barrier()

    # Timer: total
    time_total_end = MPI.Wtime()

    # Manage timers
    mcdc["runtime_total"] = time_total_end - time_total_start
    mcdc["runtime_preparation"] = time_prep_end - time_prep_start
    mcdc["runtime_simulation"] = time_simulation_end - time_simulation_start
    mcdc["runtime_output"] = time_output_end - time_output_start
    output_module.create_runtime_datasets(mcdc)
    if master:
        print_module.print_runtime(mcdc)

    # ==================================================================================
    # Finalizing
    # ==================================================================================

    # GPU teardowns
    from mcdc.transport.simulation import teardown_gpu

    teardown_gpu(mcdc)


# ======================================================================================
# Preparation
# ======================================================================================


def preparation():
    import math
    import numpy as np

    from mpi4py import MPI

    from mcdc.object_.simulation import simulation
    from mcdc.object_.material import MaterialMG

    # ==================================================================================
    # Simulation settings
    # ==================================================================================

    # Get settings
    settings = simulation.settings

    # Set physics mode
    settings.multigroup_mode = isinstance(simulation.materials[0], MaterialMG)

    # Set appropriate time boundary
    settings.time_boundary = min(
        [settings.time_boundary] + [tally.time[-1] for tally in simulation.tallies]
    )

    # Reset time grid size of all tallies if census-based tally is desired
    if settings.use_census_based_tally:
        N_bin = settings.census_tally_frequency
        for tally in simulation.tallies:
            tally._use_census_based_tally(N_bin)

    # ==================================================================================
    # Simulation parameters
    # ==================================================================================

    # Normalize source probability
    norm = 0.0
    for source in simulation.sources:
        norm += source.probability
    for source in simulation.sources:
        source.probability /= norm

    # Create root universe if not defined
    if len(simulation.universes[0].cells) == 0:
        simulation.universes[0].cells = simulation.cells

    # Initial guess
    simulation.k_eff = settings.k_init

    # Activate tally scoring for fixed-source
    if not settings.eigenvalue_mode:
        simulation.cycle_active = True
    # All active eigenvalue cycle?
    elif settings.N_inactive == 0:
        simulation.cycle_active = True

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
    simulation.bank_active.size[0] = size_active
    simulation.bank_census.size[0] = size_census
    simulation.bank_source.size[0] = size_source
    simulation.bank_future.size[0] = size_future

    # ==================================================================================
    # Generate Numba-supported "Objects"
    # ==================================================================================

    from mcdc.code_factory.numba_objects_generator import generate_numba_objects

    if MPI.COMM_WORLD.Get_rank() == 0:
        from mcdc.code_factory.numba_objects_generator import make_literals

        make_literals(simulation)
    mcdc_arr, data = generate_numba_objects(simulation)
    mcdc = mcdc_arr[0]

    # Reload mcdc getters and setters
    import importlib
    import mcdc.mcdc_get as mcdc_get
    import mcdc.mcdc_set as mcdc_set

    importlib.reload(mcdc_get)
    importlib.reload(mcdc_set)

    # ==================================================================================
    # Platform Targeting, Adapters, Toggles, etc
    # ==================================================================================

    # Adapt kernels
    import numba as nb
    import mcdc.code_factory.adapt as adapt
    import mcdc.config as config
    import mcdc.transport.mpi as mpi

    # TODO: Find out why the following is needed to avoid circular import
    import mcdc.transport.particle_bank as particle_bank_module

    settings.target_gpu = True if config.target == "gpu" else False

    if config.target == "gpu":
        import mcdc.numba_types as type_

        if MPI.COMM_WORLD.Get_rank() != 0:
            adapt.harm.config.should_compile(adapt.harm.config.ShouldCompile.NEVER)
        elif config.caching == False:
            adapt.harm.config.should_compile(adapt.harm.config.ShouldCompile.ALWAYS)
        if not adapt.HAS_HARMONIZE:
            print_error(
                "No module named 'harmonize' - GPU functionality not available. "
            )
        adapt.gpu_forward_declare(
            config.args,
            data.shape,
            type_.simulation,
            type_.particle,
            type_.particle_data,
        )

    adapt.eval_toggle()
    adapt.target_for(config.target)
    if config.target == "gpu":
        build_gpu_progs(input_deck, config.args)
    adapt.nopython_mode((config.mode == "numba") or (config.mode == "numba_debug"))

    # ==================================================================================
    # Source file
    #   TODO: Use parallel h5py
    # ==================================================================================

    import h5py

    # All ranks, take turn
    for i in range(mcdc["mpi_size"]):
        if mcdc["mpi_rank"] == i:
            if settings.use_source_file:
                with h5py.File(settings.source_file_name, "r") as f:
                    # Get source particle size
                    N_particle = f["particles_size"][()]

                    # Redistribute work
                    mpi.distribute_work(N_particle, mcdc)
                    N_local = mcdc["mpi_work_size"]
                    start = mcdc["mpi_work_start"]
                    end = start + N_local

                    # Add particles to source bank
                    mcdc["bank_source"]["particles"][:N_local] = f["particles"][
                        start:end
                    ]
                    mcdc["bank_source"]["size"] = N_local
        MPI.COMM_WORLD.Barrier()

    # ==================================================================================
    # Adapt functions
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
    import mcdc.transport.rng as rng

    if config.mode == "python":
        rng.wrapping_add = rng.wrapping_add_python
        rng.wrapping_mul = rng.wrapping_mul_python

    # ==================================================================================
    # Finalize data: wrapping into a tuple
    # ==================================================================================

    from mcdc.transport.simulation import setup_gpu

    setup_gpu(mcdc)

    return mcdc_arr, data


# ======================================================================================
# Visualize geometry
# ======================================================================================

from numba import njit
import numpy as np
from mcdc.transport.geometry.interface import locate_particle


@njit(cache=True)
def _compute_material_row(
    first_coord,
    second_midpoint,
    first_key_idx,
    second_key_idx,
    reference_key_idx,
    reference_val,
    time_val,
    particle_arr,
    mcdc,
    data,
):
    """
    Compute material IDs for a single row of pixels using numba.

    Parameters
    ----------
    first_coord : float
        First axis coordinate
    second_midpoint : np.ndarray
        Midpoints along the second axis
    first_key_idx : int
        Index for first axis (0=x, 1=y, 2=z)
    second_key_idx : int
        Index for second axis (0=x, 1=y, 2=z)
    reference_key_idx : int
        Index for reference axis (0=x, 1=y, 2=z)
    reference_val : float
        Value for the reference (slice) coordinate
    time_val : float
        Time value for the visualization
    particle_arr : np.ndarray
        Pre-allocated particle array of size (1,) reused for computation.
    mcdc : structured array
        MCDC simulation data
    data : structured array
        Additional simulation data

    Returns
    -------
    np.ndarray
        1D array of material IDs for this row (-1 for lost/outside particles).
    """
    n_second = len(second_midpoint)
    row_materials = np.empty(n_second, dtype=np.int32)

    particle = particle_arr[0]

    # Set time and energy
    particle["t"] = time_val
    particle["g"] = 0
    particle["E"] = 1e6
    particle["ux"] = 0.0
    particle["uy"] = 0.0
    particle["uz"] = 1.0

    # Set reference coordinate
    if reference_key_idx == 0:
        particle["x"] = reference_val
    elif reference_key_idx == 1:
        particle["y"] = reference_val
    else:
        particle["z"] = reference_val

    # Set first axis coordinate
    if first_key_idx == 0:
        particle["x"] = first_coord
    elif first_key_idx == 1:
        particle["y"] = first_coord
    else:
        particle["z"] = first_coord

    for j in range(n_second):
        # Set second axis coordinate
        if second_key_idx == 0:
            particle["x"] = second_midpoint[j]
        elif second_key_idx == 1:
            particle["y"] = second_midpoint[j]
        else:
            particle["z"] = second_midpoint[j]

        # Reset IDs for fresh lookup
        particle["cell_ID"] = -1
        particle["material_ID"] = -1

        if locate_particle(particle_arr, mcdc, data):
            row_materials[j] = particle["material_ID"]
        else:
            row_materials[j] = -1

    return row_materials


def visualize(
    vis_type,
    x=0.0,
    y=0.0,
    z=0.0,
    pixels=(100, 100),
    colors=None,
    time=[0.0],
    save_as=None,
):
    """
    2D visualization of the created model

    Parameters
    ----------
    vis_plane : {'xy', 'yz', 'xz', 'zx', 'yz', 'zy'}
        Axis plane to visualize
    x : float or array_like
        Plane x-position (float) for 'yz' plot. Range of x-axis for 'xy' or 'xz' plot.
    y : float or array_like
        Plane y-position (float) for 'xz' plot. Range of y-axis for 'xy' or 'yz' plot.
    z : float or array_like
        Plane z-position (float) for 'xy' plot. Range of z-axis for 'xz' or 'yz' plot.
    time : array_like
        Times at which the geometry snapshots are taken
    pixels : array_like
        Number of respective pixels in the two axes in vis_plane
    colors : array_like
        List of pairs of material and its color
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import sys

    from matplotlib import colors as mpl_colors

    # Use cached preparation if available
    global _visualize_cache
    if _visualize_cache is None:
        _visualize_cache = preparation()
    mcdc_container, data = _visualize_cache
    mcdc = mcdc_container[0]

    import mcdc.numba_types as type_

    # Color assignment for materials (by material ID)
    if colors is not None:
        new_colors = {}
        for item in colors.items():
            new_colors[item[0].ID] = mpl_colors.to_rgb(item[1])
        colors = new_colors
    else:
        colors = {}
        for i in range(len(mcdc["materials"])):
            colors[i] = plt.cm.Set1(i)[:-1]
    WHITE = mpl_colors.to_rgb("white")

    # Set reference axis
    for axis in ["x", "y", "z"]:
        if axis not in vis_type:
            reference_key = axis

    if reference_key == "x":
        reference = x
    elif reference_key == "y":
        reference = y
    elif reference_key == "z":
        reference = z

    # Set first and second axes
    first_key = vis_type[0]
    second_key = vis_type[1]

    if first_key == "x":
        first = x
    elif first_key == "y":
        first = y
    elif first_key == "z":
        first = z

    if second_key == "x":
        second = x
    elif second_key == "y":
        second = y
    elif second_key == "z":
        second = z

    # Axis pixels grids and midpoints
    first_grid = np.linspace(first[0], first[1], pixels[0] + 1)
    first_midpoint = 0.5 * (first_grid[1:] + first_grid[:-1])

    second_grid = np.linspace(second[0], second[1], pixels[1] + 1)
    second_midpoint = 0.5 * (second_grid[1:] + second_grid[:-1])

    # Map axis keys to indices for the numba function
    key_to_idx = {"x": 0, "y": 1, "z": 2}
    first_idx = key_to_idx[first_key]
    second_idx = key_to_idx[second_key]
    ref_idx = key_to_idx[reference_key]

    # Create Color Palette for fast lookup
    max_id = max(colors.keys())
    palette = np.zeros((max_id + 1, 3))
    for mat_id, col in colors.items():
        palette[mat_id] = col

    for t in time:
        # Pre-allocate particles
        particle_container = np.zeros(1, dtype=type_.particle)

        # Material IDs grid
        material_ids = np.empty((pixels[0], pixels[1]), dtype=np.int32)
        last_progress = -1

        # Loop over rows with progress bar
        for i in range(pixels[0]):
            row_materials = _compute_material_row(
                first_midpoint[i],
                second_midpoint,
                first_idx,
                second_idx,
                ref_idx,
                reference,
                t,
                particle_container,
                mcdc,
                data,
            )
            material_ids[i, :] = row_materials

            # Update progress bar
            percent = (i + 1) / pixels[0]
            if int(percent * 100) > last_progress:
                last_progress = int(percent * 100)
                sys.stdout.write("\r Visualizing: [%-28s] %d%%" % ("=" * int(percent * 28), percent * 100))
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Color Mapping
        pixel_data = np.full(pixels + (3,), WHITE)
        valid_mask = material_ids >= 0
        pixel_data[valid_mask] = palette[
            material_ids[valid_mask]
        ]  # Apply colors using palette lookup

        pixel_data = np.transpose(pixel_data, (1, 0, 2))
        plt.imshow(pixel_data, origin="lower", extent=first + second)
        plt.xlabel(first_key + " [cm]")
        plt.ylabel(second_key + " [cm]")
        plt.title(reference_key + " = %.2f cm" % reference + ", time = %.2f s" % t)
        if save_as is not None:
            if len(time) > 1:
                plt.savefig(f"{save_as}_{t:03}.png")
            else:
                plt.savefig(save_as + ".png")
            plt.clf()
        else:
            plt.show()
