import h5py
import importlib.metadata
import numpy as np

####

import mcdc.mcdc_get as mcdc_get
import mcdc.print_ as print_module

from mcdc.constant import (
    MESH_UNIFORM,
    MESH_STRUCTURED,
    TALLY_MESH,
)

# ======================================================================================
# Main output
# ======================================================================================


def generate_output(mcdc, data):
    from mcdc import simulation

    if not mcdc["mpi_master"]:
        return

    settings = mcdc["settings"]

    # Header
    if settings["use_progress_bar"]:
        print_module.print_msg("")
    print_module.print_msg(" Generating output HDF5 files...")

    # Create the file
    file = h5py.File(settings["output_name"] + ".h5", "w")

    # Version
    file["version"] = importlib.metadata.version("mcdc")

    # Settings
    create_object_dataset(file, "settings", simulation.settings)

    # No need to output tally if time census-based tally is used
    if mcdc["settings"]["use_census_based_tally"]:
        return

    # Tallies
    create_tally_dataset(file, mcdc, data)

    # Eigenvalues
    if mcdc["settings"]["eigenvalue_mode"]:
        N_cycle = mcdc["settings"]["N_cycle"]
        file.create_dataset(
            "k_cycle", data=mcdc_get.simulation.k_cycle_chunk(0, N_cycle, mcdc, data)
        )
        file.create_dataset("k_mean", data=mcdc["k_avg_running"])
        file.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
        file.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
        file.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
        file.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
        file.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
        file.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
        file.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
        if mcdc["settings"]["use_gyration_radius"]:
            file.create_dataset(
                "gyration_radius",
                data=mcdc_get.simulation.gyration_radius_chunk(0, N_cycle, mcdc, data),
            )

    # Save particle?
    if mcdc["settings"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"][0]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Remove unwanted particle fields
        neutrons = np.concatenate(neutrons[:])

        # Create dataset
        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
            file.create_dataset("particles", data=neutrons[:])
            file.create_dataset("particles_size", data=len(neutrons[:]))

    # Close the file
    file.close()


# ======================================================================================
# Input objects
# ======================================================================================


def create_object_dataset(file, group_name, object_):
    for name in [
        x
        for x in dir(object_)
        if (not x.startswith("__") and not callable(getattr(object_, x)))
    ]:
        file[f"{group_name}/{name}"] = getattr(object_, name)


# ======================================================================================
# Runtimes
# ======================================================================================


def create_runtime_datasets(mcdc):
    import h5py
    import mcdc.config as config

    if not mcdc["mpi_master"]:
        return

    base_name = mcdc["settings"]["output_name"]

    main_output = h5py.File(f"{base_name}.h5", "a")
    create_runtime_dataset(main_output, mcdc)
    main_output.close()

    if config.args.runtime_output:
        runtime_output = h5py.File(f"{base_name}.h5", "w")
        create_runtime_dataset(runtime_output, mcdc)
        runtime_output.close()


def create_runtime_dataset(file, mcdc):
    for name in [
        "total",
        "preparation",
        "simulation",
        "output",
        "bank_management",
    ]:
        file.create_dataset(f"runtime/{name}", data=np.array([mcdc["runtime_" + name]]))


# ======================================================================================
# Tally
# ======================================================================================


def create_tally_dataset(file, mcdc, data):
    from mcdc.object_.tally import decode_score_type

    # Loop over all tally types
    for tally in mcdc["tallies"]:
        tally_name = tally["name"]

        # Filter grids
        file.create_dataset(
            f"tallies/{tally_name}/grid/mu", data=mcdc_get.tally.mu_all(tally, data)
        )
        file.create_dataset(
            f"tallies/{tally_name}/grid/azi",
            data=mcdc_get.tally.azi_all(tally, data),
        )
        file.create_dataset(
            f"tallies/{tally_name}/grid/energy",
            data=mcdc_get.tally.energy_all(tally, data),
        )
        file.create_dataset(
            f"tallies/{tally_name}/grid/time",
            data=mcdc_get.tally.time_all(tally, data),
        )

        # Mesh grid (TODO: Make mesh dataset in a separate group)
        if tally["child_type"] == TALLY_MESH:
            mesh_tally = mcdc["mesh_tallies"][tally["child_ID"]]
            mesh_base = mcdc["meshes"][mesh_tally["mesh_ID"]]
            mesh_type = mesh_base["child_type"]
            mesh_ID = mesh_base["child_ID"]
            if mesh_type == MESH_UNIFORM:
                mesh = mcdc["uniform_meshes"][mesh_ID]
                x = np.linspace(
                    mesh["x0"], mesh["x0"] + mesh["dx"] * mesh["Nx"], mesh["Nx"] + 1
                )
                y = np.linspace(
                    mesh["y0"], mesh["y0"] + mesh["dy"] * mesh["Ny"], mesh["Ny"] + 1
                )
                z = np.linspace(
                    mesh["z0"], mesh["z0"] + mesh["dz"] * mesh["Nz"], mesh["Nz"] + 1
                )
            elif mesh_type == MESH_STRUCTURED:
                mesh = mcdc["structured_meshes"][mesh_ID]
                x = mcdc_get.structured_mesh.x_all(mesh, data)
                y = mcdc_get.structured_mesh.y_all(mesh, data)
                z = mcdc_get.structured_mesh.z_all(mesh, data)
            file.create_dataset(f"tallies/{tally_name}/grid/x", data=x)
            file.create_dataset(f"tallies/{tally_name}/grid/y", data=y)
            file.create_dataset(f"tallies/{tally_name}/grid/z", data=z)

        # Get and reshape tally
        N_bin = tally["bin_length"]
        start_mean = tally["bin_sum_offset"]
        start_sdev = tally["bin_sum_square_offset"]
        mean = data[start_mean : start_mean + N_bin]
        sdev = data[start_sdev : start_sdev + N_bin]
        shape = tuple([int(x) for x in mcdc_get.tally.bin_shape_all(tally, data)])
        mean = mean.reshape(shape)
        sdev = sdev.reshape(shape)

        # Roll tally so that score is in the front
        roll_reference = 4
        if tally["child_type"] == TALLY_MESH:
            roll_reference = 7
        mean = np.rollaxis(mean, roll_reference, 0)
        sdev = np.rollaxis(sdev, roll_reference, 0)

        # Iterate over scores
        for i in range(tally["scores_length"]):
            score_type = mcdc_get.tally.scores(i, tally, data)
            score_mean = np.squeeze(mean[i])
            score_sdev = np.squeeze(sdev[i])
            score_name = decode_score_type(score_type, lower_case=True)
            group_name = f"tallies/{tally_name}/{score_name}/"
            file.create_dataset(group_name + "mean", data=score_mean)
            file.create_dataset(group_name + "sdev", data=score_sdev)


def generate_census_based_tally(mcdc, data):
    idx_batch = mcdc["idx_batch"]
    idx_census = mcdc["idx_census"]
    base_name = mcdc["settings"]["output_name"]

    # Create or get the file
    file_name = f"{base_name}-batch_{idx_batch}-census_{idx_census}.h5"
    file = h5py.File(file_name, "w")
    create_tally_dataset(file, mcdc, data)
    file.close()


def replace_dataset(file, field, data):
    if field in file:
        del file[field]
    file.create_dataset(field, data=data)


def recombine_tallies():
    """Combine the tally output into a single file"""
    import h5py
    from mpi4py import MPI
    from mcdc.object_.tally import decode_score_type

    if MPI.COMM_WORLD.Get_rank() > 0:
        return

    # Get simulation and settings
    from mcdc.object_.simulation import simulation

    settings = simulation.settings
    if not settings.use_census_based_tally:
        print("Census-based tally is not used, nothing to recombine.")

    # Settings parameters
    base_name = settings.output_name
    N_census = settings.N_census
    N_batch = settings.N_batch
    frequency = settings.census_tally_frequency
    Nt = frequency * (N_census - 1)

    # Append the tally dataset structure to the main output
    main_file = h5py.File(f"{base_name}.h5", "a")
    reference_file = h5py.File(f"{base_name}-batch_0-census_0.h5", "r")
    tally_group = main_file.create_group("tallies")
    for tally in simulation.tallies:
        name = f"tallies/{tally.name}"
        reference_file.copy(name, tally_group)
    reference_file.close()

    # Set the time grid
    time_grid = np.zeros(Nt + 1)
    for i in range(N_census - 1):
        start = settings.census_time[i - 1] if i > 0 else 0.0
        end = settings.census_time[i]
        new_grid = np.linspace(start, end, frequency + 1)
        offset = i * frequency + 1
        time_grid[offset : offset + frequency] = new_grid[1:]
    for tally in simulation.tallies:
        name = f"tallies/{tally.name}/grid/time"
        replace_dataset(main_file, name, time_grid)

    # Combine the tallies
    for tally in simulation.tallies:
        # The combined shape
        shape = tally.bin_shape
        shape[3] = Nt

        for score in tally.scores:
            score_name = f"tallies/{tally.name}/{decode_score_type(score, True)}"

            mean = np.zeros(shape)
            sdev = np.zeros(shape)

            # Selective squeeze
            axes_to_squeeze = [x for x, size in enumerate(shape) if size == 1 and x > 3]
            mean = np.squeeze(mean, axis=tuple(axes_to_squeeze))
            sdev = np.squeeze(sdev, axis=tuple(axes_to_squeeze))

            for i_census in range(N_census - 1):
                # Accumulate sum and sum of square
                for i_batch in range(N_batch):
                    file_name = f"{base_name}-batch_{i_batch}-census_{i_census}.h5"
                    file = h5py.File(file_name, "r")
                    offset = i_census * frequency

                    score = file[f"{score_name}/mean"][()]
                    mean[:, :, :, offset : offset + frequency] += score
                    sdev[:, :, :, offset : offset + frequency] += score * score

                    file.close()

            # Squeeze
            mean = np.squeeze(mean)
            sdev = np.squeeze(sdev)

            # Compute statistics
            mean /= N_batch
            sdev = np.sqrt((sdev / N_batch - np.square(mean)) / (N_batch - 1))

            replace_dataset(main_file, f"{score_name}/mean", mean)
            replace_dataset(main_file, f"{score_name}/sdev", sdev)

    main_file.close()
