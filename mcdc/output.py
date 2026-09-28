import importlib.metadata
from pathlib import Path

import h5py
import numpy as np

####

import mcdc.mcdc_get as mcdc_get
import mcdc.print_ as print_module

from mcdc.constant import (
    MESH_UNIFORM,
    MESH_STRUCTURED,
)

# ======================================================================================
# Main output
# ======================================================================================


def generate_output(mcdc, data, simulationPy, no_tally_output=False):
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
    create_object_dataset(file, "settings", simulationPy.settings)

    # No need to output tally if time census-based tally is used
    if mcdc["settings"]["use_census_based_tally"]:
        file.close()
        return

    # Tallies
    if not no_tally_output:
        create_tally_dataset(file, mcdc, data)

    # Eigenvalues
    if not no_tally_output and mcdc["settings"]["neutron_eigenvalue_mode"]:
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

    if not mcdc["mpi_master"]:
        return

    base_name = mcdc["settings"]["output_name"]

    main_output = h5py.File(f"{base_name}.h5", "a")
    create_runtime_dataset(main_output, mcdc)
    main_output.close()


def create_runtime_dataset(file, mcdc):
    for name in [
        "total",
        "preparation",
        "simulation",
        "output",
        "bank_management",
    ]:
        file.create_dataset(f"runtime/{name}", data=np.array([mcdc["runtime_" + name]]))


def generate_performance_output(simulation):
    """Append performance metrics to the standard output on the master rank."""

    if not simulation["mpi_master"]:
        return

    settings = simulation["settings"]

    # Include inactive eigenvalue cycles in the total transport workload.
    N_repeat = (
        settings["N_cycle"]
        if settings["neutron_eigenvalue_mode"]
        else settings["N_batch"]
    )

    N_history = int(settings["N_particle"]) * int(N_repeat)

    with h5py.File(f"{settings['output_name']}.h5", "a") as file:
        group = file.create_group("performance")
        group.create_dataset("runtime", data=simulation["runtime_total"])
        group.create_dataset("N_history", data=N_history)
        group.create_dataset("N_rank", data=simulation["mpi_size"])
        group.create_dataset(
            "effective_variance", data=simulation["effective_variance"]
        )


def read_census_score(simulation, data, tally, score, batch, census):
    """Read one flattened census score; absent batch contributions are zero."""
    from mcdc.object_.tally import decode_score_type

    path = census_based_tally_file_name(
        simulation["settings"]["output_name"], batch, census
    )
    if not path.is_file():
        return np.zeros(tally["bin_length"] // tally["scores_length"])
    score_type = mcdc_get.tally.scores(score, tally, data)
    score_name = decode_score_type(score_type, lower_case=True)
    with h5py.File(path, "r") as file:
        return np.asarray(
            file[f"tallies/{tally['name']}/{score_name}/mean"][()], dtype=np.float64
        ).reshape(-1)


# ======================================================================================
# Tally
# ======================================================================================


def create_tally_dataset(file, mcdc, data):
    from mcdc.constant import TALLY_TRACKLENGTH, TALLY_COLLISION
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
        mesh_filtered_tally = None
        if tally["sub_type"] == TALLY_TRACKLENGTH:
            mesh_filtered_tally = mcdc["tracklength_tallies"][tally["sub_ID"]]
        elif tally["sub_type"] == TALLY_COLLISION:
            mesh_filtered_tally = mcdc["collision_tallies"][tally["sub_ID"]]

        if mesh_filtered_tally is not None and mesh_filtered_tally["mesh_filtered"]:
            mesh_base = mcdc["meshes"][mesh_filtered_tally["mesh_filter_ID"]]
            mesh_type = mesh_base["sub_type"]
            mesh_ID = mesh_base["sub_ID"]
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
        start_mean = tally["bin_mean_offset"]
        start_sdev = tally["bin_sum_squared_deviations_offset"]
        mean = data[start_mean : start_mean + N_bin]
        sdev = data[start_sdev : start_sdev + N_bin]
        shape = tuple([int(x) for x in mcdc_get.tally.bin_shape_all(tally, data)])
        mean = mean.reshape(shape)
        sdev = sdev.reshape(shape)

        # Roll tally so that score is in the front
        roll_reference = 4
        if mesh_filtered_tally is not None and mesh_filtered_tally["mesh_filtered"]:
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
    file_name = census_based_tally_file_name(base_name, idx_batch, idx_census)
    file = h5py.File(file_name, "w")
    create_tally_dataset(file, mcdc, data)
    file.close()


def replace_dataset(file, field, data):
    if field in file:
        del file[field]
    file.create_dataset(field, data=data)


def census_based_tally_file_name(base_name, idx_batch, idx_census):
    """Return the intermediate tally path for one batch and time census."""
    return Path(f"{base_name}-batch_{idx_batch}-census_{idx_census}.h5")


def clear_census_based_tally_files(settings):
    """Remove intermediate tallies that could otherwise leak across runs."""
    for idx_batch in range(settings.N_batch):
        for idx_census in range(settings.N_census):
            census_based_tally_file_name(
                settings.output_name, idx_batch, idx_census
            ).unlink(missing_ok=True)


def recombine_tallies(simulationPy, simulation):
    """Combine census-based tally files into the main output file.

    Parameters
    ----------
    simulationPy : mcdc.Simulation
        Python simulation object containing tally definitions and settings.
    simulation : numpy.void
        Packed runtime simulation state. Recombination is performed only by
        its designated MPI master rank.
    """
    from mcdc.object_.tally import decode_score_type

    if not simulation["mpi_master"]:
        return

    settings = simulationPy.settings
    if not settings.use_census_based_tally:
        print("Census-based tally is not used, nothing to recombine.")
        return

    # Settings parameters
    base_name = settings.output_name
    N_census = settings.N_census
    N_batch = settings.N_batch
    frequency = settings.census_tally_frequency
    Nt = frequency * (N_census - 1)

    # Append the tally dataset structure to the main output
    with h5py.File(f"{base_name}.h5", "a") as main_file:
        if "tallies" in main_file:
            del main_file["tallies"]
        tally_group = main_file.create_group("tallies")

        reference_path = census_based_tally_file_name(base_name, 0, 0)
        with h5py.File(reference_path, "r") as reference_file:
            for tally in simulationPy.tallies:
                name = f"tallies/{tally.name}"
                reference_file.copy(name, tally_group)

        # Set the time grid
        time_grid = np.zeros(Nt + 1)
        for i_census in range(N_census - 1):
            start = settings.census_time[i_census - 1] if i_census > 0 else 0.0
            end = settings.census_time[i_census]
            new_grid = np.linspace(start, end, frequency + 1)
            offset = i_census * frequency + 1
            time_grid[offset : offset + frequency] = new_grid[1:]
        for tally in simulationPy.tallies:
            name = f"tallies/{tally.name}/grid/time"
            replace_dataset(main_file, name, time_grid)

        # Combine the tallies
        for tally in simulationPy.tallies:
            census_shape = tuple(int(size) for size in tally.bin_shape[:-1])
            combined_shape = list(census_shape)
            combined_shape[3] = Nt
            combined_shape = tuple(combined_shape)

            for score in tally.scores:
                score_name = f"tallies/{tally.name}/{decode_score_type(score, True)}"
                mean = np.zeros(combined_shape)
                sum_squared_deviations = np.zeros(combined_shape)

                for i_census in range(N_census - 1):
                    offset = i_census * frequency
                    time_slice = [slice(None)] * len(combined_shape)
                    time_slice[3] = slice(offset, offset + frequency)
                    time_slice = tuple(time_slice)

                    for i_batch in range(N_batch):
                        file_name = census_based_tally_file_name(
                            base_name, i_batch, i_census
                        )

                        # Empty particle banks end a batch before later census files are
                        # written. Those absent contributions are physically zero.
                        if file_name.is_file():
                            with h5py.File(file_name, "r") as file:
                                score_data = np.asarray(
                                    file[f"{score_name}/mean"][()]
                                ).reshape(census_shape)
                        else:
                            score_data = np.zeros(census_shape)

                        # Match the Welford update used in tally closeout.
                        # Missing files must still participate as zero samples.
                        delta = score_data - mean[time_slice]
                        mean[time_slice] += delta / (i_batch + 1)
                        sum_squared_deviations[time_slice] += delta * (
                            score_data - mean[time_slice]
                        )

                if N_batch > 1:
                    variance = sum_squared_deviations / (N_batch - 1) / N_batch
                    sdev = np.sqrt(np.maximum(variance, 0.0))
                else:
                    sdev = np.zeros_like(mean)

                replace_dataset(main_file, f"{score_name}/mean", np.squeeze(mean))
                replace_dataset(main_file, f"{score_name}/sdev", np.squeeze(sdev))
