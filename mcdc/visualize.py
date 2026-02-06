from numba import njit
import numpy as np
from mcdc.main import preparation
from mcdc.transport.geometry.interface import locate_particle

_visualize_cache = None


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
        Particle array of size (1,) used for particle lookup.
    mcdc : structured array
        MCDC simulation data
    data : structured array
        Additional simulation data
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
        # Create particle array and material IDs grid
        particle_arr = np.zeros(1, dtype=type_.particle)
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
                particle_arr,
                mcdc,
                data,
            )
            material_ids[i, :] = row_materials

            # Update progress bar
            percent = (i + 1) / pixels[0]
            if int(percent * 100) > last_progress:
                last_progress = int(percent * 100)
                sys.stdout.write(
                    "\r Visualizing: [%-28s] %d%%"
                    % ("=" * int(percent * 28), percent * 100)
                )
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
