import sys
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support

all_dims = ["mu", "azi", "energy", "time", "x", "y", "z"]
xyz = {"x", "y", "z"}
skip_dims = {"mu", "azi", "energy"}


# ======================================================================================
# Helper functions
# ======================================================================================


# bin edges for one axis, else [0, 1]
def get_bins(tally, dim):
    if dim in tally["grid"]:
        return tally["grid"][dim][:]
    return np.array([0.0, 1.0])


# keep axes with >1 bin
def get_active_dims(tally):
    active = []
    for dim in all_dims:
        if len(get_bins(tally, dim)) - 1 > 1:
            active.append(dim)
    return active


# ======================================================================================
# Conversion
# ======================================================================================


# write a single tally
def write_vtr(h5_file, tally, tally_name, output_prefix):
    bins = {d: get_bins(tally, d) for d in all_dims}
    active = get_active_dims(tally)

    score_list = [k for k in tally.keys() if k != "grid"]
    scores_data = [s for s in score_list if len(tally[f"{s}/mean"].shape) > 0]

    if not scores_data:
        print(f"\n  no spatial scores found in '{tally_name}'")
        return 0

    ref_data = tally[f"{scores_data[0]}/mean"][:]
    print(f"\n  reading tally '{tally_name}', scores: {scores_data}")

    # discard non-spatial
    def nonspatial(data):
        shift = 0
        for i, dim in enumerate(active):
            if dim in skip_dims:
                data = data.sum(axis=i - shift)
                shift += 1
        return data

    ref_data = nonspatial(ref_data)
    remaining = [d for d in active if d not in skip_dims]
    time = "time" in remaining
    n_steps = len(bins["time"]) - 1 if time else 1
    spatial_dims = [d for d in remaining if d in xyz]

    files_written = 0
    for step in range(n_steps):
        ref_slice = ref_data[step] if time else ref_data
        shape = ref_slice.shape

        nx, ny, nz = 1, 1, 1
        x_edges = np.array([0.0, 1.0])
        y_edges = np.array([0.0, 1.0])
        z_edges = np.array([0.0, 1.0])

        for j, dim in enumerate(spatial_dims):
            if j >= len(shape):
                continue
            if dim == "x":
                nx = shape[j]
                x_edges = bins["x"][: nx + 1]
            elif dim == "y":
                ny = shape[j]
                y_edges = bins["y"][: ny + 1]
            elif dim == "z":
                nz = shape[j]
                z_edges = bins["z"][: nz + 1]

        total_cells = nx * ny * nz

        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(nx + 1, ny + 1, nz + 1)
        grid.SetXCoordinates(
            numpy_support.numpy_to_vtk(x_edges.astype(np.float64), deep=True)
        )
        grid.SetYCoordinates(
            numpy_support.numpy_to_vtk(y_edges.astype(np.float64), deep=True)
        )
        grid.SetZCoordinates(
            numpy_support.numpy_to_vtk(z_edges.astype(np.float64), deep=True)
        )

        for score in scores_data:
            data = nonspatial(tally[f"{score}/mean"][:])
            data = data[step] if time else data
            flat = data.flatten(order="F")  # x varies fastest

            if len(flat) == total_cells:
                # one number per cell
                arr = numpy_support.numpy_to_vtk(flat, deep=True)
                arr.SetName(score)
                grid.GetCellData().AddArray(arr)
            elif len(flat) == total_cells * 3:
                # 3 numbers per cell
                vec = flat.reshape(total_cells, 3, order="F")
                arr = numpy_support.numpy_to_vtk(vec, deep=True)
                arr.SetName(score)
                arr.SetNumberOfComponents(3)
                grid.GetCellData().AddArray(arr)
                grid.GetCellData().SetVectors(arr)
            else:
                print(f"  '{score}' size doesn't match the grid at step {step}")
                continue

        output_name = f"{output_prefix}_{tally_name}_t{step:03d}.vtr"
        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName(output_name)
        writer.SetInputData(grid)
        writer.Write()

        files_written += 1
        print(f"  wrote {output_name}  ({nx}x{ny}x{nz} cells, scores: {scores_data})")

    return files_written


# runs write_vtr on every matching tally in the file
def convert(h5_file, tally_name=None):
    output_prefix = h5_file.replace(".h5", "")

    with h5py.File(h5_file, "r") as f:
        all_tallies = list(f["tallies"].keys())
        print(f"opening {h5_file}, found tallies: {all_tallies}")

        tally_convert = [tally_name] if tally_name else all_tallies
        total = 0

        # convert each tally
        for name in tally_convert:
            if name not in all_tallies:
                print(f"\n  can't find tally '{name}'")
                continue
            tally = f[f"tallies/{name}"]
            scores = [k for k in tally.keys() if k != "grid"]
            if "grid" not in tally or not scores:
                print(f"\n  skipping '{name}', no convertable data")
                continue
            total += write_vtr(h5_file, tally, name, output_prefix)

    print(f"\ndone --> {total} .vtr file(s) written")


# ======================================================================================
# Command line
# ======================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 h5_to_vtr.py output.h5 [tally_name]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
