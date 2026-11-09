import sys
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support

all_dims = ["mu", "azi", "energy", "time", "face", "u", "v"]
skip_dims = {"mu", "azi", "energy"}

axis_idx = {"x": 0, "y": 1, "z": 2}
span_axes = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}


# ======================================================================================
# Helper functions
# ======================================================================================


# splits face label
def parse_face(label):
    axis = label[0]
    side = label[1:]
    if axis not in span_axes or side not in ("min", "max"):
        return None
    s1, s2 = span_axes[axis]
    return axis, side, s1, s2


def get_bins(tally, dim):
    if dim in tally["grid"]:
        return tally["grid"][dim][:]
    return np.array([0.0, 1.0])


# mu/azi/energy only kept if bins >1
def get_active_dims(tally):
    active = []
    for dim in all_dims:
        if dim in skip_dims:
            if len(get_bins(tally, dim)) - 1 > 1:
                active.append(dim)
        else:
            active.append(dim)
    return active


# pulls the real box bounds straight from file
def bounds_from_grid(tally):
    grid = tally["grid"]
    if "face" not in grid or "u" not in grid or "v" not in grid:
        return None, None
    u_raw = grid["u"][:]
    v_raw = grid["v"][:]
    if u_raw.ndim != 2 or v_raw.ndim != 2:
        return None, None

    faces = [f.decode() if isinstance(f, bytes) else f for f in grid["face"][:]]
    bounds = {}
    per_face_edges = {}
    for i, label in enumerate(faces):
        parsed = parse_face(label)
        if parsed is None:
            return None, None
        axis, side, a1, a2 = parsed
        u_edges, v_edges = u_raw[i], v_raw[i]
        bounds[a1 + "0"] = float(u_edges[0])
        bounds[a1 + "1"] = float(u_edges[-1])
        bounds[a2 + "0"] = float(v_edges[0])
        bounds[a2 + "1"] = float(v_edges[-1])
        per_face_edges[i] = (u_edges, v_edges)

    if not all(k in bounds for k in ("x0", "x1", "y0", "y1", "z0", "z1")):
        return None, None
    box = (
        bounds["x0"],
        bounds["x1"],
        bounds["y0"],
        bounds["y1"],
        bounds["z0"],
        bounds["z1"],
    )
    return box, per_face_edges


# build face mesh
def build_face_mesh(
    bounds, face_label, vals_2d, score_name, u_edges=None, v_edges=None
):
    x0, x1, y0, y1, z0, z1 = bounds
    box_min = [x0, y0, z0]
    box_max = [x1, y1, z1]
    fixed_axis, fixed_val_key, span1, span2 = parse_face(face_label)
    fi, s1, s2 = axis_idx[fixed_axis], axis_idx[span1], axis_idx[span2]
    fixed_val = box_min[fi] if fixed_val_key == "min" else box_max[fi]
    n_u, n_v = vals_2d.shape

    span1_edges = (
        u_edges
        if u_edges is not None
        else np.linspace(box_min[s1], box_max[s1], n_u + 1)
    )
    span2_edges = (
        v_edges
        if v_edges is not None
        else np.linspace(box_min[s2], box_max[s2], n_v + 1)
    )

    pts = vtk.vtkPoints()
    for v2 in span2_edges:
        for v1 in span1_edges:
            pt = [0.0, 0.0, 0.0]
            pt[fi] = fixed_val
            pt[s1] = v1
            pt[s2] = v2
            pts.InsertNextPoint(pt)

    row_len = n_u + 1
    cells = vtk.vtkCellArray()
    flat_vals = []
    for j in range(n_v):
        for i in range(n_u):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, j * row_len + i)
            quad.GetPointIds().SetId(1, j * row_len + i + 1)
            quad.GetPointIds().SetId(2, (j + 1) * row_len + i + 1)
            quad.GetPointIds().SetId(3, (j + 1) * row_len + i)
            cells.InsertNextCell(quad)
            flat_vals.append(vals_2d[i, j])

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    arr = numpy_support.numpy_to_vtk(np.array(flat_vals), deep=True)
    arr.SetName(score_name)
    poly.GetCellData().AddArray(arr)
    return poly


# default layout with no bounds
net_layout = {0: (0, 1), 1: (2, 1), 2: (1, 1), 3: (3, 1), 4: (1, 2), 5: (1, 0)}


# laid out flat for no bounds
def build_face_mesh_flat(face_idx, vals_2d, score_name, spacing):
    col, row = net_layout.get(face_idx, (face_idx, 4))
    ox, oy = col * spacing, -row * spacing
    n_u, n_v = vals_2d.shape

    pts = vtk.vtkPoints()
    for j in range(n_v + 1):
        for i in range(n_u + 1):
            pts.InsertNextPoint(ox + i, oy + j, 0.0)

    row_len = n_u + 1
    cells = vtk.vtkCellArray()
    flat_vals = []
    for j in range(n_v):
        for i in range(n_u):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, j * row_len + i)
            quad.GetPointIds().SetId(1, j * row_len + i + 1)
            quad.GetPointIds().SetId(2, (j + 1) * row_len + i + 1)
            quad.GetPointIds().SetId(3, (j + 1) * row_len + i)
            cells.InsertNextCell(quad)
            flat_vals.append(vals_2d[i, j])

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    arr = numpy_support.numpy_to_vtk(np.array(flat_vals), deep=True)
    arr.SetName(score_name)
    poly.GetCellData().AddArray(arr)
    return poly


# ======================================================================================
# Conversion
# ======================================================================================


def write_vtp(h5_file, tally, tally_name, output_prefix, bounds_override):
    per_face_edges = None
    bounds = bounds_override
    if bounds is None:
        bounds, per_face_edges = bounds_from_grid(tally)
        if bounds is not None:
            print(f"\n  [{tally_name}] got box bounds from the h5 file")
    use_flat_layout = bounds is None
    if use_flat_layout:
        print(
            f"\n  [{tally_name}] no box coordinates found, using a flat unfolded layout instead."
        )
        print(
            f"    data is exact, only the 3D placement is schematic. pass --bounds for the real thing."
        )

    score_list = [k for k in tally.keys() if k != "grid"]
    scores_data = [s for s in score_list if len(tally[f"{s}/mean"].shape) > 0]
    if not scores_data:
        print(f"  no usable scores found in '{tally_name}', skipping")
        return 0
    score = "current-in" if "current-in" in scores_data else scores_data[0]

    active = get_active_dims(tally)
    raw = tally[f"{score}/mean"][:]
    if raw.ndim != len(active):
        print(f"  [{tally_name}] unexpected array shape (ndim={raw.ndim}), skipping")
        return 0

    # discard non-spatial
    def nonspatial(data):
        shift = 0
        for i, dim in enumerate(active):
            if dim in skip_dims:
                data = data.sum(axis=i - shift)
                shift += 1
        return data

    data = nonspatial(raw)
    remaining = [d for d in active if d not in skip_dims]
    time = "time" in remaining
    n_steps = data.shape[remaining.index("time")] if time else 1

    face_grid = tally["grid"]["face"][:]
    face_labels = [f.decode() if isinstance(f, bytes) else f for f in face_grid]
    n_faces = len(face_labels)

    print(f"  [{tally_name}] score: {score}, dims: {remaining}, faces: {n_faces}")

    files_written = 0
    for step in range(n_steps):
        step_data = data[step] if time else data
        append = vtk.vtkAppendPolyData()
        for face_idx in range(min(n_faces, step_data.shape[0])):
            if use_flat_layout:
                n_u, n_v = step_data[face_idx].shape
                spacing = max(n_u, n_v) + 2
                append.AddInputData(
                    build_face_mesh_flat(face_idx, step_data[face_idx], score, spacing)
                )
            else:
                u_e, v_e = per_face_edges[face_idx] if per_face_edges else (None, None)
                append.AddInputData(
                    build_face_mesh(
                        bounds,
                        face_labels[face_idx],
                        step_data[face_idx],
                        score,
                        u_edges=u_e,
                        v_edges=v_e,
                    )
                )
        append.Update()

        tag = "_flat" if use_flat_layout else ""
        suffix = f"_t{step:03d}" if n_steps > 1 else ""
        output_name = (
            f"{output_prefix}_{tally_name.replace(' ', '_')}_current{tag}{suffix}.vtp"
        )
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_name)
        writer.SetInputData(append.GetOutput())
        writer.Write()

        files_written += 1
        print(f"  wrote {output_name}")

    return files_written


def convert(h5_file, tally_name=None, bounds=None):
    output_prefix = h5_file.replace(".h5", "")

    with h5py.File(h5_file, "r") as f:
        all_tallies = list(f["tallies"].keys())
        print(f"opening {h5_file}, found tallies: {all_tallies}")

        tally_convert = [tally_name] if tally_name else all_tallies
        total = 0

        for name in tally_convert:
            if name not in all_tallies:
                print(f"\n  can't find tally '{name}'")
                continue
            tally = f[f"tallies/{name}"]
            if "grid" not in tally or "face" not in tally["grid"]:
                print(f"\n  skipping '{name}' -- not a surface current tally")
                continue
            total += write_vtp(h5_file, tally, name, output_prefix, bounds)

    print(f"\ndone -- {total} .vtp file(s) written")


# ======================================================================================
# Command line
# ======================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: python3 h5_to_vtp.py output.h5 [tally_name] [--bounds x0 x1 y0 y1 z0 z1]"
        )
        sys.exit(1)

    args = sys.argv[1:]
    h5_file = args[0]
    tally_name = None
    bounds = None

    if "--bounds" in args:
        idx = args.index("--bounds")
        bounds = tuple(float(v) for v in args[idx + 1 : idx + 7])
        args = args[:idx]

    if len(args) > 1:
        tally_name = args[1]

    convert(h5_file, tally_name, bounds)
