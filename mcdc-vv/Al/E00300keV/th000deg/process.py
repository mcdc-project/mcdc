import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_LIBRARY_NAME = "electron-vv-data"
PROCESS_DATA_LIBRARY_DIR = None
PROCESS_DATA_LIBRARY_ENV = "MCDC_VV_PROCESS_DATA_LIBRARY_DIR"


def _find_repo_root(start_path):
    path = Path(start_path).resolve()
    for parent in (path.parent, *path.parents):
        if parent.name == "MCDC-VV-electron":
            return parent
    raise FileNotFoundError("Could not locate the MCDC-VV-electron repository root.")


def resolve_process_data_library_dir(process_path=__file__):
    repo_root = _find_repo_root(process_path)

    if PROCESS_DATA_LIBRARY_DIR is None:
        return str((repo_root / DEFAULT_DATA_LIBRARY_NAME).resolve())

    candidate = Path(PROCESS_DATA_LIBRARY_DIR).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return str(candidate.resolve())


def _format_energy_dir(energy_keV):
    return f"E{int(round(float(energy_keV))):05d}keV"


def _format_angle_dir(angle_deg):
    return f"th{int(round(float(angle_deg))):03d}deg"


def resolve_case_directory(problem, material, energy_keV, angle_deg, process_path=__file__):
    repo_root = _find_repo_root(process_path)
    case_dir = (
        repo_root
        / "verification"
        / "benchmark"
        / "continuous_energy"
        / problem
        / material
        / _format_energy_dir(energy_keV)
        / _format_angle_dir(angle_deg)
    )
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    return case_dir


def _find_input_value(name, text):
    match = re.search(rf"^\s*{name}\s*=\s*(.+?)\s*(#.*)?$", text, re.M)
    if not match:
        raise ValueError(f"{name} not found in input.py")
    return match.group(1).strip()


def load_case_metadata(case_dir):
    input_path = Path(case_dir) / "input.py"
    if not input_path.exists():
        raise FileNotFoundError(f"input.py not found in {case_dir}")

    text = input_path.read_text(encoding="utf-8")
    return {
        "material_symbol": _find_input_value("MATERIAL_SYMBOL", text).strip("\"' "),
        "energy_eV": float(eval(_find_input_value("ENERGY", text), {}, {})),
        "angle_deg": float(eval(_find_input_value("ANGLE", text), {}, {})),
        "csda_range": float(eval(_find_input_value("CSDA_RANGE", text), {}, {})),
        "rho_g_cm3": float(eval(_find_input_value("RHO_G_CM3", text), {}, {})),
        "n_particles": float(eval(_find_input_value("N_PARTICLES", text), {}, {})),
    }


def resolve_h5_output(case_dir):
    answer_path = Path(case_dir) / "answer.h5"
    if answer_path.exists():
        return answer_path

    candidates = [
        path
        for path in Path(case_dir).iterdir()
        if path.is_file() and path.suffix == ".h5"
    ]
    if not candidates:
        raise FileNotFoundError(f"No HDF5 output found in {case_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_case_input(case_dir, process_path=__file__):
    data_library_dir = resolve_process_data_library_dir(process_path)
    env = os.environ.copy()
    env[PROCESS_DATA_LIBRARY_ENV] = data_library_dir
    env["MCDC_LIB"] = data_library_dir

    print(f"Running input: {Path(case_dir) / 'input.py'}")
    print(f"Using process data library: {data_library_dir}")
    subprocess.run([sys.executable, "input.py"], cwd=case_dir, check=True, env=env)


def process_case(case_dir):
    import h5py
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    case_dir = Path(case_dir).resolve()
    ref_path = case_dir / "reference.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"reference.npz not found in {case_dir}")

    h5_path = resolve_h5_output(case_dir)
    if h5_path.name != "answer.h5":
        print(f"Using HDF5 output: {h5_path.name}")

    metadata = load_case_metadata(case_dir)

    ref = np.load(ref_path)
    theo_fmr = ref["fmr_theo_tiger"].astype(float)
    theo_edep = ref["edep_theo_tiger"].astype(float)

    exp_fmr_A = exp_edep_A = None
    exp_fmr_B = exp_edep_B = None
    exp_fmr = exp_edep = None

    if "fmr_exp_lw_A" in ref:
        exp_fmr_A = ref["fmr_exp_lw_A"].astype(float)
        exp_edep_A = ref["edep_exp_lw_A"].astype(float)

    if "fmr_exp_lw_B" in ref:
        exp_fmr_B = ref["fmr_exp_lw_B"].astype(float)
        exp_edep_B = ref["edep_exp_lw_B"].astype(float)
    else:
        exp_fmr = ref["fmr_exp_lw"].astype(float)
        exp_edep = ref["edep_exp_lw"].astype(float)

    with h5py.File(h5_path, "r") as file:
        z = file["tallies/edep/grid/z"][:].astype(float)
        dz = z[1:] - z[:-1]
        edep = np.atleast_1d(file["tallies/edep/energy_deposition/mean"][()]).astype(
            float
        )

    L = metadata["csda_range"] / metadata["rho_g_cm3"]
    z_centers = 0.5 * (z[:-1] + z[1:])
    mcdc_fmr = z_centers / L
    edep_mcdc = edep / metadata["rho_g_cm3"] / dz / 1e6

    plt.figure(figsize=(14, 9), constrained_layout=True)
    mpl.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 29,
            "axes.labelsize": 25,
            "legend.fontsize": 22,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
        }
    )

    plt.minorticks_on()
    plt.grid(
        True, which="major", linestyle="-", linewidth=0.8, color="#3B3A3AFF", alpha=0.5
    )
    plt.grid(
        True, which="minor", linestyle=":", linewidth=0.6, color="#8A8686", alpha=0.7
    )

    plt.plot(
        mcdc_fmr,
        edep_mcdc,
        label="MCDC Simulation",
        marker="o",
        markersize=9,
        linewidth=3.0,
    )
    plt.plot(
        theo_fmr,
        theo_edep,
        label="TIGER Theoretical",
        marker="s",
        markersize=9,
        linewidth=3.0,
    )

    if exp_fmr_A is not None:
        plt.plot(
            exp_fmr_A,
            exp_edep_A,
            label="Lockwood Experimental A",
            marker="D",
            markersize=9,
            linewidth=3.0,
        )
    if exp_fmr_B is not None:
        plt.plot(
            exp_fmr_B,
            exp_edep_B,
            label="Lockwood Experimental B",
            marker="*",
            markersize=9,
            linewidth=3.0,
        )
    if exp_fmr is not None:
        plt.plot(
            exp_fmr,
            exp_edep,
            label="Lockwood Experimental",
            marker="D",
            markersize=9,
            linewidth=3.0,
        )

    energy_MeV = metadata["energy_eV"] / 1e6
    angle_deg = metadata["angle_deg"]

    plt.xlim(0, 1)
    plt.xlabel("Fraction of Mean Range", labelpad=15)
    plt.ylabel("Energy Deposition (MeV/g/cm²)", labelpad=15)
    plt.title(
        f"Energy Deposition of {energy_MeV:g} MeV Electrons in {metadata['material_symbol']} at {angle_deg:g}° Incidence",
        pad=20,
    )
    plt.legend()

    out_dir = case_dir.parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    out_name = (
        f"fig_{metadata['material_symbol']}_{energy_MeV:g}MeV_"
        f"th{int(round(angle_deg))}_{metadata['n_particles']:.0e}".replace("+", "")
        + ".png"
    )
    out_path = out_dir / out_name

    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"Saved: {out_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run and/or post-process VV electron benchmark cases."
    )
    parser.add_argument("-p", "--problem", help="Problem name, e.g. lockwood")
    parser.add_argument("-m", "--material", help="Material name, e.g. Al")
    parser.add_argument(
        "-e", "--energy", type=float, help="Incident energy in keV, e.g. 1000"
    )
    parser.add_argument(
        "-a", "--angle", type=float, help="Incident angle in degrees, e.g. 30"
    )
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Run the selected input but skip post-processing.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip the input run and only post-process the selected case.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved case and data paths without running anything.",
    )
    return parser


def main(argv=None, process_path=__file__):
    parser = build_parser()
    args = parser.parse_args(argv)

    has_selector = any(
        value is not None for value in (args.problem, args.material, args.energy, args.angle)
    )
    if has_selector and not all(
        value is not None for value in (args.problem, args.material, args.energy, args.angle)
    ):
        parser.error("Provide -p, -m, -e, and -a together.")
    if args.input_only and args.process_only:
        parser.error("--input-only and --process-only cannot be used together.")
    if not has_selector and args.input_only:
        parser.error("--input-only requires -p, -m, -e, and -a.")

    if has_selector:
        case_dir = resolve_case_directory(
            args.problem, args.material, args.energy, args.angle, process_path
        )
        data_library_dir = resolve_process_data_library_dir(process_path)

        if args.dry_run:
            print(f"Resolved case directory: {case_dir}")
            print(f"Resolved process data library: {data_library_dir}")
            return

        if not args.process_only:
            run_case_input(case_dir, process_path)
        if not args.input_only:
            process_case(case_dir)
        return

    if args.dry_run:
        print(f"Resolved case directory: {Path.cwd().resolve()}")
        print(f"Resolved process data library: {resolve_process_data_library_dir(process_path)}")
        return

    process_case(Path.cwd())


if __name__ == "__main__":
    main()
