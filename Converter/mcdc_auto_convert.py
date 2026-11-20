import sys
import h5py

import h5_to_vtr
import h5_to_vtp

# ======================================================================================
# Conversion
# ======================================================================================

xyz = {"x", "y", "z"}


# looks at each tally's grid and hands it to whichever writer fits
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
            if "grid" not in tally:
                print(f"\n  [{name}] no grid found, skipping")
                continue
            grid_keys = set(tally["grid"].keys())

            if any(d in grid_keys for d in xyz):
                total += h5_to_vtr.write_vtr(h5_file, tally, name, output_prefix)
            elif "face" in grid_keys:
                total += h5_to_vtp.write_vtp(
                    h5_file, tally, name, output_prefix, bounds
                )
            else:
                print(
                    f"\n  [{name}] don't recognize this tally shape -- grid has: {sorted(grid_keys)}"
                )

    print(f"\ndone -- {total} file(s) written")


# ======================================================================================
# Command line
# ======================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: python3 mcdc_auto_convert.py output.h5 [tally_name] [--bounds x0 x1 y0 y1 z0 z1]"
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
