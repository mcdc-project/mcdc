import sys
import extract_geometry
import mcdc_auto_convert

# ======================================================================================
# Command line
# ======================================================================================

# usage:
#   python3 mcdc_convert.py --input file.py --output file.h5
#   python3 mcdc_convert.py --input file.py
#   python3 mcdc_convert.py --output file.h5


# runs the geometry converter and/or the tally converter, so you don't have to call them separately
def main():
    args = sys.argv[1:]
    input_file = None
    output_file = None

    if "--input" in args:
        input_file = args[args.index("--input") + 1]
    if "--output" in args:
        output_file = args[args.index("--output") + 1]

    if not input_file and not output_file:
        print("usage: python3 mcdc_convert.py --input file.py --output file.h5")
        print("need --input, --output, or both")
        sys.exit(1)

    if input_file:
        print(f"--- geometry from {input_file} ---")
        extract_geometry.extract(input_file)
        print()

    if output_file:
        print(f"--- tallies from {output_file} ---")
        mcdc_auto_convert.convert(output_file)


if __name__ == "__main__":
    main()
