import h5py

tally_name = "Spherical fission detector"

with h5py.File("output.h5", "r") as file:
    mean = file[f"tallies/{tally_name}/fission/mean"][()]
    sdev = file[f"tallies/{tally_name}/fission/sdev"][()]

print(f"\n{tally_name}: {mean} +/- {sdev}\n")
