import h5py
import matplotlib.pyplot as plt
import sys

isotope = sys.argv[1]

with h5py.File(f"../../proton_generated_lib/{isotope}-293.6K.h5") as f:
    print(f'atomic number = {f["atomic_number"][()]}')
    print(f'atomic weight ratio = {f["atomic_weight_ratio"][()]}')
    print(f'fissionable = {f["fissionable"][()]}')
    print(f'nuclide name = {f["nuclide_name"][()]}')

    elastic_xs = f["proton_reactions"]["elastic_scattering"]["MT-002"]["xs"][()]
    inelastic_xs = f["proton_reactions"]["inelastic_scattering"]["MT-005"]["xs"][()]

    plt.plot(elastic_xs, label="elastic")
    plt.plot(inelastic_xs, label="inelastic")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Incident Energy (MeV)")
    plt.ylabel("Cross Section")
    plt.savefig(f"{isotope}_xs.png")
