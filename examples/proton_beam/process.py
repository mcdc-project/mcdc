import h5py
import numpy as np
import matplotlib.pyplot as plt

energy = 1  # MeV

# with h5py.File(f"output_{energy}mev.h5") as f:
with h5py.File(f"output.h5") as f:
    isotope_edep = list(f["tallies"].keys())[0]

    edep = f["tallies"][f"{isotope_edep}"]["energy_deposition"]["mean"][()]
    xgrid = f["tallies"][f"{isotope_edep}"]["grid"]["x"][()]
    # print(f'xgrid = {xgrid}')

    normalized_edep = np.zeros_like(edep)
    centers = np.zeros_like(edep)
    for i in range(len(xgrid) - 1):
        width = xgrid[i + 1] - xgrid[i]
        centers[i] = xgrid[i] + width / 2
        normalized_edep[i] = edep[i] / (energy * 1e6) / width

    normalized_edep = np.array(normalized_edep)
    index_of_depth_at_max = np.argmax(normalized_edep)

    print(rf"peak location: {xgrid[index_of_depth_at_max]} um")
    print(f"peak magnitude = {np.max(normalized_edep)}")


# TODO: add automatic range calculations based on PSTAR data
range = 0.001645

plt.plot(centers * 1e4, normalized_edep, label="edep tally")
plt.vlines(
    range * 1e4,
    0,
    np.max(normalized_edep),
    linestyle="--",
    label="theoretical Bragg peak for 1 MeV protons",
    color="red",
)
plt.title(f"Energy Deposition of {energy} MeV Protons in a Slab of Si-28")
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("MeV/cm")
plt.ylim(0, 1300)
plt.legend()
plt.savefig(f"Si-28_edep_{energy}MeV.png")
# plt.show()
