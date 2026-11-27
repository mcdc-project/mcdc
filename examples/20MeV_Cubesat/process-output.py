import h5py
import numpy as np
import matplotlib.pyplot as plt

# Neutron current entering each sensitive volume (SV), per source particle,
# summed over the six SV faces and all angle bins
SVS = ["obc", "eps", "adcs", "comms"]
FACES = ["x0", "x1", "y0", "y1", "z0", "z1"]

with h5py.File("cubesat_20MeV.h5", "r") as output:
    energy = output["tallies/obc_neutron_source_x0/grid/energy"][:]
    current = {}
    for sv in SVS:
        mean = 0.0
        var = 0.0
        for face in FACES:
            tally = output[f"tallies/{sv}_neutron_source_{face}/current-in"]
            mean = mean + tally["mean"][:].sum(axis=(0, 1))
            var = var + (tally["sdev"][:] ** 2).sum(axis=(0, 1))
        current[sv] = (mean, np.sqrt(var))

# Per unit energy
de = energy[1:] - energy[:-1]
energy_mid = np.sqrt(energy[1:] * energy[:-1])

figure, axis = plt.subplots()
for sv in SVS:
    mean, sdev = current[sv]
    mean = mean / de
    sdev = sdev / de
    axis.step(energy_mid, mean, where="mid", label=sv.upper())
    axis.fill_between(energy_mid, mean - sdev, mean + sdev, step="mid", alpha=0.25)
axis.set_xscale("log")
axis.set_yscale("log")
# Keep the low-count sub-thermal bins from stretching the axis
smallest = min((current[sv][0] / de)[current[sv][0] > 0].min() for sv in SVS)
axis.set_ylim(bottom=0.1 * smallest)
axis.set_xlabel("Energy [eV]")
axis.set_ylabel("Incoming current [1/eV per source neutron]")
axis.grid()
axis.legend()
figure.tight_layout()
figure.savefig("cubesat_20MeV_sv_spectra.png", dpi=150)

for sv in SVS:
    mean, sdev = current[sv]
    print(
        f"{sv.upper():5s} incoming current: {mean.sum():.4e} +/- {np.sqrt((sdev**2).sum()):.1e}"
    )
