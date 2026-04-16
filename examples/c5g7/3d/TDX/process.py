import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
from matplotlib import colors

# =============================================================================
# Plot results
# =============================================================================

# Get results
with h5py.File("output.h5", "r") as f:
    fis_avg = f["tallies/mesh_tally_0/fission/mean"][:]
    fis_sd = f["tallies/mesh_tally_0/fission/sdev"][:]
    t = f["tallies/mesh_tally_0/grid/t"][:]
t_mid = 0.5 * (t[:-1] + t[1:])
dt = t[1:] - t[:-1]

# Get reference
with h5py.File("reference.h5", "r") as f:
    fis_ref = f["tally/fission/mean"][:]
    fis_ref_sd = f["tally/fission/sdev"][:]

# Normalize
norm = fis_ref[0] / dt[0]
fis_avg /= norm * dt
fis_sd /= norm * dt
fis_ref /= norm * dt
fis_ref_sd /= norm * dt

# Plot
plt.plot(t_mid, fis_ref, "-m", fillstyle="none", label="Ref. (MC)")
plt.fill_between(
    t_mid, fis_ref - fis_ref_sd, fis_ref + fis_ref_sd, alpha=0.2, color="m"
)
plt.plot(t_mid, fis_avg, "ok", fillstyle="none", label="MC")
plt.fill_between(t_mid, fis_avg - fis_sd, fis_avg + fis_sd, alpha=0.2, color="k")
plt.yscale("log")
plt.ylabel("Normalized fission rate")
plt.xlabel("time [s]")
plt.axvspan(0, 5, facecolor="gray", alpha=0.2)
plt.axvspan(5, 10, facecolor="green", alpha=0.2)
plt.axvspan(10, 15, facecolor="red", alpha=0.2)
plt.axvspan(15, 20, facecolor="blue", alpha=0.2)
plt.annotate(
    "Phase 1",
    (2.5, 0.3),
    color="black",
    ha="center",
    va="center",
    backgroundcolor="white",
)
plt.annotate(
    "Phase 2",
    (7.5, 0.3),
    color="black",
    ha="center",
    va="center",
    backgroundcolor="white",
)
plt.annotate(
    "Phase 3",
    (12.5, 0.3),
    color="black",
    ha="center",
    va="center",
    backgroundcolor="white",
)
plt.annotate(
    "Phase 4",
    (17.5, 4.0),
    color="black",
    ha="center",
    va="center",
    backgroundcolor="white",
)
plt.xlim([0.0, 20.0])
plt.ylim([0.09, 200.0])
plt.grid(which="both")
plt.legend()
plt.show()
