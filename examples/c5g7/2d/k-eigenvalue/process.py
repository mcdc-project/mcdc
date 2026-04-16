import numpy as np
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("output.h5", "r") as f:
    phi_avg = f["tallies/mesh_tally_0/flux/mean"][:]
    phi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]
    k = f["k_cycle"][:]
    k_avg = f["k_mean"][()]
    k_sd = f["k_sdev"][()]
    rg = f["gyration_radius"][:]
    x = f["tallies/mesh_tally_0/grid/x"][:]
    y = f["tallies/mesh_tally_0/grid/y"][:]

dx = x[1] - x[0]
x_mid = 0.5 * (x[1:] + x[:-1])
y_mid = 0.5 * (y[1:] + y[:-1])
Y, X = np.meshgrid(x_mid, y_mid)

norm = np.sum(phi_avg)
phi_avg = phi_avg / norm / dx**2
phi_sd = phi_sd / norm / dx**2

phi_fast = np.zeros_like(phi_avg[0])
phi_thermal = np.zeros_like(phi_avg[0])
phi_fast_sd = np.zeros_like(phi_avg[0])
phi_thermal_sd = np.zeros_like(phi_avg[0])

for i in range(5):
    phi_fast += phi_avg[i]
    phi_fast_sd += np.square(phi_sd[i])
phi_fast_sd = np.sqrt(phi_fast_sd) / phi_fast
for i in range(5, 7):
    phi_thermal += phi_avg[i]
    phi_thermal_sd += np.square(phi_sd[i])
phi_thermal_sd = np.sqrt(phi_thermal_sd) / phi_thermal

print("k = %.5f +- %.5f" % (k_avg, k_sd))
# Plot
N_iter = len(k)
(p1,) = plt.plot(np.arange(1, N_iter + 1), k, "-b", label="MC")
(p2,) = plt.plot(
    np.arange(1, N_iter + 1), np.ones(N_iter) * k_avg, ":r", label="MC-avg"
)
plt.fill_between(
    np.arange(1, N_iter + 1),
    np.ones(N_iter) * (k_avg - k_sd),
    np.ones(N_iter) * (k_avg + k_sd),
    alpha=0.2,
    color="r",
)
plt.xlabel("Iteration #")
plt.ylabel(r"$k$")
plt.grid()
ax2 = plt.gca().twinx()
(p3,) = ax2.plot(np.arange(1, N_iter + 1), rg, "g--", label="GyRad")
plt.ylabel(r"Gyration radius [cm]")
lines = [p1, p2, p3]
plt.legend(lines, [l.get_label() for l in lines])
plt.show()

plt.pcolormesh(X, Y, phi_fast, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast neutron flux")
plt.show()

plt.pcolormesh(X, Y, phi_fast_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast neutron flux stdev")
plt.show()

plt.pcolormesh(X, Y, phi_thermal, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal neutron flux")
plt.show()

plt.pcolormesh(X, Y, phi_thermal_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal neutron flux stdev")
plt.show()
