import numpy as np
from scipy.integrate import quad

# =============================================================================
# Reference solution generator
# =============================================================================

# Scattering ratio
c = 1.1
i = complex(0, 1)

# Spatial grid
J = 201
x = np.linspace(-20.5, 20.5, J + 1)

# Time grid
K = 20
t = np.linspace(0.0, 20.0, K + 1)


def integrand(u, eta, t):
    q = (1 + eta) / (1 - eta)
    xi = (np.log(q) + i * u) / (eta + i * np.tan(u / 2))
    return (
        1.0
        / (np.cos(u / 2)) ** 2
        * (xi**2 * np.e ** (c * t / 2 * (1 - eta**2) * xi)).real
    )


def phi(x, t):
    if t == 0.0 or abs(x) >= t:
        return 0.0
    eta = x / t
    integral = quad(integrand, 0.0, np.pi, args=(eta, t))[0]
    return np.e**-t / 2 / t * (1 + c * t / 4 / np.pi * (1 - eta**2) * integral)


def phi_t(t, x):
    if t == 0.0 or abs(x) >= t:
        return 0.0
    eta = x / t
    integral = quad(integrand, 0.0, np.pi, args=(eta, t))[0]
    return np.e**-t / 2 / t * (1 + c * t / 4 / np.pi * (1 - eta**2) * integral)


def phiX(x, t0, t1):
    return quad(phi_t, t0, t1, args=(x))[0]


phi_avg = np.zeros([K, J])

for k in range(K):
    for j in range(J):
        x0 = x[j]
        x1 = x[j + 1]
        dx = x1 - x0
        t0 = t[k]
        t1 = t[k + 1]
        dt = t1 - t0
        phi_avg[k, j] = quad(phiX, x0, x1, args=(t0, t1))[0] / dx / dt

for j in range(J + 1):
    for k in range(K):
        t0 = t[k]
        t1 = t[k + 1]
        dt = t1 - t0

phi_avg = np.nan_to_num(phi_avg)

np.savez("reference.npz", x=x, t=t, phi=phi_avg)
