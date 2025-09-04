import numpy as np
from scipy.integrate import quad


def reference(x, mu):
    dx = x[1:] - x[:-1]
    dmu = mu[1:] - mu[:-1]
    I = len(x) - 1
    N = len(mu) - 1

    # Parameters
    SigmaT3 = 1.0
    SigmaT1 = 1.5
    SigmaT2 = 2.0
    q1 = 1.0 / 6.0 / 2
    q2 = 1.0 / 6.0 / 2
    q3 = 1.0 / 6.0 / 2
    x1 = 2.0
    x2 = 4.0
    x3 = 6.0
    tau1 = SigmaT1 * x1
    tau2 = SigmaT2 * (x2 - x1)
    tau3 = SigmaT3 * (x3 - x2)

    # Angular flux
    def psi1(mu, x):
        if mu > 0.0:
            return q1 / SigmaT1 * (1.0 - np.exp(-SigmaT1 * x / mu))
        elif mu < 0.0:
            return (psi2(mu, x1) - q1 / SigmaT1) * np.exp(
                -SigmaT1 * (x1 - x) / np.abs(mu)
            ) + q1 / SigmaT1

    def psi2(mu, x):
        if mu > 0.0:
            return (psi1(mu, x1) - q2 / SigmaT2) * np.exp(
                -SigmaT2 * (x - x1) / mu
            ) + q2 / SigmaT2
        elif mu < 0.0:
            return (psi3(mu, x2) - q2 / SigmaT2) * np.exp(
                -SigmaT2 * (x2 - x) / np.abs(mu)
            ) + q2 / SigmaT2

    def psi3(mu, x):
        if mu > 0.0:
            return (psi2(mu, x2) - q3 / SigmaT3) * np.exp(
                -SigmaT3 * (x - x2) / mu
            ) + q3 / SigmaT3
        elif mu < 0.0:
            return q3 / SigmaT3 * (1.0 - np.exp(-SigmaT3 * (x3 - x) / np.abs(mu)))

    # Flux
    def phi1(x):
        return quad(psi1, -1, 1, args=(x), points=[0.0])[0]

    def phi2(x):
        return quad(psi2, -1, 1, args=(x), points=[0.0])[0]

    def phi3(x):
        return quad(psi3, -1, 1, args=(x), points=[0.0])[0]

    # Integrands for current
    def mu_psi1(mu, x):
        if mu > 0.0:
            return mu * psi1(mu, x)
        elif mu < 0.0:
            return mu * psi1(mu, x)

    def mu_psi2(mu, x):
        if mu > 0.0:
            return mu * psi2(mu, x)
        elif mu < 0.0:
            return mu * psi2(mu, x)

    def mu_psi3(mu, x):
        if mu > 0.0:
            return mu * psi3(mu, x)
        elif mu < 0.0:
            return mu * psi3(mu, x)

    # Current
    def J1(x):
        return quad(mu_psi1, -1, 1, args=(x), points=[0.0])[0]

    def J2(x):
        return quad(mu_psi2, -1, 1, args=(x), points=[0.0])[0]

    def J3(x):
        return quad(mu_psi3, -1, 1, args=(x), points=[0.0])[0]

    # Angular flux
    def psi1_(x, mu0, mu1):
        return quad(psi1, mu0, mu1, args=(x), points=[0.0])[0]

    def psi2_(x, mu0, mu1):
        return quad(psi2, mu0, mu1, args=(x), points=[0.0])[0]

    def psi3_(x, mu0, mu1):
        return quad(psi3, mu0, mu1, args=(x), points=[0.0])[0]

    phi = np.zeros(I)
    psi = np.zeros((I, N))
    J = np.zeros(I)

    for i in range(int(I / 3)):
        phi[i] = quad(phi1, x[i], x[i + 1])[0] / dx[i]
        J[i] = quad(J1, x[i], x[i + 1])[0] / dx[i]
        for n in range(N):
            mu0 = mu[n]
            mu1 = mu[n + 1]
            psi[i, n] = quad(psi1_, x[i], x[i + 1], args=(mu0, mu1))[0] / dx[i] / dmu[n]
    for i in range(int(I / 3), int(2 * I / 3)):
        phi[i] = quad(phi2, x[i], x[i + 1])[0] / dx[i]
        J[i] = quad(J2, x[i], x[i + 1])[0] / dx[i]
        for n in range(N):
            mu0 = mu[n]
            mu1 = mu[n + 1]
            psi[i, n] = quad(psi2_, x[i], x[i + 1], args=(mu0, mu1))[0] / dx[i] / dmu[n]
    for i in range(int(2 * I / 3), I):
        phi[i] = quad(phi3, x[i], x[i + 1])[0] / dx[i]
        J[i] = quad(J3, x[i], x[i + 1])[0] / dx[i]
        for n in range(N):
            mu0 = mu[n]
            mu1 = mu[n + 1]
            psi[i, n] = quad(psi3_, x[i], x[i + 1], args=(mu0, mu1))[0] / dx[i] / dmu[n]
    for n in range(N):
        mu0 = mu[n]
        mu1 = mu[n + 1]

    return phi, J, psi
