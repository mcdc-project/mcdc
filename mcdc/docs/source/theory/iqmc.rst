.. _iqmc:

====
iQMC
====

The iterative Quasi-Monte Carlo (iQMC) method replaces the pseudo-random number sequences used in conventional Monte Carlo transport with **low-discrepancy sequences** (e.g., Halton or Sobol sequences).
This reduces the variance convergence rate from the standard :math:`O(1/\sqrt{N})` to as fast as :math:`O((\log N)^d / N)` for :math:`d`-dimensional problems, yielding significant efficiency gains for smooth solutions.

Algorithm
---------

iQMC reformulates the transport problem as a fixed-point iteration and combines it with Krylov linear solvers such as GMRES.
At each iteration:

#. Low-discrepancy sample points are generated for the source particle phase-space coordinates.
#. Particles are transported using the standard MC kernel, but with quasi-random initial conditions rather than pseudo-random ones.
#. Scattering and fission source tallies are accumulated to update the right-hand side of the transport equation.
#. A Krylov solver (e.g., GMRES) accelerates the convergence of the source iteration.

This process repeats until the scattering/fission source converges.

Spatial Error Mitigation
------------------------

A known challenge with iQMC is spatial discretization error introduced by the tally mesh.
MC/DC addresses this with **linear discontinuous (LD) source tilting**, which uses a piecewise-linear representation of the scattering and fission sources within each mesh cell rather than a flat (piecewise-constant) approximation.
Additionally, **effective scattering and fission rate tallies** improve the consistency between the transport solve and the source update, reducing spatial bias.

Applications
------------

iQMC is applicable to both fixed-source and k-eigenvalue problems.
Output data from iQMC simulations is stored in the ``iqmc/tally/`` group of the HDF5 output file.

For more details, see:

- S. Pasmann, I. Variansyah, C. T. Kelley, and R. G. McClarren. "Mitigating Spatial Error in iQMC with Linear Discontinuous Source Tilting and Effective Scattering and Fission Rate Tallies." *NSE* (2024). Preprint DOI 10.48550/arXiv.2401.04029.
- S. Pasmann, I. Variansyah, C. T. Kelley, and R. G. McClarren. "A Quasi-Monte Carlo Method with Krylov Linear Solvers for Multigroup Neutron Transport Simulations." *NSE* (2023). DOI 10.1080/00295639.2022.2143704.