.. _project_vvp_k_eigenvalue_one_group_slab:

======================================
Homogeneous One-Group Slab Criticality
======================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → k-eigenvalue suite → Finite-slab criticality problems → Homogeneous one-group slab criticality

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/k_eigenvalue/cases/one_group_slab>`_

Problem setup
-------------

This one-group multiplying medium occupies a 10 cm slab oriented along the :math:`z` axis with vacuum boundaries.
Its capture, scattering, and fission cross sections are 0.25, 0.50, and 0.25 cm\ :sup:`-1`, respectively, and its prompt neutron yield is 2.519421.
The value of the yield selects a reference multiplication factor of 1.2 for the finite, leakage-dependent system.

The initial fission source is uniform and isotropic, and the fundamental flux shape is tallied over 100 spatial cells.
Every cycle uses 10,000 particles, with 10 inactive cycles followed by the varied number of active cycles.

Exercised features
------------------

- One-group :math:`k`-eigenvalue transport in finite geometry.
- Leakage through opposing vacuum boundaries.
- Fission-source banking and population control across cycles.
- Inactive-cycle source convergence and active-cycle estimation.
- Spatial fundamental-mode flux tallying and multiplication-factor uncertainty estimation.

Mathematical derivation
-----------------------

The one-group transport eigenproblem is

.. math::

   \mu\frac{\partial\psi}{\partial z}
   +\Sigma_t\psi
   =\frac{1}{2}
    \left(\Sigma_s+\frac{\nu\Sigma_f}{k}\right)\phi,
   \qquad
   \phi(z)=\int_{-1}^{1}\psi(z,\mu)\,d\mu,

with vacuum incidence at :math:`z=0` and :math:`z=L`.
Solving along characteristics from both boundaries and integrating over direction gives the collision-integral equation

.. math::

   \phi(z)
   =\frac{a}{2}
    \int_0^L
    E_1\left(\Sigma_t|z-z'|\right)
    \phi(z')\,dz',
   \qquad
   a=\Sigma_s+\frac{\nu\Sigma_f}{k}.

Define the positive integral operator

.. math::

   (K\phi)(z)
   =\int_0^L E_1\left(\Sigma_t|z-z'|\right)\phi(z')\,dz'.

If :math:`K\phi=\lambda_K\phi` for its dominant symmetric eigenmode, then the transport equation requires

.. math::

   1=\frac{a\lambda_K}{2},
   \qquad
   a=\frac{2}{\lambda_K}.

Solving the definition of :math:`a` for the multiplication factor gives

.. math::

   k=\frac{\nu\Sigma_f}{2/\lambda_K-\Sigma_s}.

The reference discretizes :math:`K` on a uniform fine mesh of width :math:`h`.
Because the kernel depends only on separation, the matrix is symmetric Toeplitz.
Using :math:`dE_2(x)/dx=-E_1(x)`, its first-column entries are

.. math::

   K_0=\frac{2}{\Sigma_t}
       \left[1-E_2\left(\frac{\Sigma_th}{2}\right)\right]

for the logarithmically singular diagonal cell and

.. math::

   K_j=\frac{1}{\Sigma_t}
       \left[
       E_2\left(\Sigma_t(j-\tfrac12)h\right)
       -E_2\left(\Sigma_t(j+\tfrac12)h\right)
       \right],
   \qquad j\geq1.

Power iteration computes :math:`\lambda_K` and its positive mode on a mesh 40 times finer than the tally mesh.
The selected neutron yield gives :math:`k=1.2` from the equation above.
The fine flux is normalized by

.. math::

   \sum_j\phi_jh=1,

then averaged in blocks of 40 cells to form the MC/DC cell-wise reference.

Results
-------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--one_group_slab_flux.png
         :alt: Flux-shape convergence for the homogeneous one-group slab criticality problem.

         Relative fundamental-mode flux-shape error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--one_group_slab_k-effective.png
         :alt: Multiplication-factor convergence for the homogeneous one-group slab criticality problem.

         Relative multiplication-factor error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--one_group_slab_k-effective_errorbar.png
         :alt: Multiplication-factor estimates and uncertainties for the homogeneous one-group slab criticality problem.

         MC/DC estimates, reported uncertainties, and the reference value.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--comparison--one_group_slab_flux.png
         :alt: Fundamental-mode flux comparison for the homogeneous one-group slab criticality problem.

         Highest-statistics normalized flux and semi-analytical reference.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--comparison--one_group_slab_k_history.png
         :alt: Cycle-by-cycle multiplication-factor history for the homogeneous one-group slab criticality problem.

         Inactive and active cycle history, active mean, uncertainty, and reference.

The multiplication factor converges closely to 1.2 and its uncertainty contracts as active cycles are added.
The flux error is statistically noisier but decreases overall, and the highest-statistics solution reproduces the symmetric leakage-shaped fundamental mode.
