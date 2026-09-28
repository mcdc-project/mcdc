.. _project_vvp_fixed_source_slab_absorbium:

================
Absorbium Slab
================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Slab problems → Absorbium slab

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/slab_absorbium>`_

Problem setup
-------------

This steady-state, one-group problem places a uniform isotropic source in a 6 cm slab with vacuum boundaries.
The slab contains three 2 cm purely absorbing layers with macroscopic capture cross sections of 1.5, 2.0, and 1.0 cm\ :sup:`-1`, respectively.
The spatial mesh uses 60 cells, and the angular tally divides the direction cosine into 32 bins.

The strong attenuation and material discontinuities test source sampling, collision distance sampling, surface crossings, material changes, and both scalar- and angular-flux normalization without scattering.

Exercised features
------------------

- One-dimensional :math:`z`-axis geometry with multiple material regions.
- Vacuum boundary conditions and material-interface crossings.
- Purely absorbing transport with a uniform isotropic volume source.
- Scalar-flux and angle-resolved mesh tallies.

Mathematical derivation
-----------------------

Let layer :math:`r` occupy :math:`[a_r,b_r]`, with total cross section :math:`\Sigma_r`.
The source is uniform over the 6 cm slab and normalized to one source particle, so its angle-integrated density is :math:`1/6` and its angular density is :math:`q=1/12`.
Because there is no scattering, each layer satisfies

.. math::

   \mu\frac{\partial\psi_r}{\partial z}
   +\Sigma_r\psi_r=q.

For :math:`\mu>0`, integration from the left edge gives

.. math::

   \psi_r(z,\mu)
   =\frac{q}{\Sigma_r}
    +\left[
       \psi_r(a_r,\mu)-\frac{q}{\Sigma_r}
     \right]
     \exp\left[-\frac{\Sigma_r(z-a_r)}{\mu}\right].

For :math:`\mu<0`, integration backward from the right edge gives

.. math::

   \psi_r(z,\mu)
   =\frac{q}{\Sigma_r}
    +\left[
       \psi_r(b_r,\mu)-\frac{q}{\Sigma_r}
     \right]
     \exp\left[-\frac{\Sigma_r(b_r-z)}{|\mu|}\right].

The vacuum conditions are

.. math::

   \psi_1(0,\mu)=0\quad(\mu>0),
   \qquad
   \psi_3(6,\mu)=0\quad(\mu<0),

and angular flux is continuous at :math:`z=2` and :math:`z=4` for every direction.
Starting from the appropriate vacuum boundary and applying the two characteristic expressions recursively therefore determines :math:`\psi_r` throughout all three layers.

The pointwise scalar flux and current are

.. math::

   \phi(z)=\int_{-1}^{1}\psi(z,\mu)\,d\mu,
   \qquad
   J(z)=\int_{-1}^{1}\mu\psi(z,\mu)\,d\mu.

For spatial cell :math:`i` and angular bin :math:`n`, the exact quantities compared with MC/DC are

.. math::

   \overline{\phi}_i
   =\frac{1}{\Delta z_i}
    \int_{z_i}^{z_{i+1}}\phi(z)\,dz,
   \qquad
   \overline{J}_i
   =\frac{1}{\Delta z_i}
    \int_{z_i}^{z_{i+1}}J(z)\,dz,

.. math::

   \overline{\psi}_{i,n}
   =\frac{1}{\Delta z_i\Delta\mu_n}
    \int_{z_i}^{z_{i+1}}
    \int_{\mu_n}^{\mu_{n+1}}
    \psi(z,\mu)\,d\mu\,dz.

Adaptive quadrature evaluates these averages, splitting the directional integrals at :math:`\mu=0`.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--slab_absorbium_flux.png
         :alt: Scalar-flux convergence for the absorbium slab.

         Relative scalar-flux error over the source-particle study.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--slab_absorbium_angular_flux.png
         :alt: Angular-flux convergence for the absorbium slab.

         Relative angular-flux error over the source-particle study.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--slab_absorbium_flux.png
         :alt: Scalar-flux comparison for the absorbium slab.

         Highest-statistics scalar flux and the analytical reference.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--slab_absorbium_angular_flux.png
         :alt: Angular-flux comparison for the absorbium slab.

         Highest-statistics angular flux and the analytical reference.

Both scalar- and angular-flux errors follow the expected inverse-square-root trend, and the highest-statistics profiles reproduce the attenuation and interface behavior of the characteristic solution.
