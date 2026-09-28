.. _project_vvp_k_eigenvalue_kornreich:

=========================
Kornreich-Parsons Slab
=========================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → k-eigenvalue suite → Finite-slab criticality problems → Kornreich-Parsons slab

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/k_eigenvalue/cases/kornreich>`_

Problem setup
-------------

This one-group finite slab alternates four beryllium reflector regions with three uranium fuel regions, with every region one material mean free path thick and vacuum boundaries at both ends.
The reflector has :math:`(\Sigma_t,\Sigma_s,\nu\Sigma_f)=(0.371,0.334,0)` cm\ :sup:`-1`, while the fuel has :math:`(0.415,0.334,0.178)` cm\ :sup:`-1`.
MC/DC assigns the fuel removal not attributed to scattering to fission and selects the neutron yield to preserve the published :math:`\nu\Sigma_f`.

The initial source is uniform and isotropic.
Each region is divided into 50 tally cells, and each cycle uses 10,000 particles with 20 inactive cycles before active sampling begins.

Exercised features
------------------

- One-group :math:`k`-eigenvalue transport in a finite heterogeneous slab.
- Repeated fuel-reflector interfaces and vacuum leakage.
- Fission-source banking and population control across cycles.
- Inactive-cycle source convergence and active-cycle estimation.
- Fine-mesh fundamental-mode flux tallying across seven material regions.

Mathematical derivation
-----------------------

Introduce optical distance

.. math::

   \tau(x)=\int_0^x\Sigma_t(s)\,ds.

Every physical region is one mean free path thick, so the seven-region slab becomes a uniform interval in :math:`\tau`.
Define the material-dependent ratios

.. math::

   c_s(\tau)=\frac{\Sigma_s}{\Sigma_t},
   \qquad
   c_f(\tau)=\frac{\nu\Sigma_f}{\Sigma_t},

where :math:`c_f=0` in reflector regions.
Angular integration of the vacuum-boundary transport equation gives

.. math::

   \phi(\tau)
   =\frac{1}{2}\int_0^7
    E_1(|\tau-\tau'|)
    \left[c_s(\tau')
          +\frac{c_f(\tau')}{k}\right]
    \phi(\tau')\,d\tau'.

Let

.. math::

   (Kq)(\tau)
   =\frac{1}{2}\int_0^7E_1(|\tau-\tau'|)q(\tau')\,d\tau'.

The eigenproblem can then be rearranged as

.. math::

   (I-KC_s)\phi=\frac{1}{k}KC_f\phi,

or

.. math::

   M\phi=k\phi,
   \qquad
   M=(I-KC_s)^{-1}KC_f.

One application of :math:`M` first forms the fission source :math:`KC_f\phi` and then solves the fixed-source scattering equation

.. math::

   \phi^{(m+1)}=KC_f\phi^{(n)}+KC_s\phi^{(m)}

to convergence.
Outer power iteration on this operator supplies the dominant multiplication factor and flux mode.

The reference uses equal optical-width cells :math:`h_\tau` and a symmetric Toeplitz kernel.
Its cell-integrated first column is

.. math::

   K_0=1-E_2(h_\tau/2),

.. math::

   K_j=\frac{1}{2}
       \left[
       E_2((j-\tfrac12)h_\tau)
       -E_2((j+\tfrac12)h_\tau)
       \right],
   \qquad j\geq1.

The optical mesh is refined by a factor of eight relative to the tally mesh.
The numerical dominant eigenvalue is required to reproduce the published value

.. math::

   k=1.17361

within a relative tolerance of :math:`10^{-6}`.
Finally, physical cell widths are :math:`\Delta x_j=h_\tau/\Sigma_{t,j}`, the fine mode is normalized by

.. math::

   \sum_j\phi_j\Delta x_j=1,

and groups of eight fine cells are averaged to obtain the MC/DC tally reference.

Results
-------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--kornreich_flux.png
         :alt: Flux-shape convergence for the Kornreich-Parsons slab.

         Relative fundamental-mode flux-shape error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--kornreich_k-effective.png
         :alt: Multiplication-factor convergence for the Kornreich-Parsons slab.

         Relative multiplication-factor error.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--convergence--kornreich_k-effective_errorbar.png
         :alt: Multiplication-factor estimates and uncertainties for the Kornreich-Parsons slab.

         MC/DC estimates, reported uncertainties, and the published reference.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--comparison--kornreich_flux.png
         :alt: Fundamental-mode flux comparison for the Kornreich-Parsons slab.

         Highest-statistics normalized flux and collision-integral reference.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--k_eigenvalue--comparison--kornreich_k_history.png
         :alt: Cycle-by-cycle multiplication-factor history for the Kornreich-Parsons slab.

         Inactive and active cycle history, active mean, uncertainty, and reference.

The flux-shape error and eigenvalue bias decrease overall, although individual sampling levels fluctuate around the ideal guide.
The largest campaign remains slightly below the published multiplication factor, so future campaigns can monitor whether longer active histories remove the residual difference.

Reference
---------

- D. E. Kornreich and D. K. Parsons, `The Green's Function Method for Effective Multiplication Benchmark Calculations in Multi-Region Slab Geometry <https://doi.org/10.1016/j.anucene.2004.03.012>`_, Annals of Nuclear Energy, 2004.
