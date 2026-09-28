.. _project_vvp_fixed_source_mms_two_group_slab:

==============================
Manufactured Two-Group Slab
==============================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Slab problems → Manufactured two-group slab

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/mms_two_group_slab>`_

Problem setup
-------------

This manufactured problem uses a 10 cm slab with vacuum boundaries, two energy groups, and isotropic scattering.
Both groups have :math:`\Sigma_t=1` cm\ :sup:`-1` and :math:`\Sigma_c=0.6` cm\ :sup:`-1`, while every entry of the outgoing-by-incoming scattering matrix is 0.2 cm\ :sup:`-1`.

The manufactured right-hand sides are

.. math::

   R_1(x)=0.5+0.01x,
   \qquad
   R_2(x)=0.6-0.01x.

Substitution into the transport equations gives the positive, angle-integrated volume sources

.. math::

   q_1(x)=0.56+0.02x,
   \qquad
   q_2(x)=0.76-0.02x.

MC/DC samples each spatial source exactly as a piecewise-linear distribution.
White half-space boundary sources supply the corresponding isotropic incoming fluxes at both ends, and a 100-cell mesh tallies both energy groups.

Exercised features
------------------

- One-dimensional :math:`x`-axis geometry with vacuum boundaries.
- Two-group transport with within-group and intergroup isotropic scattering.
- Group-dependent piecewise-linear spatial source distributions.
- Isotropic incoming boundary flux represented by white half-space sources.
- Energy-resolved mesh-flux tallying.

Mathematical derivation
-----------------------

For group :math:`g`, the steady transport equation is

.. math::

   \mu\frac{\partial\psi_g}{\partial x}+\psi_g
   =\frac{1}{2}\sum_{g'=1}^{2}\Sigma_{s,g\leftarrow g'}\phi_{g'}
    +\frac{q_g(x)}{2}.

The manufactured total right-hand side is selected as the linear function

.. math::

   R_g(x)=a_g+b_gx,
   \qquad
   (a_1,a_2)=(0.5,0.6),
   \qquad
   (b_1,b_2)=(0.01,-0.01).

Solving :math:`\mu\partial_x\psi_g+\psi_g=R_g` along a characteristic and imposing isotropic incoming values :math:`R_g(0)` on the left and :math:`R_g(L)` on the right gives

.. math::

   \psi_g(x,\mu)
   =R_g(x)-\mu b_g
    +\mu b_g
    \begin{cases}
    \exp(-x/\mu),&\mu>0,\\
    \exp[-(L-x)/|\mu|],&\mu<0.
    \end{cases}

Directional integration uses

.. math::

   E_n(y)=\int_1^\infty\frac{e^{-yt}}{t^n}\,dt

and yields

.. math::

   \phi_g(x)
   =2R_g(x)+b_g\left[E_3(x)-E_3(L-x)\right].

Because the two slopes are equal and opposite, their boundary-layer terms cancel in :math:`\phi_1+\phi_2`.
Every scattering-matrix entry is 0.2 cm\ :sup:`-1`, so the isotropic scattering contribution to either angular equation is constant:

.. math::

   \frac{1}{2}\sum_{g'=1}^{2}0.2\phi_{g'}=0.22.

The required angle-integrated external source is therefore

.. math::

   q_g(x)=2\left[R_g(x)-0.22\right],

which gives the two positive source profiles stated above.
The four boundary-source strengths follow from the incident current of an isotropic angular flux,

.. math::

   J_{\mathrm{in}}=\int_0^1\mu\psi_{\mathrm{in}}\,d\mu
   =\frac{\psi_{\mathrm{in}}}{2}.

The two volume sources integrate to 6.6 each, and the four boundary currents sum to 1.1, giving total external strength :math:`S=14.3`.
All analytical fluxes are divided by :math:`S` to match per-source-particle MC/DC normalization.

Finally, since :math:`dE_4(x)/dx=-E_3(x)`, the exact average in spatial cell :math:`[x_i,x_{i+1}]` is

.. math::

   \overline{\phi}_{g,i}
   =\frac{1}{S}\left[
      2\left(a_g+b_g\frac{x_i+x_{i+1}}{2}\right)
      +\frac{b_g}{\Delta x_i}
       \left[-E_4(x)-E_4(L-x)\right]_{x=x_i}^{x=x_{i+1}}
    \right].

Results
-------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--mms_two_group_slab_source.png
         :alt: Manufactured source distributions for the two-group slab.

         Positive group-wise volume-source distributions.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--mms_two_group_slab_flux.png
         :alt: Flux convergence for the manufactured two-group slab.

         Relative multigroup flux error from :math:`10^4` through :math:`10^6` histories.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--mms_two_group_slab_flux.png
         :alt: Flux comparison for the manufactured two-group slab.

         Highest-statistics group fluxes and manufactured reference.

The error metrics closely follow inverse-square-root sampling behavior, while the solution comparison resolves the distinct slopes of both manufactured group fluxes.
