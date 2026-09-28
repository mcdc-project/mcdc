.. _project_vvp_fixed_source_slab_isobeam_td:

====================================
Time-Dependent Isotropic-Beam Slab
====================================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Slab problems → Time-dependent isotropic-beam slab

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/slab_isobeam_td>`_

Problem setup
-------------

This one-group transient transports neutrons through a 5 cm purely absorbing slab oriented along the :math:`y` axis.
The macroscopic capture cross section is 1 cm\ :sup:`-1`, both slab boundaries are vacuum, and the particle speed is one distance unit per time unit.

A white source immediately inside the lower boundary emits only toward positive :math:`y` and is uniform from time zero through time five.
The flux is tallied over 50 spatial cells and 50 time intervals.
The case tests directional boundary-source sampling, causal transient propagation, and two-dimensional space-time tally normalization.

Exercised features
------------------

- One-dimensional :math:`y`-axis geometry with vacuum boundaries.
- Time-dependent purely absorbing transport with finite particle speed.
- Directional white boundary-source sampling.
- Space-time mesh-flux tallying and causal-front resolution.

Mathematical derivation
-----------------------

For :math:`\mu>0`, the purely absorbing angular equation is

.. math::

   \frac{1}{v}\frac{\partial\psi}{\partial t}
   +\mu\frac{\partial\psi}{\partial y}
   +\Sigma_t\psi=0.

The white boundary source is uniform over :math:`0\leq t\leq T` and samples the incident current distribution.
At position :math:`y` and time :math:`t\leq T`, only directions satisfying the flight-time condition :math:`y/(v\mu)\leq t` contribute.
The scalar flux is consequently

.. math::

   \phi(y,t)
   =\frac{1}{T}
    \int_{y/(vt)}^1
    \exp\left(-\frac{\Sigma_ty}{\mu}\right)d\mu,
   \qquad y\leq vt,

and :math:`\phi(y,t)=0` for :math:`y>vt`.
Let :math:`b=\Sigma_ty` and substitute :math:`u=b/\mu`.
Using :math:`E_1(u)=\int_u^\infty e^{-s}/s\,ds` gives

.. math::

   \int_{y/(vt)}^1e^{-b/\mu}\,d\mu
   =b\left[E_1(\Sigma_tvt)-E_1(b)\right]
    +e^{-b}
    -\frac{y}{vt}e^{-\Sigma_tvt}.

Therefore,

.. math::

   \phi(y,t)
   =\frac{1}{T}\left[
      \Sigma_ty\left[E_1(\Sigma_tvt)-E_1(\Sigma_ty)\right]
      +e^{-\Sigma_ty}
      -\frac{y}{vt}e^{-\Sigma_tvt}
    \right].

For spatial cell :math:`j` and time interval :math:`k`, the comparison uses

.. math::

   \overline{\phi}_{k,j}
   =\frac{1}{\Delta t_k\Delta y_j}
    \int_{t_k}^{t_{k+1}}
    \int_{y_j}^{y_{j+1}}
    \phi(y,t)\,dy\,dt,

which is evaluated by adaptive quadrature with the causal zero condition applied pointwise.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--slab_isobeam_td_flux.png
         :alt: Flux convergence for the time-dependent isotropic-beam slab.

         Relative space-time flux error over the source-particle study.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--slab_isobeam_td_flux.gif
         :alt: Animated flux comparison for the time-dependent isotropic-beam slab.

         Highest-statistics MC/DC and analytical flux throughout the transient.

The error follows the expected statistical trend, and the animation shows that MC/DC reproduces both the moving causal front and the attenuated profile behind it.
