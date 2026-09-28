.. _project_vvp_fixed_source_reed:

======================
Reed's Slab Problem
======================

**VVP map:** :doc:`Verification <../../../index>` → Analytical → Neutron → Fixed-source suite → Slab problems → Reed's slab problem

**Case files:** `MC/DC-VVP case folder <https://github.com/mcdc-project/mcdc-vvp/tree/dev/verification/analytical/neutron/fixed_source/cases/reed>`_

Problem setup
-------------

Reed's one-group benchmark is modeled on the half-domain from 0 to 8 cm, with reflection at the symmetry plane and vacuum at the outer boundary.
Four material regions create a severe transport problem: a strongly absorbing region from 0 to 2 cm, a moderately absorbing region from 2 to 3 cm, a void from 3 to 5 cm, and a scattering material from 5 to 8 cm.
Their capture cross sections are 50, 5, 0, and 0.1 cm\ :sup:`-1`, respectively, and the outer material has a scattering cross section of 0.9 cm\ :sup:`-1`.

An intense isotropic source occupies the innermost absorbing region, while a source one hundred times weaker occupies the first centimeter of the outer material.
The 80-cell tally resolves sharp attenuation, void streaming, and the transition to a scattering-dominated region.

Exercised features
------------------

- One-dimensional :math:`z`-axis heterogeneous slab geometry.
- Reflective symmetry and vacuum outer boundaries.
- Strong absorption, isotropic scattering, and void streaming in one model.
- Multiple isotropic volume sources with unequal strengths.
- Fine spatial mesh-flux tallying across material discontinuities.

Mathematical derivation
-----------------------

For discrete directions :math:`\mu_m` and weights :math:`w_m`, the one-group :math:`S_N` equations in constant-material subregion :math:`r` are

.. math::

   \mu_m\frac{d\psi_m}{dz}
   +\Sigma_{t,r}\psi_m
   =\frac{\Sigma_{s,r}}{2}
    \sum_nw_n\psi_n
    +\frac{q_r}{2}.

With :math:`\boldsymbol{\psi}=[\psi_1,\ldots,\psi_N]^T`, :math:`D_\mu=D(\mu)`, and :math:`\mathbf{1}` the vector of ones, this becomes

.. math::

   \frac{d\boldsymbol{\psi}}{dz}
   =A_r\boldsymbol{\psi}+b_r,

.. math::

   A_r=D_\mu^{-1}
   \left[-\Sigma_{t,r}I
   +\frac{\Sigma_{s,r}}{2}\mathbf{1}w^T\right],
   \qquad
   b_r=D_\mu^{-1}\frac{q_r}{2}\mathbf{1}.

The exact solution within each subregion is

.. math::

   \boldsymbol{\psi}_r(z)
   =\boldsymbol{\psi}_{p,r}
    +e^{A_r(z-z_r)}c_r,
   \qquad
   A_r\boldsymbol{\psi}_{p,r}+b_r=0.

The constants :math:`c_r` follow from reflective symmetry at :math:`z=0`, vacuum incidence at :math:`z=8`, and continuity of every angular flux at :math:`z=2`, 3, 5, and 6.
The interface at :math:`z=6` is required because the material remains unchanged but the weak outer source ends there.
The scalar flux is :math:`\Phi_r(z)=w^T\boldsymbol{\psi}_r(z)`.

Warsa's symbolic solution, as implemented by the reference generator, has the following compact form:

.. math::

   \begin{aligned}
   \Phi_1(z)&=1-\sum_{n=1}^4a_{1,n}\cosh(\lambda_{1,n}z),\\
   \Phi_2(z)&=\sum_{n=1}^4
      \left(a_{2,n}^-e^{-\lambda_{2,n}^-z}
           +a_{2,n}^+e^{\lambda_{2,n}^+z}\right),\\
   \Phi_3(z)&=1.105109108062394,\\
   \Phi_4(z)&=10-\sum_{n=1}^4
      \left(a_{4,n}^-e^{-\lambda_{4,n}z}
           +a_{4,n}^+e^{\lambda_{4,n}z}\right),\\
   \Phi_5(z)&=\sum_{n=1}^4
      \left(a_{5,n}^-e^{-\lambda_{5,n}z}
           -a_{5,n}^+e^{\lambda_{5,n}z}\right).
   \end{aligned}

The subregions for :math:`\Phi_1` through :math:`\Phi_5` are :math:`[0,2]`, :math:`[2,3]`, :math:`[3,5]`, :math:`[5,6]`, and :math:`[6,8]`, respectively.
The numerical mode constants are

.. math::

   \begin{aligned}
   \lambda_1={}&(52.06761236,62.76152119,95.14161079,272.57664812),\\
   a_1={}&(5.96168048\!\times\!10^{-47},6.78355315\!\times\!10^{-56},
   7.20274050\!\times\!10^{-84},6.34541151\!\times\!10^{-238}),\\
   \lambda_2^-={}&(5.206761236,6.276152119,9.514161079,27.25766481),\\
   a_2^-={}&(1.68580877\!\times\!10^3,3.14386737\!\times\!10^4,
   2.87997711\!\times\!10^7,8.59419051\!\times\!10^{22}),\\
   \lambda_2^+={}&(27.25766481,9.514161079,6.276152119,5.206761236),\\
   a_2^+={}&(1.29842604\!\times\!10^{-36},1.43234466\!\times\!10^{-13},
   1.51456227\!\times\!10^{-9},1.59443121\!\times\!10^{-8}).
   \end{aligned}

For the final two source subregions,

.. math::

   \begin{aligned}
   \lambda_4={}&\lambda_5
   =(0.5254295183,1.108937229,1.615640334,4.554850586),\\
   a_4^-={}&(75.34793865,20.42874998,712.9175418,2.71640937\!\times\!10^9),\\
   a_4^+={}&(0.1983746884,7.82476533\!\times\!10^{-5},
   9.74666021\!\times\!10^{-6},2.89509835\!\times\!10^{-13}),\\
   a_5^-={}&(31.53212163,26.25911060,1841.223066,1.55559355\!\times\!10^{11}),\\
   a_5^+={}&(3.11931035\!\times\!10^{-3},6.33640114\!\times\!10^{-7},
   3.52875768\!\times\!10^{-8},4.40551434\!\times\!10^{-18}).
   \end{aligned}

The reference generator uses analytical antiderivatives of these exponentials and hyperbolic functions.
For tally cell :math:`[z_i,z_{i+1}]` contained in subregion :math:`r`, the normalized comparison value is

.. math::

   \overline{\phi}_i
   =\frac{1}{100\Delta z_i}
    \int_{z_i}^{z_{i+1}}\Phi_r(z)\,dz.

This cell averaging preserves the sharp solution behavior without evaluating the reference only at tally midpoints.

Results
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--convergence--reed_flux.png
         :alt: Flux convergence for Reed's problem.

         Relative scalar-flux error over the source-particle study.

   .. grid-item::

      .. figure:: https://github.com/mcdc-project/mcdc/releases/download/vvp-results/verification--analytical--neutron--fixed_source--comparison--reed_flux.png
         :alt: Flux comparison for Reed's problem.

         Highest-statistics MC/DC flux and the semi-analytical reference.

The global error decreases with increasing sampling effort, while the maximum metric shows the greater pointwise variability expected in the sharply attenuated regions.
The highest-statistics solution captures the discontinuous material response and void-streaming structure.

References
----------

- W. H. Reed, *New Difference Schemes for the Neutron Transport Equation*, Nuclear Science and Engineering, 1971.
- J. S. Warsa, `Analytical S_N solutions in heterogeneous slabs using symbolic algebra <https://doi.org/10.1016/S0306-4549(01)00080-9>`_, Annals of Nuclear Energy, 2002.
