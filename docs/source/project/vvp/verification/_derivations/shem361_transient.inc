Let :math:`\phi(t)\in\mathbb{R}^{G}` be the group-flux vector and :math:`C(t)\in\mathbb{R}^{J}` the delayed-neutron precursor vector, with :math:`G=361`.
The fixed-source variants use

.. math::

   \Sigma_t'=\Sigma_t+0.5\Sigma_c.

Define the speed and decay matrices

.. math::

   V=D(v),
   \qquad
   \Lambda=D(\lambda).

With no spatial leakage, the coupled neutron and precursor balances are

.. math::

   \frac{d\phi}{dt}
   =V\left[
      \left(\Sigma_s+\nu\Sigma_{f,p}-D(\Sigma_t')\right)\phi
      +\chi_d\Lambda C
    \right],

.. math::

   \frac{dC}{dt}
   =\nu_d\Sigma_f\phi-\Lambda C.

For the combined state :math:`y=[\phi,C]^T`, these equations become

.. math::

   \frac{dy}{dt}=By,

with

.. math::

   B=
   \begin{bmatrix}
   V\left(\Sigma_s+\nu\Sigma_{f,p}-D(\Sigma_t')\right)
   &V\chi_d\Lambda\\
   \nu_d\Sigma_f&-\Lambda
   \end{bmatrix}.

A unit neutron-density pulse in group 361 corresponds to

.. math::

   y(0)=
   \begin{bmatrix}
   v_{361}e_{361}\\
   0
   \end{bmatrix},

because group flux equals speed times neutron density.
The state at the end of time interval :math:`k` is

.. math::

   y_{k+1}=e^{B\Delta t_k}y_k.

MC/DC reports interval-averaged flux, so the matching reference is

.. math::

   \overline{y}_k
   =\frac{1}{\Delta t_k}
    \int_0^{\Delta t_k}e^{B\tau}y_k\,d\tau
   =B^{-1}
    \frac{e^{B\Delta t_k}-I}{\Delta t_k}y_k.

The first :math:`G` components give the time-averaged spectrum.
The corresponding total neutron density is

.. math::

   \overline{n}_k
   =\sum_{g=1}^{G}\frac{\overline{\phi}_{k,g}}{v_g}.

Matrix exponentials propagate the reference sequentially over the logarithmic time grid.
