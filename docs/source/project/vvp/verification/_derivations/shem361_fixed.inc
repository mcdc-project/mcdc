Let :math:`\phi\in\mathbb{R}^{361}` be the group-flux spectrum, with matrices written in outgoing-by-incoming group order.
Increasing capture by 50 percent changes the total cross section to

.. math::

   \Sigma_t'=\Sigma_t+0.5\Sigma_c.

In an infinite homogeneous medium there is no leakage, so the steady group balance is

.. math::

   D(\Sigma_t')\phi
   =\Sigma_s\phi+\nu\Sigma_f\phi+Q,

where :math:`D(a)` denotes the diagonal matrix formed from vector :math:`a`.
Collecting the loss and production terms gives

.. math::

   A\phi=Q,
   \qquad
   A=D(\Sigma_t')-\Sigma_s-\nu\Sigma_f.

The isotropic external source is placed entirely in group 361, so

.. math::

   Q_g=\begin{cases}
   1, & g=361,\\
   0, & \text{otherwise}.
   \end{cases}

The analytical spectrum is therefore the direct linear-system solution

.. math::

   \phi=A^{-1}Q.

Because the medium is homogeneous and infinite, this group balance is the complete transport solution and requires no spatial discretization.
