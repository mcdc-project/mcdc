Let :math:`\alpha` multiply the original capture contribution, so the total cross section used by the eigenproblem is

.. math::

   \Sigma_t^{(\alpha)}
   =\Sigma_t+(\alpha-1)\Sigma_c.

The infinite homogeneous medium has no leakage, and its group balance is

.. math::

   \left[D(\Sigma_t^{(\alpha)})-\Sigma_s\right]\phi
   =\frac{1}{k}\nu\Sigma_f\phi.

Defining

.. math::

   L=D(\Sigma_t^{(\alpha)})-\Sigma_s,
   \qquad
   F=\nu\Sigma_f,

gives the generalized eigenproblem

.. math::

   F\phi=kL\phi.

All finite, nearly real eigenpairs are retained, and the eigenpair with the largest real eigenvalue supplies the physical fundamental mode.
Its sign is selected so that the group sum is positive, and the spectrum is normalized by

.. math::

   \sum_{g=1}^{361}\phi_g=1.

The resulting dominant eigenvalue is the reference multiplication factor and the normalized eigenvector is the reference energy spectrum.
