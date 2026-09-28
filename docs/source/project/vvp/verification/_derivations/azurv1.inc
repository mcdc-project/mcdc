With unit neutron speed and unit total cross section, the one-group angular flux satisfies

.. math::

   \frac{\partial \psi}{\partial t}
   +\mu\frac{\partial \psi}{\partial x}
   +\psi
   =\frac{c}{2}\phi,
   \qquad
   \phi(x,t)=\int_{-1}^{1}\psi(x,\mu,t)\,d\mu,

with the isotropic pulse represented by

.. math::

   \psi(x,\mu,0^+)=\frac{1}{2}\delta(x).

The effective production ratio is

.. math::

   c=\frac{\Sigma_s+\nu\Sigma_f}{\Sigma_t}.

Fourier transformation in space with wavenumber :math:`p` and Laplace transformation in time with variable :math:`s` give

.. math::

   \widetilde{\psi}(p,\mu,s)
   =\frac{1+c\widetilde{\phi}(p,s)}
          {2(s+1+ip\mu)}.

Integrating over direction defines

.. math::

   L(p,s)
   =\frac{1}{2}\int_{-1}^{1}\frac{d\mu}{s+1+ip\mu}
   =\frac{1}{2ip}
     \ln\left(\frac{s+1+ip}{s+1-ip}\right),

so the transformed scalar flux is

.. math::

   \widetilde{\phi}(p,s)=\frac{L(p,s)}{1-cL(p,s)}.

Ganapol's inversion of these transforms can be written for :math:`|x|<t` by defining

.. math::

   \eta=\frac{x}{t},
   \qquad
   q=\frac{1+\eta}{1-\eta},
   \qquad
   \xi(u)=\frac{\ln q+iu}{\eta+i\tan(u/2)}.

The real-space scalar flux used by the reference generator is

.. math::

   \phi(x,t)
   =\frac{e^{-t}}{2t}
    \left[
      1+\frac{ct}{4\pi}(1-\eta^2)
      \int_0^\pi
      \sec^2\left(\frac{u}{2}\right)
      \mathop{\mathrm{Re}}\left\{
        \xi(u)^2
        \exp\left[
          \frac{ct}{2}(1-\eta^2)\xi(u)
        \right]
      \right\}
      \,du
    \right].

Causality gives :math:`\phi(x,t)=0` for :math:`t=0` or :math:`|x|\geq t` in the regular, bin-averaged solution.
For spatial cell :math:`[x_j,x_{j+1}]` and time interval :math:`[t_k,t_{k+1}]`, the plotted reference is

.. math::

   \overline{\phi}_{k,j}
   =\frac{1}{\Delta t_k\Delta x_j}
    \int_{x_j}^{x_{j+1}}
    \int_{t_k}^{t_{k+1}}
    \phi(x,t)\,dt\,dx.

Adaptive quadrature evaluates the angular-inversion integral and both bin integrations.
