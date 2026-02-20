.. _k_eigenvalue:

===========================
k-Eigenvalue Calculations
===========================

A k-eigenvalue (criticality) calculation determines the effective
neutron multiplication factor :math:`k_{\text{eff}}` and the
fundamental-mode fission source distribution of a system.
MC/DC implements this via the standard **power iteration** algorithm.

The k-Eigenvalue Equation
--------------------------

In steady state the Boltzmann equation with fission becomes an
eigenvalue problem:

.. math::

   \hat{\Omega}\cdot\nabla\psi
   + \Sigma_t\,\psi
   =
   \int \Sigma_s\,\psi'\,dE'\,d\Omega'
   + \frac{1}{k}\,\frac{\chi}{4\pi}\int \nu\Sigma_f\,\phi'\,dE'

where :math:`k = k_{\text{eff}}` is the eigenvalue.  A system is

- **critical** if :math:`k = 1`,
- **supercritical** if :math:`k > 1`,
- **subcritical** if :math:`k < 1`.


Power Iteration
----------------

MC/DC solves the eigenvalue problem using power (source) iteration:

#. An initial guess for :math:`k^{(0)}` is provided (default: 1.0).
#. Source particles are sampled from the current fission source
   distribution.
#. Particles are transported; fission sites are banked.
#. The new eigenvalue estimate is updated:

   .. math::

      k^{(i+1)} = k^{(i)} \;\frac{W_{\text{fission}}^{(i+1)}}{W_{\text{source}}^{(i)}}

   where :math:`W` denotes the total statistical weight.
#. The fission bank becomes the source for the next cycle.
#. Repeat until convergence.

Users configure eigenmode via:

.. code-block:: python3

   mcdc.settings.set_eigenmode(N_inactive=50, N_active=200, k_init=1.0)

- ``N_inactive`` — Cycles discarded for fission source convergence.
- ``N_active`` — Cycles used for tally accumulation.
- ``k_init`` — Initial :math:`k` guess.


Inactive vs. Active Cycles
----------------------------

The first ``N_inactive`` cycles allow the fission source distribution
to converge from the (often arbitrary) initial guess to the
fundamental eigenmode.  Tallies are **not** accumulated during inactive
cycles to avoid bias.

During the ``N_active`` cycles, tally scores are accumulated and
batch statistics (mean, standard deviation) are computed for both
:math:`k_{\text{eff}}` and spatial quantities.


Shannon Entropy
----------------

MC/DC can optionally track the **Shannon entropy** of the fission
source distribution as a convergence diagnostic.  The spatial domain
is divided into :math:`M` mesh cells, and the entropy at cycle
:math:`i` is:

.. math::

   H^{(i)} = -\sum_{m=1}^{M} p_m \log_2 p_m

where :math:`p_m` is the fraction of fission source weight in cell
:math:`m`.  A plateau in :math:`H` over successive cycles indicates
that the source has converged.


Gyration Radius
----------------

The **gyration radius** measures the spatial spread of the fission
source around its centre of mass:

.. math::

   R_g = \sqrt{\frac{\sum_j w_j\,|\mathbf{r}_j - \mathbf{r}_{\text{cm}}|^2}
   {\sum_j w_j}}

MC/DC reports gyration radius diagnostics in the output when running
eigenvalue problems (see the :ref:`example_c5g7_k_eigenvalue` for an
example).


Tips for k-Eigenvalue Simulations
----------------------------------

- **Sufficient inactive cycles**: Too few inactive cycles biases
  :math:`k_{\text{eff}}` and spatial tallies.  Check Shannon entropy
  convergence.
- **Particle count**: Start with :math:`10^3`–:math:`10^4` particles per
  cycle for development, then increase to :math:`10^5`–:math:`10^6` for
  production.
- **Initial source distribution**: A uniform distribution over the
  fissile region is a reasonable default.

References
----------

- T. M. Sutton and A. Morel. "Iteration Acceleration Techniques for
  Monte Carlo Eigenvalue Calculations." *Transactions of ANS* (1996).
- F. B. Brown. "On the Use of Shannon Entropy of the Fission
  Distribution for Assessing Convergence of Monte Carlo Criticality
  Calculations." PHYSOR (2006).
