.. _mc_basics:

============================
Monte Carlo Transport Basics
============================

This page provides a concise primer on the Monte Carlo method for neutron
transport as implemented in MC/DC.  Readers already familiar with MC
transport may skip ahead to the advanced theory pages.

The Boltzmann Transport Equation
---------------------------------

MC/DC solves the linear Boltzmann transport equation for the angular
neutron flux :math:`\psi(\mathbf{r}, \hat{\Omega}, E, t)`:

.. math::

   \frac{1}{v}\frac{\partial\psi}{\partial t}
   + \hat{\Omega}\cdot\nabla\psi
   + \Sigma_t\,\psi
   =
   \int_{4\pi}\!\int_0^\infty \Sigma_s(E'\to E,\hat{\Omega}'\to\hat{\Omega})\,\psi'\,dE'\,d\Omega'
   + \frac{\chi(E)}{4\pi}\int_0^\infty \nu\Sigma_f(E')\,\phi(E')\,dE'
   + Q

where

- :math:`\Sigma_t`, :math:`\Sigma_s`, :math:`\Sigma_f` are the macroscopic total,
  scattering, and fission cross sections (cm\ :sup:`-1`),
- :math:`\nu` is the average number of neutrons emitted per fission,
- :math:`\chi(E)` is the fission spectrum,
- :math:`\phi(E) = \int_{4\pi}\psi\,d\Omega` is the scalar flux,
- :math:`Q` is an external source.

The scalar flux :math:`\phi` is the primary quantity of interest for
most tallies.


Random Walk
-----------

Monte Carlo solves the transport equation by simulating individual
neutron **random walks** (histories).  Each history proceeds as:

#. **Birth** — A neutron is sampled from the source distribution
   :math:`Q(\mathbf{r}, \hat{\Omega}, E, t)`.
#. **Free flight** — The distance to the next collision is sampled
   from the exponential distribution:

   .. math::

      d = -\frac{\ln\xi}{\Sigma_t(E)}

   where :math:`\xi` is a uniform random number on :math:`(0,1)`.
   If a geometry boundary is reached first, the particle crosses (or
   reflects/leaks) and the free flight continues.
#. **Collision** — The interaction type is selected by the ratio of
   partial to total cross sections: scattering
   (:math:`\Sigma_s / \Sigma_t`), capture (:math:`\Sigma_c / \Sigma_t`),
   or fission (:math:`\Sigma_f / \Sigma_t`).

   - **Scattering**: New direction and energy are sampled from the
     differential scattering kernel.
   - **Capture**: The neutron is absorbed (history terminates in analog mode).
   - **Fission**: :math:`\lfloor\nu + \xi\rfloor` secondary neutrons are
     produced and banked for later processing.

#. **Termination** — The history ends when the particle is absorbed,
   leaks out of the domain, or falls below a weight threshold.


Tallying
--------

During the random walk, MC/DC accumulates **tally scores** — estimates
of physical quantities such as scalar flux, fission rate, or neutron
density.

MC/DC supports two main estimator types:

**Track-length estimator**
   Accumulates the path length :math:`\ell` of each flight through a
   tally region, weighted by the particle weight :math:`w`:

   .. math::

      \hat{\phi}_V = \frac{1}{V} \sum_{\text{tracks}} w\,\ell

   This is the default estimator for mesh tallies (``TallyMesh``).

**Collision estimator**
   Scores at each collision site:

   .. math::

      \hat{\phi}_V = \frac{1}{V\,\Sigma_t} \sum_{\text{collisions}} w

Both estimators are unbiased for the volume-averaged scalar flux.
The track-length estimator generally has lower variance because it
scores on every flight segment, not just at collision points.

Cell tallies (``TallyCell``) and surface tallies (``TallySurface``) use
similar estimator concepts applied to specific geometric cells or
surface crossings.

Statistical Uncertainty
-----------------------

MC/DC uses **batch statistics** to estimate the uncertainty of tally
results.  The simulation is divided into :math:`N_b` independent
batches, each with :math:`N_p / N_b` particles.  The sample mean and
sample standard deviation of the batch means provide the central
estimate and its statistical error:

.. math::

   \bar{x} = \frac{1}{N_b}\sum_{b=1}^{N_b} x_b, \qquad
   \sigma_{\bar{x}} = \sqrt{\frac{1}{N_b(N_b-1)}\sum_{b=1}^{N_b}(x_b - \bar{x})^2}

The relative standard deviation
:math:`\sigma_{\bar{x}} / \bar{x}` decreases as
:math:`O(1/\sqrt{N_p})`.  For faster convergence, see:

- :ref:`variance_reduction` — implicit capture, weight roulette,
  population control.
- :ref:`ww` — weight windows.
- :ref:`iqmc` — quasi-Monte Carlo methods for  :math:`O((\log N)^d / N)`
  convergence.

References
----------

- E. E. Lewis and W. F. Miller, Jr. *Computational Methods of
  Neutron Transport.* John Wiley & Sons (1984).
- L. L. Carter and E. D. Cashwell. *Particle-Transport Simulation
  with the Monte Carlo Method.* ERDA (1975).
