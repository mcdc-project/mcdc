.. _variance_reduction:

=================================
Variance Reduction Techniques
=================================

Analog Monte Carlo converges as :math:`O(1/\sqrt{N})`, which can be
prohibitively slow for deep-penetration or rare-event problems.
MC/DC provides several **variance reduction** (VR) techniques that
reduce the statistical uncertainty per particle history without
introducing bias.

All techniques below are activated through the ``mcdc.simulation``
interface.

.. contents:: On this page
   :local:
   :depth: 2

Implicit Capture
-----------------

Also called **survival biasing** or **absorption suppression**.
Instead of terminating a particle at every capture event, the particle's
weight is reduced by the non-absorption probability after each
collision:

.. math::

   w' = w \; \frac{\Sigma_s + \Sigma_f}{\Sigma_t}

The particle continues with the reduced weight, ensuring that every
history contributes to tallies for longer.  This is especially
effective in highly absorbing media.

**Usage:**

.. code-block:: python3

   mcdc.simulation.implicit_capture()

.. note::

   Implicit capture is enabled by default in most MC/DC examples.
   Without a companion technique (weight roulette or weight windows)
   to eliminate very-low-weight particles, a memory overhead can build
   up over time.

Weight Roulette
----------------

When a particle's weight drops below a threshold
:math:`w_{\text{thresh}}`, **Russian roulette** is applied:

- With probability :math:`p = w / w_{\text{target}}`, the particle
  survives and its weight is set to :math:`w_{\text{target}}`.
- With probability :math:`1 - p`, the particle is killed.

This prevents an ever-growing population of low-weight particles while
preserving the expected weight (unbiased).

**Usage:**

.. code-block:: python3

   mcdc.simulation.weight_roulette(weight_threshold=0.25, weight_target=1.0)

``weight_threshold`` and ``weight_target`` should be chosen so that
:math:`w_{\text{thresh}} < w_{\text{target}}`; a common ratio is
:math:`w_{\text{thresh}} / w_{\text{target}} \approx 0.25`.


Weighted Emission
------------------

In fission problems, the number of secondary neutrons is
:math:`\lfloor\nu + \xi\rfloor` in analog mode.  **Weighted emission**
instead emits a fixed number of secondaries (``weight_target`` worth
of weight), adjusting their weights so that the total expected weight
is preserved:

.. math::

   w_{\text{child}} = \frac{w \cdot \nu}{n_{\text{emit}}}

This reduces the variance of the fission source weight distribution.

**Usage:**

.. code-block:: python3

   mcdc.simulation.weighted_emission(active=True, weight_target=1.0)


Population Control
-------------------

In time-dependent (transient) problems, the neutron population can
grow or decay exponentially, making it difficult to maintain a
well-sampled phase space.  **Population control** adjusts the particle
bank at each time census by splitting high-weight particles and
rouletting low-weight ones, targeting a uniform weight close to
:math:`w_{\text{target}}`.

**Usage:**

.. code-block:: python3

   mcdc.simulation.population_control()

Population control is typically combined with a time census
(``set_time_census``) that checkpoints the particle population at
specified time boundaries.


Weight Windows
---------------

Weight windows define position-dependent (and optionally energy- and
time-dependent) target weights and bounds.  They combine splitting and
roulette to focus computational effort in regions of high importance.

MC/DC supports both user-defined and automatically generated weight
windows.  See :ref:`ww` for a full description of the available
strategies (``WW_USER`` and ``WW_PREVIOUS``) and modification schemes
(``WW_MIN`` and ``WW_WOLLABER``).


Combining Techniques
---------------------

VR techniques are designed to be composable.  A typical production
setup might use:

.. code-block:: python3

   mcdc.simulation.implicit_capture()
   mcdc.simulation.weight_roulette(weight_threshold=0.25, weight_target=1.0)

For time-dependent fission problems:

.. code-block:: python3

   mcdc.simulation.implicit_capture()
   mcdc.simulation.weighted_emission(active=True, weight_target=1.0)
   mcdc.simulation.population_control()

The order of activation does not matter — MC/DC applies them in the
correct transport-physics order internally.

For quasi-Monte Carlo acceleration of the source iteration, see
:ref:`iqmc`.

References
----------

- T. E. Booth. "A Sample Problem for Variance Reduction in MCNP."
  LA-10363-MS, LANL (1985).
- A. B. Wollaber. "Advanced Monte Carlo Methods for Radiation
  Transport." PhD diss., Univ. of Michigan (2016).
