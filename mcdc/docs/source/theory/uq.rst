.. _uq:

==========================
Uncertainty Quantification
==========================

MC/DC supports sampling-based (non-intrusive) uncertainty quantification (UQ).
The fundamental challenge in combining UQ with Monte Carlo transport is that the solver itself introduces stochastic noise, which can bias or obscure the UQ statistics of interest.

Variance Deconvolution
----------------------

The core technique implemented in MC/DC is the **variance deconvolution estimator**.
When uncertain input parameters (e.g., cross sections, densities) are sampled across multiple realizations, the total observed variance in the output contains contributions from both the parametric uncertainty and the Monte Carlo statistical noise.

By the law of total variance,

.. math::

   \text{Var}[Y] = \text{Var}[\mathbb{E}[Y \mid \theta]] + \mathbb{E}[\text{Var}[Y \mid \theta]]

where :math:`Y` is the quantity of interest and :math:`\theta` represents the uncertain parameters.
The first term is the **UQ variance** we want to estimate, while the second term is the **MC noise** that must be subtracted.
The variance deconvolution estimator uses the batched variance estimates from each realization to separate these two contributions, enabling accurate UQ variance computation even when the MC noise is significant.

Implementation
--------------

MC/DC uses a dedicated RNG stream (``SEED_SPLIT_UQ``) for UQ samples, keeping them statistically independent from the transport random walks.
UQ variance statistics are stored in the output HDF5 file alongside the standard tally results, allowing post-processing of both the mean response and its parametric uncertainty.

For more details, see:

- K. B. Clements, G. Geraci, A. J. Olson, and T. S. Palmer. "A variance deconvolution estimator for efficient uncertainty quantification in Monte Carlo radiation transport applications." *JQSRT* (2024). DOI 10.1016/j.jqsrt.2024.108958.
- K. B. Clements, G. Geraci, A. J. Olson, and T. S. Palmer. "Global Sensitivity Analysis in Monte Carlo Radiation Transport." *M&C 2023*.
