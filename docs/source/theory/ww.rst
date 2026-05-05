.. _ww:

===============
Weight Windows
===============

Weight windows are a variance reduction technique used to control the statistical weight of particles as they traverse different regions of the problem domain.
They improve computational efficiency by preferentially sampling particles in regions of high importance while suppressing computation in less important regions.

Overview
--------

A weight window defines, for each region of phase space (position, energy, time), a target weight :math:`w_t` and a lower bound :math:`w_{\ell}`.
When a particle enters a region:

- If its weight :math:`w` falls below :math:`w_{\ell}`, Russian roulette is applied: the particle is either killed or its weight is increased to :math:`w_t`.
- If :math:`w` exceeds :math:`w_t / w_{\ell} \cdot w_t` (the upper bound), the particle is split into multiple particles each carrying weight :math:`w_t`.
- Otherwise, the particle is left unchanged.

This keeps particle weights within a controlled band, reducing the variance of tally estimators.

Methods
-------

MC/DC defines two weight window strategies:

- **User-defined** (``WW_USER``): the user explicitly provides the weight window parameters for each region.
- **Previous-iteration** (``WW_PREVIOUS``): weight window bounds are derived from the tally results of a previous simulation or iteration, enabling adaptive weight windows without manual tuning.

Two modification schemes are supported:

- **Minimum** (``WW_MIN``): a straightforward lower-bound–based splitting and roulette scheme.
- **Wollaber** (``WW_WOLLABER``): an automatic parameter adjustment method following the approach of Wollaber, which uses the importance map to dynamically set weight window bounds.

.. note::

   Weight window support is under active development.
   The constants and interfaces are defined, but full transport integration may not be available in all execution modes.
