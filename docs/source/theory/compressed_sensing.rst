.. _compressed_sensing:

==================
Compressed Sensing
==================

Compressed sensing is a signal-processing technique that reconstructs a sparse signal from far fewer measurements than traditional Nyquist sampling would require.
In the context of Monte Carlo neutron transport, compressed sensing can be applied to recover spatial or temporal tally distributions from a limited number of particle histories.

Motivation
----------

Monte Carlo simulations can be expensive, and tallies over fine spatial or temporal meshes may have large statistical uncertainties unless a prohibitive number of particles are tracked.
If the underlying solution is sparse or compressible in some basis (e.g., wavelet, Fourier, or modal), compressed sensing theory guarantees that accurate reconstruction is possible from significantly fewer observations.

This is particularly relevant for:

- Fine-mesh tally reconstruction from coarse-mesh Monte Carlo results,
- Time-eigenvalue estimation using dynamic mode decomposition (DMD) with limited snapshots,
- Reducing the overall particle count needed for acceptable tally quality in large problems.

Related Work
------------

Smith, Variansyah, and McClarren explored **Compressed Dynamic Mode Decomposition** for time-eigenvalue calculations, combining sparse sampling ideas with DMD to extract dominant time eigenvalues from transient MC simulations with fewer snapshots than standard approaches require.

For more details, see:

- E. Smith, I. Variansyah, and R. G. McClarren. "Compressed Dynamic Mode Decomposition for Time-Eigenvalue Calculations." *M&C 2023*. Preprint DOI 10.48550/arXiv.2208.10942.

.. note::

   Compressed sensing is an active research area within CEMeNT.
   Full integration into the MC/DC transport solver is under development.