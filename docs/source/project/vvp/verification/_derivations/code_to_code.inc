Suppose :math:`R_i^{(m)}(N)` is the estimate from participating code :math:`m` in tally bin :math:`i` at common sampling level :math:`N`.
For :math:`M` participating codes, the fixed comparison field is the arithmetic mean at the largest shared sample,

.. math::

   R_i^*=\frac{1}{M}\sum_{m=1}^{M}R_i^{(m)}(N_{\max}).

This same field normalizes every sampling level.
For each code pair :math:`a<b`, the pointwise relative difference is

.. math::

   d_i^{(a,b)}(N)
   =\frac{R_i^{(a)}(N)-R_i^{(b)}(N)}{R_i^*},

evaluated only where :math:`R_i^*\neq0`.
With :math:`P=M(M-1)/2` code pairs, the reported pair-averaged 2-norm is

.. math::

   D_2(N)
   =\left[
      \frac{1}{P}
      \sum_{a<b}\left\|d^{(a,b)}(N)\right\|_2^2
    \right]^{1/2},

and the maximum relative difference is

.. math::

   D_{\max}(N)
   =\max_{a<b}\max_i\left|d_i^{(a,b)}(N)\right|.

For the current two-code campaign, these expressions reduce to the norm and maximum of the single MC/DC–OpenMC difference field.
If the codes differ only through independent Monte Carlo noise and increase their sampling together, both metrics should decrease proportionally to :math:`N^{-1/2}`.
