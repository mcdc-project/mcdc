.. _cont_movement:

===================
Continuous Movement
===================

MC/DC implements **continuous geometry movement** for time-dependent Monte Carlo transport.
Unlike the step-function approach used by most codes (where geometry is frozen within each time step and updated discretely), MC/DC moves surfaces and sources continuously in time, yielding higher-fidelity results for transient problems.

Motivation
----------

In many reactor transient scenarios — such as control rod insertion or withdrawal, fuel pellet movement, or pulsed neutron experiments — the geometry changes during the simulation.
Step-function approximations introduce temporal discretization error that decreases only with finer time steps.
Continuous movement eliminates this error source entirely by solving for the exact intersection of a particle trajectory with a moving surface.

Implementation
--------------

Surfaces and sources can be assigned piecewise-constant velocities using the ``move`` method:

.. code-block:: python3

   surface.move(velocities=[[vx, vy, vz]], durations=[dt])

Each call specifies one or more velocity segments and their durations.
A final static segment (zero velocity, infinite duration) is appended automatically to keep the surface stationary after the prescribed motion ends.
Internally, MC/DC precomputes a ``move_time_grid`` and cumulative ``move_translations`` for each moving object.

Distance-to-Moving-Surface
--------------------------

When computing the distance to a moving surface, MC/DC transforms the particle into the **surface's reference frame** by subtracting the surface translation and adjusting the direction for the relative velocity:

.. math::

   \mathbf{r}' = \mathbf{r} - \left(\mathbf{T}_0 + \mathbf{V} \cdot t_{\text{local}}\right)

.. math::

   \hat{\mathbf{u}}' = \hat{\mathbf{u}} - \frac{\mathbf{V}}{v}

where :math:`\mathbf{V}` is the surface velocity in the current time segment and :math:`v` is the particle speed.
In each piecewise-constant velocity segment, the problem reduces to a stationary distance calculation.
If no intersection is found within the current segment, the particle is advanced to the next time boundary and the computation repeats with the next velocity segment.

Sources can be moved with the same interface (``source.move(...)``), and source particle positions are adjusted to account for the source's displacement at the particle's birth time.

For more details, see:

- I. Variansyah and R. G. McClarren. "High-fidelity treatment for object movement in time-dependent Monte Carlo transport simulations." *M&C 2023*. Preprint DOI 10.48550/arXiv.2305.07641.
