
.. _cpu_execution:

=====================
Running MC/DC on CPUs
=====================

Executing MC/DC in something like a jupyter notebook is possible but not recommended,
especially when using MPI and/or Numba.
The instructions below assume you have an existing MC/DC installation.
MPI can be quite tricky to configure if on an HPC; if you're having trouble,
consult our :ref:`installation`, your HPC admin, or our `GitHub issues page <https://github.com/mcdc-project/mcdc/issues>`_.

For the architectural relationship between Python and Numba-CPU execution, see
:doc:`../../developer_guide/architecture/transport_execution`.

Pure Python Mode
----------------

To run in pure Python mode (slower, no acceleration)

.. code-block:: python3

    python input.py

Numba Mode
----------

.. code-block:: python3

    python input.py --mode=numba

When running in Numba mode a significant amount of time is taken compiling Python functions to performant binaries.
Only the functions used in a specific simulation will be compiled.
When ``--caching`` is enabled, compiled binaries can be reused by subsequent
runs of a compatible simulation.
The cache can be used as an effective ahead-of-time compilation scheme where binaries can be compiled once and shared between machines.
For more information on caching, see :ref:`developer_guide/contributing/index:Caching` and
`Numba Caching <https://numba.readthedocs.io/en/stable/developer/caching.html>`_.

MC/DC also has the ability to run Numba in a debugging mode.
This will result in less performant code and longer compile times but will allow for better error messages from Numba and other packages.

.. code-block:: python3

    python input.py --mode=numba_debug


For more information on the exact behavior of this option, see
:ref:`developer_guide/contributing/index:Debugging`.

Using MPI
---------

MC/DC can be executed using MPI with or without Numba acceleration.
If ``numba-mode`` is enabled the ``jit`` compilation, which is executed on all threads, can take between 30s-2min.
For smaller problems, Numba compilation time could exceed runtime, and pure python mode could be preferable.
Below, ``--mode`` can equal python or numba. MC/DC gets MPI functionality via `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_. 
As an example, to run on 36 processes in Numba mode with `SLURM <https://slurm.schedmd.com/documentation.html>`_:

.. code-block:: python3

    srun -n 36 python input.py --mode=<python/numba>

For systems that do not use SLURM (i.e., a local system) try ``mpiexec`` or ``mpirun`` in its stead.

Tally reduction memory
----------------------

To reduce tallies in smaller MPI messages, set the number of contiguous chunks
before running the simulation.

.. code-block:: python3

    simulation.settings.tally_reduction_chunks = 8
    simulation.run()

The default is ``1``, which reduces each tally in a single call.
The value must be a positive integer.
Each rank reduces the chunks in the same order, with at most one chunk per bin
for small tallies.
CPU reductions use views of the existing score array.
GPU-backed reductions reuse a host buffer sized for the largest chunk.
Smaller messages can reduce MPI's temporary memory demand, with additional
collective-call overhead.
