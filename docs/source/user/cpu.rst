
.. _cpu:

=====================
Running MC/DC on CPUs
=====================

Executing MC/DC in something like a jupyter notebook is possible but not recommended,
especially when using MPI and/or Numba.
The instructions below assume you have an existing MC/DC installation.
MPI can be quite tricky to configure if on an HPC; if you're having trouble,
consult our :ref:`install`, your HPC admin, or our `GitHub issues page <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_.

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
These binaries will be cached, meaning that in subsequent runs of the same simulation the compilation step can be avoided.
The cache can be used as an effective ahead-of-time compilation scheme where binaries can be compiled once and shared between machines.
For more information on caching see :ref:`contribution/index:Caching <contribution/index:Caching>` and `Numba Caching <https://numba.readthedocs.io/en/stable/developer/caching.html>`_.

MC/DC also has the ability to run Numba in a debugging mode.
This will result in less performant code and longer compile times but will allow for better error messages from Numba and other packages.

.. code-block:: python3

    python input.py --mode=numba_debug


For more information on the exact behavior of this option see :ref:`contribution/index:Debugging <contribution/index:Debugging>`

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

CPU Profiling
-------------
