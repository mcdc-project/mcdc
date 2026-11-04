.. _faq:

==========================
Frequently Asked Questions
==========================

General
-------

**What Python versions does MC/DC support?**

MC/DC supports Python 3.11 and newer.

**What platforms are supported?**

MC/DC is validated on linux-64 (x86), win-64, osx-64 (Intel), osx-arm64 (Apple Silicon), linux-ppc64 (IBM POWER9), linux-nvidia-cuda, and linux-amd-rocm.

**Should I use pip or conda to install MC/DC?**

For **personal machines and simple setups**, ``pip`` inside a ``venv`` is the easiest route (see :ref:`user_guide/getting_started/installation:Installing with pip`).
For **HPCs or non-standard hardware**, a **conda environment** can provide more robust environment management while pip installs MC/DC (see :ref:`user_guide/getting_started/installation:Installing MC/DC via conda`).

.. list-table:: pip vs. conda at a glance
   :widths: 30 35 35
   :header-rows: 1

   * - 
     - **pip + venv**
     - **conda**
   * - Ease of setup
     - Easier
     - More steps
   * - HPC compatibility
     - Good (most systems)
     - Best (handles mpi4py, POWER9)
   * - Dependency isolation
     - Good
     - Excellent
   * - MPI support
     - Needs system MPI
     - Can isolate an ``mpi4py`` build matched to the system MPI

**Where can I find cross-section data for continuous-energy simulations?**

See :doc:`Data preparation <data_preparation/index>` for source libraries and instructions for preparing transport data.
The :doc:`data format reference <../reference/data_format/index>` documents the prepared files.

Installation
------------

**I get** ``ModuleNotFoundError: No module named 'mcdc'`` **right after installing.**

Make sure you installed MC/DC inside the same environment you are running from.
If you used ``pip install -e .``, confirm the environment is activated:

.. code-block:: sh

   # venv
   source <name_of_venv>/bin/activate

   # conda
   conda activate <env_name>

**pip install fails with mpi4py errors on an HPC.**

mpi4py must be compiled against the system MPI.
Load the correct MPI module first, then install from source:

.. code-block:: sh

   module load <mpi_module>        # e.g., mvapich2, openmpi, spectrum-mpi
   CC=mpicc pip install --no-binary mpi4py mpi4py

See :ref:`user_guide/troubleshooting:Building mpi4py from Source` for more details.

**I get Numba version errors or** ``TypingError`` **on older Numba versions.**

MC/DC requires **Numba >= 0.61.0**.
If you are on an older version, upgrade:

.. code-block:: sh

   pip install 'numba>=0.61.0'

If your system constrains the Numba version, see :ref:`user_guide/troubleshooting:Numba Version Compatibility` for compatibility guidance.

Running Simulations
-------------------

**How do I run in parallel with MPI?**

.. code-block:: sh

   mpiexec -n <nprocs> python input.py --mode=numba

On HPCs, use the appropriate launcher, such as ``srun``, ``jsrun``, or ``flux run``.
See :ref:`user_guide/execution/batch_systems:Batch Job Scripts` for ready-to-use templates.

**My simulation is very slow — what should I check?**

#. Are you running in ``--mode=numba``?
   Python mode is orders of magnitude slower.
#. The first Numba run incurs JIT compilation overhead of approximately 15–80 seconds.
#. Subsequent runs with ``--caching`` are much faster.
#. Check your particle count — start small and scale up.

**I see** ``SyntaxWarning: invalid escape sequence`` **on import.**

This is a known cosmetic warning in some older releases (see `#211 <https://github.com/mcdc-project/mcdc/issues/211>`_).
It does not affect simulation results.
Updating to the latest MC/DC version resolves it.

Post-processing
---------------

**How do I read MC/DC output files?**

MC/DC writes results to HDF5 (``.h5``) files.
Use ``h5py`` to read them:

.. code-block:: python3

   import h5py
   with h5py.File("output.h5", "r") as f:
       print(list(f.keys()))                    # ['runtime', 'tallies']
       print(list(f["tallies"].keys()))          # list of tally names

See the post-processing section in :ref:`user_guide/getting_started/first_simulation:First MC/DC Simulation` for a complete example.

**What visualization tools work with MC/DC output?**

- ``matplotlib`` for quick 1-D / 2-D plots.
- ``simulation.visualize_model(...)`` for built-in geometry inspection.
- `ParaView <https://www.paraview.org/>`_ or `VisIt <https://sd.llnl.gov/simulation/computer-codes/visit>`_ for 3-D data.
