.. _faq:

==========================
Frequently Asked Questions
==========================

General
-------

**What Python versions does MC/DC support?**

MC/DC is tested with Python 3.10, 3.11, and 3.12.
We recommend Python 3.12 for the best performance and compatibility with Numba.

**What platforms are supported?**

MC/DC is validated on linux-64 (x86), win-64, osx-64 (Intel), osx-arm64 (Apple Silicon),
linux-ppc64 (IBM POWER9), linux-nvidia-cuda, and linux-amd-rocm.

**Should I use pip or conda to install MC/DC?**

For **personal machines and simple setups**, ``pip`` inside a ``venv`` is the easiest route
(see :ref:`install:Installing with pip`).
For **HPCs or non-standard hardware** (e.g., POWER9 on Lassen, or when mpi4py is
troublesome), a **conda environment** with the ``install.sh`` script is more robust
(see :ref:`install:Installing MC/DC via conda`).

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
     - Can build mpi4py from source via ``install.sh``

**Where can I find cross-section data for continuous-energy simulations?**

CE data libraries are provided to CEMeNT members via an internal repository.
Due to export controls they cannot be publicly distributed.
If you need cross-section data, we recommend using
`OpenMC <https://docs.openmc.org>`_ or `NJOY <http://www.njoy21.io/>`_ to generate it,
then converting to MC/DC format with the tool in ``tools/data_library_generator/``.
See :ref:`install:Configuring Continuous Energy Library` for setup instructions.

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

Or use the conda path with ``bash install.sh --hpc``, which handles this automatically.
See :ref:`user/troubleshooting:Building mpi4py from Source` for more details.

**I get Numba version errors or** ``TypingError`` **on older Numba versions.**

MC/DC requires **Numba >= 0.60.0**.
If you are on an older version, upgrade:

.. code-block:: sh

   pip install 'numba>=0.60.0'

If your system constrains the Numba version (e.g., due to CUDA toolkit compatibility),
see :ref:`user/troubleshooting:Numba Version Compatibility` for patching guidance.

Running Simulations
-------------------

**How do I run in parallel with MPI?**

.. code-block:: sh

   mpiexec -n <nprocs> python input.py --mode=numba

On HPCs, use the appropriate launcher (``srun``, ``jsrun``, ``flux run``).
See :ref:`user/batch_scripts:Batch Job Scripts` for ready-to-use templates.

**My simulation is very slow — what should I check?**

#. Are you running in ``--mode=numba``?  Python mode is orders of magnitude slower.
#. First Numba run incurs JIT compilation overhead (15–80 s).
   Subsequent runs with ``--caching`` are much faster.
#. Check your particle count — start small and scale up.

**I see** ``SyntaxWarning: invalid escape sequence`` **on import.**

This is a known cosmetic warning in some older releases (see `#211 <https://github.com/CEMeNT-PSAAP/MCDC/issues/211>`_).
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

See the post-processing section in :ref:`user/first_mcdc:First MC/DC Simulation` for a complete example.

**What visualization tools work with MC/DC output?**

- ``matplotlib`` for quick 1-D / 2-D plots.
- MC/DC's built-in ``mcdc.visualize()`` for geometry inspection.
- `ParaView <https://www.paraview.org/>`_ or `VisIt <https://sd.llnl.gov/simulation/computer-codes/visit>`_ for 3-D data.
