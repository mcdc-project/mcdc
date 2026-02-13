.. _troubleshooting:

===============
Troubleshooting
===============

This page collects solutions to common installation and runtime problems.
If your issue is not listed here, please check our
`GitHub issues <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_
or open a new one.

.. contents:: On this page
   :local:
   :depth: 2


Numba Version Compatibility
----------------------------

MC/DC requires **Numba >= 0.60.0**.
Symptoms of version mismatch include ``TypingError``, unexpected ``LoweringError``,
or missing ``@njit`` features.

Check your version:

.. code-block:: sh

   python -c "import numba; print(numba.__version__)"

**Upgrading Numba:**

.. code-block:: sh

   # pip
   pip install --upgrade 'numba>=0.60.0'

   # conda
   conda install numba>=0.60.0 -c conda-forge

**Pinning Numba for CUDA compatibility:**

If your system requires a specific CUDA toolkit,
Numba and ``cuda-toolkit`` versions must match.
For example, CUDA 11.8 works best with Numba 0.60.x:

.. code-block:: sh

   conda install numba=0.60 cudatoolkit=11.8 -c conda-forge

**Patching Numba for AMD GPUs (HIP):**

AMD ROCm GPU support requires a patched Numba build.
Follow the `numba-hip instructions <https://github.com/ROCm/numba-hip>`_
to apply the HIP target triple patch.
This is required before installing Harmonize for AMD targets.
See also :ref:`install:AMD GPUs`.


Building mpi4py from Source
----------------------------

On most HPCs, prebuilt mpi4py wheels are incompatible with the system MPI library.
Symptoms include ``MPI_Init`` failures, segfaults at launch, or
``ImportError: libmpi.so: cannot open shared object file``.

**Step 1 — Load the correct MPI module:**

.. code-block:: sh

   # Examples for different systems:
   module load mvapich2              # Quartz (LLNL)
   module load gcc/8 cuda/11.8      # Lassen (LLNL)
   module load cray-mpich            # Tioga / El Capitan (LLNL)

**Step 2 — Install mpi4py from source:**

.. code-block:: sh

   CC=mpicc pip install --no-cache-dir --no-binary mpi4py mpi4py

Or, if using conda with the ``install.sh`` script:

.. code-block:: sh

   bash install.sh --hpc

The ``--hpc`` flag instructs the install script to build mpi4py from source
using the currently loaded MPI module.

**Verifying the installation:**

.. code-block:: sh

   python -c "from mpi4py import MPI; print(MPI.Get_library_version())"

This should print the system MPI library version (e.g., MVAPICH2, Open MPI, Cray MPICH).


HPC Module Environments
------------------------

Before installing or running MC/DC on an HPC, load the required modules.
Incorrect or missing modules are the most common source of build failures.

.. list-table:: Recommended module loads by system
   :widths: 20 40 40
   :header-rows: 1

   * - **System**
     - **Module loads**
     - **Notes**
   * - Quartz (LLNL)
     - ``module load python/3.12``
     - Default ``intel-classic`` + ``mvapich2`` are sufficient
   * - Dane (LLNL)
     - ``module load python/3.12``
     - x86_64, similar to Quartz
   * - Lassen (LLNL)
     - ``module load gcc/8 cuda/11.8``
     - POWER9; conda recommended
   * - Tioga (LLNL)
     - ``module load cray-mpich rocm/6.0.0``
     - AMD MI250X GPUs
   * - El Capitan (LLNL)
     - ``module load cray-mpich rocm/6.0.0``
     - AMD MI300A GPUs

After loading modules, activate your Python environment (venv or conda)
before running ``pip install`` or ``bash install.sh``.


Common Runtime Errors
----------------------

**"No module named 'mcdc'"**

Your Python environment is not activated, or MC/DC was installed in a different environment.
Activate the correct one:

.. code-block:: sh

   source <venv>/bin/activate
   # or
   conda activate <env_name>

**"MCDC_LIB is not set" when running continuous-energy problems**

Set the ``MCDC_LIB`` environment variable to point to your data library directory:

.. code-block:: sh

   export MCDC_LIB=/path/to/mcdc_xsec_library

See :ref:`install:Configuring Continuous Energy Library`.

**Numba compilation takes very long (> 2 minutes)**

First compilation is expected to be slow (15–80 s depending on problem complexity).
Use ``--caching`` to save compiled binaries:

.. code-block:: sh

   python input.py --mode=numba --caching

Subsequent runs will skip compilation.
If compilation seems stuck, check that you are not running
on a login node with limited resources.

**Segmentation fault during MPI runs**

This typically indicates an mpi4py / system MPI mismatch.
Rebuild mpi4py from source as described above.
Also ensure that the number of MPI ranks does not exceed available cores:

.. code-block:: sh

   srun -n <ncores> python input.py --mode=numba

**"AttributeError: 'list' object has no attribute 'ID'" in** ``mcdc.cell()``

This error occurs when passing a Python list instead of using the ``&`` (intersection)
and ``|`` (union) region operators.
Use the operator syntax:

.. code-block:: python3

   # Correct
   mcdc.Cell(region=+s1 & -s2, fill=material)

   # Wrong — do NOT use a list
   mcdc.Cell(region=[+s1, -s2], fill=material)

See `#348 <https://github.com/CEMeNT-PSAAP/MCDC/issues/348>`_.


Getting More Help
------------------

If you are still stuck:

#. Search the `GitHub issues <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_ for similar problems.
#. Run in debug mode for more informative error messages:

   .. code-block:: sh

      python input.py --mode=numba_debug

#. Open a new issue with your error message, Python/Numba versions, and platform info.
