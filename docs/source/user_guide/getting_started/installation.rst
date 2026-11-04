.. _installation:

============
Installation
============

Use an environment manager such as venv or conda when installing MC/DC as a user or developer.
An isolated environment avoids administrator access and keeps dependencies clean.

In general, :ref:`creating-a-venv-environment` and :ref:`installing-with-pip` are the easiest options.
Creating a conda environment and :ref:`installing-with-conda` requires more steps but can be more robust on specialized systems.
A conda environment is necessary to install MC/DC on LLNL's Lassen machine.



.. _creating-a-venv-environment:

---------------------------
Creating a venv environment
---------------------------

Python `virtual environments <https://docs.python.org/3/library/venv.html>`_ are the recommended way to run MC/DC on personal machines and HPC systems.
MC/DC supports Python 3.11 and newer, and the selected Python installation must provide ``venv``.
On HPC systems, administrators often provide Python and its package sources through a module system, so load a supported module such as ``python/3.13`` before creating the environment.

Create a Python virtual environment with:

.. code-block:: sh

    python -m venv <name_of_venv>

Activate the environment after creating it:

.. code-block:: sh

    source <name_of_venv>/bin/activate

Activate the environment in each new terminal session before using MC/DC.
Once the environment is active, continue to :ref:`installing-with-pip`.


.. _installing-with-pip:

-------------------
Installing with pip
-------------------
Install MC/DC with pip inside an active venv or conda environment to avoid administrator access and keep dependencies isolated.

Install the latest stable release from PyPI when you do not need to modify MC/DC:
 
.. code-block:: sh

    pip install mcdc

----------------------
Installing from Source
----------------------
Install MC/DC from source when you need a specific branch or plan to contribute changes:
Use Python 3.14 for a contributor environment that will run Black locally; runtime-only source installations may use any supported Python version.

#. Clone the MC/DC repository: ``git clone https://github.com/mcdc-project/mcdc.git``
#. Enter the repository: ``cd mcdc``
#. Install MC/DC and its development tools: ``python -m pip install -e ".[dev]"``

The ``-e`` flag installs MC/DC as an editable package, so source changes and branch switches take effect without reinstalling the package.

.. _installing-with-conda:

--------------------------
Installing MC/DC via conda
--------------------------

`Conda <https://conda.io/en/latest/>`_ provides robust environment management on systems with non-standard hardware or constrained software stacks.
It is particularly useful on HPC systems such as Lassen, where ``mpi4py`` must match the system MPI implementation.

Install conda with `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/>`_.
The following commands install Miniconda on selected HPC architectures.

`Dane <https://hpc.llnl.gov/hardware/compute-platforms/dane>`_ (LLNL, x86_64), 

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh


`Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_ (LLNL, IBM Power9),

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
    bash Miniconda3-latest-Linux-ppc64le.sh


The following example creates and activates a Python 3.13 conda environment named ``mcdc-env``:

.. code-block:: sh

    conda create -n mcdc-env python=3.13
    conda activate mcdc-env

Clone MC/DC and enter the repository:

.. code-block:: sh

    git clone https://github.com/mcdc-project/mcdc.git
    cd mcdc

On an HPC system, load the appropriate MPI module and build ``mpi4py`` against that implementation before installing MC/DC:

.. code-block:: sh

    module load <mpi_module>
    CC=mpicc python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py
    python -m pip install -e ".[dev]"

On a local machine where pip can provide a compatible ``mpi4py`` installation, install MC/DC directly:

.. code-block:: sh

    python -m pip install -e ".[dev]"

Run ``python -m pytest`` from the repository root to verify the installation.

.. _installing-via-containers:

--------------------------
Installing via Containers
--------------------------

For container-based installation and execution, see :doc:`../container`.

.. _install-data-library:

------------------------
Preparing Transport Data
------------------------

See :doc:`../data_preparation/index` for projectile-specific requirements, commands, and source-library conventions for generating transport data.
The :doc:`MC/DC data format reference <../../reference/data_format/index>` specifies the output structure.


---------------------------------
GPU Operability (MC/DC+Harmonize)
---------------------------------

MC/DC supports most of its Numba enabled features for GPU compilation and execution.
When targeting GPUs, MC/DC uses the `Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ library as its GPU runtime, a.k.a. the thing that actually executes MC/DC functions.
Harmonize provides an event scheduler similar to those implemented in OpenMC and Shift, along with a novel asynchronous scheduler.
For more information on Harmonize and how we compile MC/DC with it, see this `TOMACs article describing the async scheduler <https://doi.org/10.1145/3626957>`_ or our publications in American Nuclear Society: Math and Comp Meeting in 2025.

If you encounter configuration problems, please file a `GitHub issue <https://github.com/mcdc-project/mcdc/issues>`_, especially when using supported supercomputers such as LLNL's `Tioga <https://hpc.llnl.gov/hardware/compute-platforms/tioga>`_, `El Capitan <https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems>`_, or `Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_.

.. rubric:: Nvidia GPUs

To compile and execute MC/DC on Nvidia GPUs, first satisfy the `Harmonize prerequisites <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (CUDA 11.8 and Numba 0.61 or newer).

#. Clone the harmonize repo: ``git clone https://github.com/CEMeNT-PSAAP/harmonize.git``
#. Install into the proper Python env: ``pip install -e .``

Operability should now be enabled. 

.. _install-amd-gpus:

.. rubric:: AMD GPUs

The prerequisites for AMD operability require a Numba patch that enables the AMD target triple in LLVM IR.
It is recommended that this is done within a Python venv virtual environment.

To compile and execute MC/DC on AMD GPUs, first satisfy the `Harmonize prerequisites <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (ROCm 6.0.0 and Numba 0.61 or newer).

#. Patch Numba to enable HIP (`instructions here <https://github.com/ROCm/numba-hip>`_)
#. Clone harmonize and `switch to the AMD <https://github.com/CEMeNT-PSAAP/harmonize/tree/amd_event_interop_revamp>`_ branch with ``git switch amd_event_interop_revamp``
#. Install Harmonize with ``pip install -e .`` or using `Harmonize's install script <https://github.com/CEMeNT-PSAAP/harmonize/tree/main>`_

Operability should now be enabled.
