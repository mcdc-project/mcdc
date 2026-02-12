.. _install:

===================
Installation Guide
===================

Whether installing MC/DC as a user or from source as a developer, 
we recommend doing so using an environment manager like venv or conda.
This will avoid the need for any admin access and keep dependencies clean.

In general, :ref:`creating-a-venv-environment` and :ref:`installing-with-pip` is easier and recommended.
Creating a conda environment and :ref:`installing-with-conda` is more robust and reliable, but is also more difficult. 
A conda environment is necessary to install MC/DC on LLNL's Lassen machine.



.. _creating-a-venv-environment:

---------------------------
Creating a venv environment
---------------------------

Python `virtual environments <https://docs.python.org/3/library/venv.html>`_ are the easy and 
recommended way to get MC/DC operating on personal machines as well as HPCs;
all you need is a working Python version with venv installed.
Particularly on HPCs, using a Python virtual environment is convenient because
system admins will have already configured venv and the pip within it to load packages and dependencies
from the proper sources. 
HPCs often use a module system, so before doing anything else, 
``module load python/<version_number>``.

A python virtual environment can (usually) be created using

.. code-block:: sh

    python -m venv <name_of_venv>

Once you have created a venv, you will need to activate it

.. code-block:: sh

    source <name_of_venv>/bin/activate

and will need to do so every time a new terminal instance is launched.
Once your environment is active, you can move on to :ref:`installing-with-pip`.


.. _installing-with-pip:

-------------------
Installing with pip
-------------------
Assuming you have a working Python environment, you can install using pip. 
Doing so within an active venv or conda environment avoids the need for any admin access
and keeps dependencies clean.

If you would like to run MC/DC as published in the main branch *and* 
do not need to develop in MC/DC, you can install from PyPI: 
 
.. code-block:: sh

    pip install mcdc

If you would like to execute a version of MC/DC from a specific branch or 
*do* plan to develop in MC/DC, you'll need to install from source: 

#. Clone the MC/DC repo: ``git clone https://github.com/CEMeNT-PSAAP/MCDC.git`` 
#. Go to your new MC/DC directory: ``cd MCDC``
#. Install the package from your MC/DC files: ``pip install -e .``

This should install all needed dependencies without a hitch. 
The `-e` flag installs MC/DC as an editable package, meaning that any changes
you make to the MC/DC source files, including checking out a different
branch,  will be immediately reflected without needing to do any re-installation.

.. _installing-with-conda:

--------------------------
Installing MC/DC via conda
--------------------------

Conda is the most robust (works even on bespoke systems) option to install MC/DC.
`Conda <https://conda.io/en/latest/>`_ is an open source package and environment management system 
that runs on Windows, macOS, and Linux. It allows for easy installing and switching between multiple
versions of software packages and their dependencies. 
Conda is really useful on systems with non-standard hardware (e.g. not x86 CPUs) like Lassen, where
mpi4py is often the most troublesome dependency. 

First, ``conda`` should be installed with `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
or `Anaconda <https://www.anaconda.com/>`_. HPC instructions: 

`Dane <https://hpc.llnl.gov/hardware/compute-platforms/dane>`_ (LLNL, x86_64), 

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh


`Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_ (LLNL, IBM Power9),

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
    bash Miniconda3-latest-Linux-ppc64le.sh


Then create and activate a new conda environment called *mcdc-env* in
which to install MC/DC. This creates an environment with python3.12 
installed:

.. code-block:: sh

    conda create -n mcdc-env python=3.12
    conda activate mcdc-env

Then, MC/DC can be installed from source by first cloning the MC/DC repository:

.. code-block:: sh

    git clone https://github.com/CEMeNT-PSAAP/MCDC.git
    cd MCDC

then using the the ``install.sh`` within it. The install script will
build MC/DC and all of its dependencies and execute any necessary patches.
This has been tested on Quartz, Dane, Tioga, Lassen, and Apple M2. 
The ``install.sh`` script **will fail outside of a conda environment**.

On HPC machines, the script will install mpi4py 
`from source <https://mpi4py.readthedocs.io/en/stable/install.html#using-distutils>`_.
This means that all appropriate modules must be loaded prior to executing.

On Quartz, the default modules are sufficient (``intel-classic`` and ``mvapich2``). 
On Lassen, ``module load gcc/8 cuda/11.8``. Then, 

.. code-block:: sh

    bash install.sh --hpc


On local machines, mpi4py will be installed using conda,

.. code-block:: sh

    bash install.sh 

To confirm that everything is properly installed, execute ``pytest`` from the MCDC directory. 

-------------------------------------
Configuring Continuous Energy Library
-------------------------------------

MC/DC has continuous energy transport capabilities.
We provide the library and easy install to members of CEMeNT and other close developers.
Due to export controls we cannot build a library and transport functionality in a single source.
If you are a member of CEMeNT you should have access to `this internal repo <https://github.com/CEMeNT-PSAAP/MCDC-Xsec>`_.
You an then either set a flag in the install script like,

.. code-block:: sh

    bash install.sh --config_cont_lib

or run the script after instillation as a stand alone operation with

.. code-block:: sh

    bash config_cont_energy.sh

Both these operations will clone the internal directory to your MCDC directory, untar the compressed folder, then set an environment variable in your bash script.
NOTE: this does assume you are using bash shell.


---------------------------------
GPU Operability (MC/DC+Harmonize)
---------------------------------

MC/DC supports most of its Numba enabled features for GPU compilation and execution.
When targeting GPUs, MC/DC uses the `Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ library as its GPU runtime, a.k.a. the thing that actually executes MC/DC functions.
How Harmonize works gets a little involved, but in short, 
Harmonize acts as MC/DC's GPU runtime by using two major scheduling schemes: an event schedular similar to those implemented in OpenMC and Shift, plus a novel scheduler.
For more information on Harmonize and how we compile MC/DC with it, see this `TOMACs article describing the async scheduler <https://doi.org/10.1145/3626957>`_ or our publications in American Nuclear Society: Math and Comp Meeting in 2025.

If you encounter problems with configuration, please file `Github issues promptly <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_ ,
especially when on supported super computers (LLNL's `Tioga <https://hpc.llnl.gov/hardware/compute-platforms/tioga>`_, `El Capitan <https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems>`_, and `Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_).

Nvidia GPUs
^^^^^^^^^^^

To compile and execute MC/DC on Nvidia GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (CUDA=11.8, Numba>=0.60.0) and a working MC/DC version >=0.10.0. Then,

#. Clone the harmonize repo: ``git clone https://github.com/CEMeNT-PSAAP/harmonize.git``
#. Install into the proper Python env: ``pip install -e .``

Operability should now be enabled. 

AMD GPUs
^^^^^^^^

The prerequisites for AMD operability are slightly more complex and
require a patch to Numba to allow for AMD target triple LLVM-IR.
It is recommended that this is done within a Python venv virtual environment.

To compile and execute MC/DC on AMD GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (ROCm=6.0.0, Numba>=0.60.0) and a working MC/DC version >=0.11.0. Then,

#. Patch Numba to enable HIP (`instructions here <https://github.com/ROCm/numba-hip>`_)
#. Clone harmonize and `switch to the AMD <https://github.com/CEMeNT-PSAAP/harmonize/tree/amd_event_interop_revamp>`_ branch with ``git switch amd_event_interop_revamp``
#. Install Harmonize with ``pip install -e .`` or using `Harmonize's install script <https://github.com/CEMeNT-PSAAP/harmonize/tree/main>`_

Operability should now be enabled.
