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

----------------------
Installing from Source
----------------------
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
which to install MC/DC. MC/DC supports Python ``>3.10``;
we recommend Python 3.11:

.. code-block:: sh

    conda create -n mcdc-env python=3.11
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

.. _installing-via-containers:

--------------------------
Installing via Containers
--------------------------

For container-based installation and execution, see :doc:`user/container`.

.. toctree::
   :maxdepth: 1

   user/container

.. _install-data-library:

-----------------------------------------
Generating a Data Library from ACE Files
-----------------------------------------

MC/DC ships with a conversion tool in ``tools/data_library_generator/`` that reads
standard ACE-format nuclear data files and writes them into MC/DC's per-nuclide
HDF5 format.  This is the primary path for creating CE libraries.

**Prerequisites:**

.. code-block:: sh

   pip install ACEtk h5py numpy tqdm

You also need a set of ACE files (e.g., from `NJOY <http://www.njoy21.io/>`_ or
an ENDF/B distribution).

**Environment variables:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Description
   * - ``MCDC_ACELIB``
     - Path to the directory containing your ACE files.
   * - ``MCDC_LIB``
     - Path to the output directory where MC/DC HDF5 files will be written.

**Running the generator:**

.. code-block:: sh

   export MCDC_ACELIB=/path/to/ace/files
   export MCDC_LIB=/path/to/mcdc/library

   cd tools/data_library_generator
   python generate.py

By default the tool only converts nuclides that do not already have a corresponding
HDF5 file in ``$MCDC_LIB``.  Use ``--rewrite`` to regenerate all files, or
``--verbose`` for detailed per-nuclide output:

.. code-block:: sh

   python generate.py --rewrite --verbose

The generator processes each ACE file as follows:

#. Reads the ACE header to determine nuclide identity (Z, A, isomeric state)
   and temperature.
#. Extracts the principal cross-section block (energy grid, elastic, capture,
   fission, inelastic channels) and writes them as HDF5 datasets grouped by
   reaction type (elastic scattering, capture, inelastic scattering, fission).
#. Extracts angular distributions (tabulated cosine PDFs) and energy
   distributions (level scattering, evaporation, Maxwellian, Kalbach-Mann,
   N-body phase space, tabulated outgoing energy) for each reaction channel.
#. For fissionable nuclides, extracts prompt/delayed :math:`\nu(E)` multiplicities,
   delayed neutron precursor fractions, decay constants, and energy spectra.

The resulting HDF5 file (e.g., ``U235-293.6K.h5``) is ready for use with ``mcdc.Material()``.


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

.. rubric:: Nvidia GPUs

To compile and execute MC/DC on Nvidia GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (CUDA=11.8, Numba>=0.60.0) and a working MC/DC version >=0.10.0. Then,

#. Clone the harmonize repo: ``git clone https://github.com/CEMeNT-PSAAP/harmonize.git``
#. Install into the proper Python env: ``pip install -e .``

Operability should now be enabled. 

.. _install-amd-gpus:

.. rubric:: AMD GPUs

The prerequisites for AMD operability are slightly more complex and
require a patch to Numba to allow for AMD target triple LLVM-IR.
It is recommended that this is done within a Python venv virtual environment.

To compile and execute MC/DC on AMD GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (ROCm=6.0.0, Numba>=0.60.0) and a working MC/DC version >=0.11.0. Then,

#. Patch Numba to enable HIP (`instructions here <https://github.com/ROCm/numba-hip>`_)
#. Clone harmonize and `switch to the AMD <https://github.com/CEMeNT-PSAAP/harmonize/tree/amd_event_interop_revamp>`_ branch with ``git switch amd_event_interop_revamp``
#. Install Harmonize with ``pip install -e .`` or using `Harmonize's install script <https://github.com/CEMeNT-PSAAP/harmonize/tree/main>`_

Operability should now be enabled.
