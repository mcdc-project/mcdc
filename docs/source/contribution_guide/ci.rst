.. _ci:

.. highlight:: none

Continuous Integration
======================

We use `github actions <https://github.com/CEMeNT-PSAAP/MCDC/actions>`_ to host and run most of our CI tests and version release information.
We run pure python unit tests and regression testing in pure Python, pure Python + MPI, numba, numba + MPI, and numba+GPU+harmonize.
When running regression tests we compare small particle count outputs to saved files in the testing directory.
If the RNG seed has not changed the results should be deterministic.


GPU COE Machine
---------------

CEMeNT currently has a `CI machine <https://github.com/CEMeNT-PSAAP/MCDC/settings/actions/runners>`_ on OSU's campus administered by the college of engineering HPC folks to do export controlled and GPU continuous integration.
It has a single Nvidia A2 (16GB VRAM) and an AMD EPYC 7313P 16-Core Processor with 64 GBs of RAM.
This is a dedicated machine with no additional users other then CEMeNT staff.

To access the machine you have to be on OSU's VPN or on campus and ssh into

.. code-block:: bash

    ssh <ONID>@cement.hpc.engr.oregonstate.edu

If you do not have account access to that machine email to Rob Yelle (``robert.yelle@oregonstate.edu``) for support and ask to be added.
From there users can use SLURM or module system to load preinstalled software.

The standard dev env for MC/DC install on this machine can be ascertained with

.. code-block:: bash

    module load cuda/11.8 gcc/10.3 mpich/4.0h_gcc-10 python/3.11
    python -m venv <MCDC_venv>
    module unload python/3.11
    source <MCDC_venv>/bin/activate

Then MC/DC and harmonize can be installed there in the normal manner for GPU capabilities.
The runner runs all the time in background of Joanna's account.
Contact her with any issues or on instructions to set up your own runner!