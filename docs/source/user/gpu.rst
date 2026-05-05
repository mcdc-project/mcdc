
.. _gpu:

=====================
Running MC/DC on GPUs
=====================

MC/DC supports most of its Numba enabled features for GPU compilation.
When targeting GPUs execution MC/DC uses the Harmonize library to schedule events.
Harmonize acts as the GPU runtime for MC/DC and has two major scheduling schemes including a novel asynchronous event scheduler.
For more information on Harmonize and how we compile MC/DC with it see our publications in M&C 2025.

Single GPU Launches
-------------------

To run problems on the GPU evoke input decks with a ``--mode=numba --target=gpu`` option appended on the python command.
For example,
.. code-block:: sh

    python input.py --mode=numba --target=gpu

A cache folder will be generated in the same directory as the input deck titled ``__harmonize_cache__`` which contains the intermediate compiler representations and compiled biniaries.

MC/DC Harmonize Runtime Options
-------------------------------

At runtime the user can interface with the Harmonize scheduler that MC/DC uses as its GPU runtime.
Configurable options include:

#. Specifying scheduling modes with ``--gpu_strategy=`` either ``event`` (default) or ``async`` (only enabled for Nvidia GPUs) 
#. Declaring the GPU arena size (size of memory allocated on the GPU measured in particles) ``--gpu_arena_size= [int_value]`` 
#. Clearing the previous cache (and forcing recompilation) ``--clear_cache``
#. Requesting Harmonize to cache its results: ``--caching``
#. Clearing the previous cache and making a new one: ``--clear_cache  --caching``

Other configurable compile-time options are available in ` ``harmonize/python/config.py`` <https://github.com/CEMeNT-PSAAP/harmonize/tree/main>`_ starting on line 15.

#. Verbose compiler operations: ``VERBOSE = False/True``
#. Harmonize debug mode: ``DEBUG  = False/True``
#. Printing raw compiled errors: ``ERROR_PRINT = True/False``
#. Using color terminal printing ``COLOR_LOG = True/False``

MPI+GPU Operability
-------------------

Multi-GPU runs are enabled and require only to be dispatched with appropriate MPI calls.
The workflow for MPI+GPU calls is the same as with normal MPI calls and looks something like (assuming you are on an HPC): 

#. load modules
#. source python environment (either with conda or venv)
#. launch nodes (either interactive or batch)
#. evoke MPI calls

For example on an interactive node using SLURM it would look something like
.. code-block:: sh

    module load <pre_requests>
    source python_venv/bin/activate
    salloc -N 1 
    srun <srun_options> python mcdc_input.py <mcdc_options>

Or when using `flux <https://flux-framework.org/>`_ scheduler (the scheduler LLNL scheduler uses on `Tioga and El Capitan <https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems>`_) (assuming an interactive node ``salloc -n1``):
.. code-block:: sh

    flux run -N 2 -n 8 -g 1 --queue=mi300a python3 input.py --mode=numba --target=gpu --gpu_arena_size=100000000 --gpu_strategy=event

which launches event scheduled MC/DC on GPUs with a GPU arena 1e9 2 nodes with 8 GPUs total (4/node) on the MI300A partition.
An example of `LSF <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0>`_ scheduling (the scheduler LLNL uses on `Lassen <https://hpc.llnl.gov/documentation/tutorials/using-lc-s-sierra-systems>`_) assuming an interactive node (``lalloc 1``)
.. code-block:: sh

    jsrun -n 4 -r 4 -a 1 -g 1 python input.py --mode=numba --target=gpu --gpu_strategy=async

which launches async scheduled MC/DC on Nvidia GPUs with a GPU arena of 1e9 on 1 node with 4 GPUs total (4/node).

GPU Profiling
-------------

Pro
