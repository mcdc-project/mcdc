.. _batch_scripts:

=================
Batch Job Scripts
=================

This page provides ready-to-use batch script templates for running MC/DC on
HPC systems with the three most common job schedulers: Slurm, Flux, and LSF.

Each template follows the same workflow:

#. Load required modules.
#. Activate the Python environment.
#. Launch MC/DC with the appropriate MPI wrapper.

Adapt the resource requests (nodes, tasks, GPUs, wall-time, queue) to your
allocation and problem size.


Slurm
-----

`Slurm <https://slurm.schedmd.com/>`_ is widely used on LLNL's Quartz and Dane,
as well as many university and national lab clusters.

**CPU-only (Numba mode):**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=mcdc_run
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=36
   #SBATCH --time=01:00:00
   #SBATCH --partition=pbatch

   module load python/3.11
   source /path/to/your/venv/bin/activate

   srun python input.py --mode=numba --caching

**GPU (Nvidia, e.g., Lassen-like systems with Slurm):**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=mcdc_gpu
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-task=1
   #SBATCH --time=00:30:00
   #SBATCH --partition=gpu

   module load python/3.11 cuda/11.8
   source /path/to/your/venv/bin/activate

   srun python input.py --mode=numba --target=gpu --gpu_strategy=event


Flux
----

`Flux <https://flux-framework.org/>`_ is the scheduler on LLNL's Tioga and
El Capitan systems (AMD MI250X / MI300A GPUs).

**CPU-only:**

.. code-block:: bash

   #!/bin/bash

   module load cray-mpich python/3.11
   source /path/to/your/venv/bin/activate

   flux run -N 2 -n 72 python input.py --mode=numba --caching

**GPU (AMD MI300A, El Capitan):**

.. code-block:: bash

   #!/bin/bash

   module load cray-mpich rocm/6.0.0 python/3.11
   source /path/to/your/venv/bin/activate

   flux run -N 2 -n 8 -g 1 --queue=mi300a \
       python input.py --mode=numba --target=gpu \
       --gpu_arena_size=100000000 --gpu_strategy=event

This launches MC/DC on 2 nodes with 8 GPUs total (4 per node) on the MI300A partition.


LSF
---

`LSF <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0>`_ is used on LLNL's
Lassen (IBM POWER9 + Nvidia V100).

**CPU-only:**

.. code-block:: bash

   #!/bin/bash
   #BSUB -J mcdc_run
   #BSUB -nnodes 2
   #BSUB -W 60
   #BSUB -q pbatch

   module load gcc/8 cuda/11.8
   conda activate mcdc-env

   jsrun -n 8 -r 4 -a 1 -c 10 python input.py --mode=numba --caching

**GPU (Nvidia V100, Lassen):**

.. code-block:: bash

   #!/bin/bash
   #BSUB -J mcdc_gpu
   #BSUB -nnodes 1
   #BSUB -W 30
   #BSUB -q pbatch

   module load gcc/8 cuda/11.8
   conda activate mcdc-env

   jsrun -n 4 -r 4 -a 1 -g 1 \
       python input.py --mode=numba --target=gpu --gpu_strategy=async

This runs MC/DC on 1 node with 4 GPUs using the asynchronous scheduler.


Tips
----

* **Start small:** Test with a short wall-time and few particles before
  submitting large production runs.
* **Use caching:** Adding ``--caching`` saves Numba-compiled binaries so that
  subsequent runs skip the JIT compilation step.
* **Clear cache when updating MC/DC:** If you update the code, run once with
  ``--clear_cache --caching`` to regenerate binaries.
* **Check module order:** On some systems the order of ``module load`` commands
  matters. Load the compiler/MPI module before CUDA/ROCm.
* **Interactive debugging:** Request an interactive node first
  (``salloc``, ``flux alloc``, or ``lalloc``) to test your command before
  committing to a batch job.
