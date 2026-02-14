# MC/DC: Monte Carlo Dynamic Code

![mcdc_logo v1](https://user-images.githubusercontent.com/26186244/173467190-74d9b09a-ef7d-4f0e-8bdf-4a076de7c43c.svg)

[![Build](https://github.com/CEMeNT-PSAAP/MCDC/actions/workflows/regression_test.yml/badge.svg)](https://github.com/CEMeNT-PSAAP/MCDC/actions/workflows/regression_test.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06415/status.svg)](https://doi.org/10.21105/joss.06415)
[![ReadTheDocs](https://github.com/CEMeNT-PSAAP/MCDC/actions/workflows/docs_test.yml/badge.svg)](https://mcdc.readthedocs.org/en/latest/ )
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

MC/DC is a performant, scalable, and machine-portable Python-based Monte Carlo 
neutron transport software, initiated by the Center for Exascale Monte Carlo 
Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)), and currently 
in active development in the Center for Advancing the Radiation Resilience of 
Electronics ([CARRE](https://carre-psaapiv.org)).

[Read the Docs](https://mcdc.readthedocs.io/en/latest/) for documentation on installation, contribution, and user guide.

## Installation

We recommend using [Python virtual environments (venv)](https://docs.python.org/3/library/venv.html) or some other environment manager (e.g. conda) to manage the MC/DC installation.
This avoids the need for admin access when installing MC/DC's dependencies and allows greater configurability for developers.
For most users working in a venv, MC/DC can be installed via pip:
```bash
pip install mcdc
```
For developers or users on HPC machines, mpi4py is often distributed as part of an HPC machines given venv.

### Common issues with `mpi4py`

The `pip mpi4py` distribution commonly has errors when building due to incompatible local MPI dependencies it builds off of. While pip does have some remedy for this, we recommend the following:
* **Mac users:** we recommend `openmpi` is [installed via homebrew](https://formulae.brew.sh/formula/open-mpi) (note that more reliable mpi4py distribution can also be [found on homebrew](https://formulae.brew.sh/formula/mpi4py)), alternatively you can use `conda` if you don't have admin privileges;
* **Linux users:** we recommend `openmpi` is installed via a root package manager if possible (e.g. `sudo apt install openmpi`) or a conda distribution (e.g. `conda install openmpi`)
* **HPC users and developers on any system:** On HPC systems that do not supply a suitable venv, `mpi4py` may need to be built using the system's existing `mpi` installation. Installing MC/DC using the [install script](https://mcdc.readthedocs.io/en/latest/install.html) we've included will handle that for you by installing dependencies using conda rather than pip.

## Running

MC/DC can be executed in different modes: via pure python or via a `jit` compiled version (Numba mode). 
Both modes have their use cases; in general, running in Numba mode is faster but more restrictive than via pure python.

### Pure Python

To run a hypothetical input deck (for example this [slab wall problem](https://github.com/CEMeNT-PSAAP/MCDC/tree/main/test/regression/slab_absorbium)) in pure python mode run:

```bash
python input.py
```

Simulation output files are saved to the directory that contains `input.py`.

### Numba mode

MC/DC supports transport kernel acceleration via 
[Numba](https://numba.readthedocs.io/en/stable/index.html)'s Just-in-Time 
compilation (currently only the CPU implementation). The *overhead* time for compilation
when running in Numba mode is about 15 to 80 seconds, depending on the physics and features 
simulated. Once compiled, the simulation runs **MUCH** faster than in 
Python mode.

To run in Numba mode:

```bash
python input.py --mode=numba
```

### Running in parallel

MC/DC supports parallel simulation via 
[MPI4Py](https://mpi4py.readthedocs.io/en/stable/). As an example, to run on 36 
processes in Numba mode with [SLURM](https://slurm.schedmd.com/documentation.html):

```bash
srun -n 36 python input.py --mode=numba
```

For systems that do not use SLURM (*i.e.*, a local system) try `mpiexec` or `mpirun` in its stead.

## Contributions

We welcome any contributions to this code base.
Please keep in mind that we do take our [code of conduct](https://github.com/CEMeNT-PSAAP/MCDC/blob/main/CODE_OF_CONDUCT.md) seriously.
Our development structure is fork-based: a developer makes a personal fork of this repo, commits contributions to their personal fork, then opens a pull request when they're ready to merge their changes into the main code base. Their contributions will then be reviewed by the primary developers. For more information on how to do this, see our [contribution guide](https://mcdc.readthedocs.io/en/latest/contribution.html).

## Bugs and Issues

Our documentation is in the early stages of development, so thank you for bearing with us while we bring it up to snuff.
If you find a novel bug or anything else you feel we should be aware of, feel free to [open an issue](https://github.com/CEMeNT-PSAAP/MCDC/issues).

## Testing

MC/DC uses continuous integration (CI) to run its unit and regression test suite. 
MC/DC also includes verification and performance tests, which are built and run nightly on internal systems.
You can find specifics on how to run these tests locally [here](https://github.com/CEMeNT-PSAAP/MCDC/tree/main/test/regression).

## Cite

To provide proper attribution to MC/DC, consider the following:

- MC/DC Origins:

  I. Variansyah, et al. (2023). Development of MC/DC: a performant, scalable, and portable Python-based Monte Carlo neutron transport code. Proc. ANS M&C 2025, Niagara Falls, Canada. https://doi.org/10.48550/arXiv.2305.07636.

- MC/DC JOSS article:

  J. Morgan, et al. (2024). Monte Carlo / Dynamic Code (MC/DC): An accelerated Python package for fully transient neutron transport and rapid methods development. Journal of Open Source Software, 9(96), 6415. https://doi.org/10.21105/joss.06415.
