# MC/DC: Monte Carlo Dynamic Code

![MC/DC logo](https://raw.githubusercontent.com/mcdc-project/mcdc/main/assets/mcdc-logo.svg)

[![Unit Tests](https://github.com/mcdc-project/mcdc/actions/workflows/unit_test.yml/badge.svg)](https://github.com/mcdc-project/mcdc/actions/workflows/unit_test.yml)
[![Regression Tests](https://github.com/mcdc-project/mcdc/actions/workflows/regression_test.yml/badge.svg)](https://github.com/mcdc-project/mcdc/actions/workflows/regression_test.yml)
[![Docker Build, Test, and Publish](https://github.com/mcdc-project/mcdc/actions/workflows/docker.yml/badge.svg)](https://github.com/mcdc-project/mcdc/actions/workflows/docker.yml)
[![Coverage Status](https://coveralls.io/repos/github/CARRE-PSAAP/mcdc/badge.svg?branch=dev)](https://coveralls.io/github/CARRE-PSAAP/mcdc?branch=dev)
[![Documentation Build](https://github.com/mcdc-project/mcdc/actions/workflows/docs_test.yml/badge.svg)](https://mcdc.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/mcdc.svg)](https://pypi.org/project/mcdc/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06415/status.svg)](https://doi.org/10.21105/joss.06415)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

MC/DC is an open-source, Python-based Monte Carlo radiation transport code that combines rapid methods development with scalable, high-performance execution on modern computing systems.
Originally initiated by the Center for Exascale Monte Carlo Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)), MC/DC development is now led by the Center for Advancing the Radiation Resilience of Electronics ([CARRE](https://carre-psaapiv.org/)).

## Features

- Monte Carlo neutron transport, with photon and charged-particle transport and coupled multi-particle capabilities under development
- Time-dependent, steady-state, and eigenvalue simulations
- Multiple physics fidelities (e.g., continuous-energy and multigroup neutron transport, with single-scattering and condensed-history charged-particle transport under development)
- Distributed-memory parallel execution with MPI
- Machine-portable Python implementation accelerated by Numba JIT compilation
- Extensible architecture for prototyping new transport methods

## Installation

MC/DC requires Python 3.11 or newer.

Install the latest stable release from PyPI:

```bash
pip install mcdc
```

For development installation and additional options, see the [Installation Guide](https://mcdc.readthedocs.io/en/dev/user_guide/getting_started/installation.html).

## Documentation

Complete documentation is available on [Read the Docs](https://mcdc.readthedocs.io/), including:

- [User Guide](https://mcdc.readthedocs.io/en/dev/user_guide/index.html)
- [Getting Started](https://mcdc.readthedocs.io/en/dev/user_guide/getting_started/index.html)
- [API Reference](https://mcdc.readthedocs.io/en/dev/reference/python_api/index.html)
- [Developer Guide](https://mcdc.readthedocs.io/en/dev/developer_guide/index.html)
- [Contributing](https://mcdc.readthedocs.io/en/dev/developer_guide/contributing/index.html)

## Citing

If you use MC/DC in published work, please cite the following article:

- J. P. Morgan et al. (2024). [*Monte Carlo / Dynamic Code (MC/DC): An accelerated Python package for fully transient neutron transport and rapid methods development*](https://doi.org/10.21105/joss.06415). *Journal of Open Source Software*, **9**(96), 6415.

## Reporting Bugs and Issues

To report bugs, request features, or ask questions, please open a [GitHub Issue](https://github.com/mcdc-project/mcdc/issues).
