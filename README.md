# MC/DC: Monte Carlo Dynamic Code

![MC/DC logo](https://raw.githubusercontent.com/mcdc-project/mcdc/main/assets/mcdc-logo.svg)

[![Build](https://github.com/mcdc-project/mcdc/actions/workflows/regression_test.yml/badge.svg)](https://github.com/mcdc-project/mcdc/actions/workflows/regression_test.yml)
[![Docker Build, Test, and Publish](https://github.com/mcdc-project/mcdc/actions/workflows/docker.yml/badge.svg)](https://github.com/mcdc-project/mcdc/actions/workflows/docker.yml)
[![Read the Docs](https://github.com/mcdc-project/mcdc/actions/workflows/docs_test.yml/badge.svg)](https://mcdc.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/mcdc.svg)](https://pypi.org/project/mcdc/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06415/status.svg)](https://doi.org/10.21105/joss.06415)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

MC/DC is an open-source, Python-based Monte Carlo radiation transport code that
combines rapid methods development with scalable, high-performance execution on
modern computing systems. Originally initiated by the Center for Exascale
Monte Carlo Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)), MC/DC
development is now led by the Center for Advancing the Radiation Resilience of
Electronics ([CARRE](https://carre-psaapiv.org/)).

## Features

- Monte Carlo neutron transport (with photon and charged-particle capabilities
  under development)
- Time-dependent, steady-state, and eigenvalue simulations
- Multiple physics fidelities (continuous-energy/multi-group,
  single-scattering/condensed-history)
- Distributed-memory parallel execution with MPI
- Machine-portable Python implementation accelerated by Numba JIT compilation
- Extensible architecture for rapid methods development and prototyping

## Installation

Install the latest stable release from PyPI:

```bash
pip install mcdc
```

For development installation and additional options, see the
[Installation Guide](https://mcdc.readthedocs.io/en/latest/install.html).

## Documentation

Complete documentation is available on
[Read the Docs](https://mcdc.readthedocs.io/), including:

- [Installation](https://mcdc.readthedocs.io/en/latest/install.html)
- [User Guide](https://mcdc.readthedocs.io/en/latest/user/index.html)
- [API Reference](https://mcdc.readthedocs.io/en/latest/pythonapi/index.html)
- [Contribution Guide](https://mcdc.readthedocs.io/en/latest/contribution/index.html)

## Citing

If you use MC/DC in published work, please cite one or more of the following
references as appropriate:

- **MC/DC Origins**
  I. Variansyah, et al. (2023). *Development of MC/DC: a performant,
  scalable, and portable Python-based Monte Carlo neutron transport code.*
  Proceedings of the ANS Mathematics & Computation Conference 2025,
  Niagara Falls, Canada.
  https://doi.org/10.48550/arXiv.2305.07636

- **MC/DC JOSS Article**
  J. Morgan, et al. (2024). *Monte Carlo / Dynamic Code (MC/DC): An
  accelerated Python package for fully transient neutron transport and rapid
  methods development.* Journal of Open Source Software, **9**(96), 6415.
  https://doi.org/10.21105/joss.06415

## Reporting Bugs and Issues

To report bugs, request features, or ask questions, please open a
[GitHub Issue](https://github.com/mcdc-project/mcdc/issues).
