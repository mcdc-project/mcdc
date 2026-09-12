# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as a guide.

## [Unreleased]

### Added

- Add CI check requiring generated Numba support to match the rebuild script, from [@ilhamv]
- Add piece-wise linear spatial distribution for source definition, from [@ilhamv]
- Add overriding option N_active, from [@ilhamv]
- Add MC/DC-VVP project documentation with verification case narratives, published results, and VVP result-generation and publication steps in the release checklist, from [@ilhamv]

### Changed

- Add standard `performance/` output metrics and replace `--runtime_output` with `--no-tally_output` to omit tally results, from [@ilhamv]
- Filter out empty numba support accessors from creation, from [@ilhamv]
- Update GPU transport support for the current MC/DC data model and Harmonize runtime, including GPU-compatible state access, particle-bank operations, array accessors, and torus intersections, from [@braxtoncuneo].
- Show previously published documentation versions in the documentation version switcher, from [@ilhamv]
- Optimize tally moments memory allocation — only allocate to non-master rank if necessary, from [@ilhamv]

### Deprecated

### Removed

- Caching of the `souce_loop` function is removed for GPU execution, from [@braxtoncuneo].
- Remove empty mcdc_get and mcdc_set members, from [@ilhamv]

### Fixed

- Improve tally variance accuracy with stable online statistics and parallel moment merging; require multiple batches for fixed-source time-census and GPU transport, from [@ilhamv]

### Security

## [0.15.2] - 2026-08-15

### Fixed

- Anticipate empty census-based tallies in batch runs for correct tally recombination, from [@ilhamv]
- Prevent independent runs with different problem sizes from sharing mutable problem-dependent Numba types, and preserve Python-managed `__pycache__` directories during startup, from [@ilhamv]

### Added

- Add `rebuild_numba_support.py` and the `-r`/`--rebuild` developer option for regenerating Numba support (mcdc_get, mcdc_set, numba_types.py) after object model changes, from [@ilhamv]

### Changed

- Generate shared Numba support independently of simulation preparation and create problem-dependent dtypes locally through pure factories, from [@ilhamv]
- Organize example documentation under the User Guide and contribution documentation under the Developer Guide, from [@ilhamv]

## [0.15.1] - 2026-08-12

### Fixed

- Prevent unbounded dependency resolution from selecting incompatible releases that break MC/DC by adding explicit upper bounds for all build, runtime, documentation, and development dependencies, from [@ilhamv]

### Added

- Add a dedicated CARRE project page, from [@ilhamv]
- Add the release policy and checklist, from [@ilhamv]
- Add the one-sentence-per-line convention for documentation source, from [@ilhamv]

### Changed

- Hide flyout in Read the Docs, from [@ilhamv]

## [0.15.0] - 2026-08-11

### Added

- Add `NeutronMultigroupData` and hybrid neutron multigroup transport with material-local physical energy grids and energy-representation policies, together with the `Material.multigroup()` convenience interface, from [@ilhamv]
- Add manually triggered unit and serial regression compatibility testing for Python 3.11, 3.12, and 3.14 while retaining Python 3.13 for automatic testing, from [@ilhamv]
- Add PEP 561-compatible inline type information and strict Pyright checks for the public Python API, from [@ilhamv]

### Changed

- **Breaking:** Unify native and neutron multigroup materials under one non-polymorphic `Material` and runtime layout, replacing `MaterialMG` and the separate native and multigroup material structures, from [@ilhamv]
- Group transport techniques under `simulation.technique`, from [@ilhamv]
- **Breaking:** Encapsulate MC/DC model building and execution within an explicit `mcdc.Simulation` instance, replacing the global simulation state and interface from [@ilhamv]
- Modernize and reorganize the documentation around distinct user, API reference, theory, project, and developer paths; adopt the PyData Sphinx Theme; and expand the architecture and extension guidance from [@ilhamv]
- Move model-specific finalization into simulation compilation and reserve runtime preparation for framework-level packing and execution setup from [@ilhamv]
- Move regression tests from the custom `run.py` harness to pytest-based collection and reporting from [@massimolarsen]
- **Breaking:** Require Python 3.11 or newer and designate Python 3.13 as the primary automatic test version, from [@ilhamv]
- Run Black and Pyright with Python 3.14 while keeping Black output compatible with every supported Python version, from [@ilhamv]
- Adopt a three-month seasonal cycle for minor releases while continuing to publish patch releases as needed, from [@ilhamv]

### Removed

- Remove the legacy `install.sh` installation helper, from [@ilhamv]

## [0.14.2] - 2026-07-15

### Fixed

- Fix 2D-vector setter writes nothing (- instead of =) from [@steps-re]
- Fix delayed neutrons are never sampled (transport/physics/neutron/native.py, fission()) from [@steps-re]
- Fix delayed emission time uses β instead of λ (transport/physics/neutron/native.py, fission())from [@steps-re]
- Fix swapped transverse-basis branches (transport/distribution.py, sample_direction()) from [@steps-re]
- Fix divide-by-zero for a -z reference (transport/distribution.py, sample_white_direction()) from [@steps-re]
- Fix tally polar_reference corrupted (object_/tally.py) from [@steps-re]

### Added

- Add layered documentation philosophy
- Tally spatial filter and scoring upgrades from [@massimolarsen] and [@ilhamv]
  - Add partial current scores
  - Add cell-filtered (and surface-cell-combo) support for current scores
  - Improved checks and error messages in tally-building user interface

### Changed

- Unit test upgrades
  - Combined `object_` and `transport` unit test for more efficient fixture reuse from [@massimolarsen]
  - Replace bare assert np.isclose with proper np.testing.assert_allclose from [@steps-re]

## [0.14.1] - 2026-07-04

### Fixed

- Documentation and packaging metadata fixes

## [0.14.0] - 2026-07-04

### Added

- CHANGELOG.md
- Data library generator upgrade from [@melekderman]
  - Electron data based on EPRDATA14 ACE-format
  - Improved organization for multi-particle data

### Changed

- **Breaking:** Redesigned the tabulated data infrastructure from [@ilhamv] and [@melekderman].
  - Added support for histogram, linear, semilog-x, semilog-y, and log-log interpolation
  - Added optional auxiliary arrays for helper data (e.g., CDFs in distributions)
  - Expanded use throughout distributions and reactions
- Unit test upgrade from [@massimolarsen]
  - Migrated from partial use of pytest to a fully pytest-based test suite
- Documentation updates
  - `mcdc.Source` documentation from [@ilhamv]
- Minor README and pull request template updates
- GitHub workflow for automatically marking stale issues
- Update CITATION.cff
- Automatic version updates on README, Sphinx docs, and pyproject. Version checker on CITATION

### Fixed

- Docker compatibility issue from [@melekderman]

## [0.13.0] - 2026-06-12

This release marks the final development phase under CEMeNT and the beginning of development under CARRE. A significant refactor was completed to improve ease of use, maintainability, and extensibility, restructuring the codebase to support future features and capabilities.

Major refactoring included:

- implementation of `code_factory`, which generates Numba- and GPU-compatible data structures from Python class objects; and
- reorganization of functions into a module-based architecture with well-defined interfaces.

GPU support is currently being updated to match the refactored architecture. The pre-refactor implementation, available in the `cement` branch, retains full GPU support. Complete GPU support for the refactored codebase (including both AMD and NVIDIA GPUs) is targeted for v0.15.0 or v0.16.0.

### Added

- `code_factory` from [@ilhamv]
- ACEtk-based data library generator from [@ilhamv]
- Multi-particle transport and physics model interfaces from [@ilhamv]
- ACE-based continuous-energy neutron physics from [@ilhamv]
- Element data library for electron transport from [@melekderman]
- Single-scattering electron transport from [@melekderman]
- Axis-aligned torus surfaces from [@Talen-Ayers]
- General cylinder surface from [@melekderman]
- Axis-aligned cone surfaces from [@melekderman]
- Docker support from [@melekderman]
- Pull request and issue templates from [@nglaser3]
- Unit tests for distribution sampling from [@massimolarsen]

### Changed

- Redesigned weight window implementation from [@nglaser3]
- Numba-optimized visualizer from [@gunnarrl]
- Documentation updates from [@melekderman]

### Removed

The following features were temporarily removed during the refactor:

- Domain decomposition
- iQMC
- UQ
- Compressed sensing
- Branchless collision
- Derivative Source Method
- Initial Condition bank

The pre-refactor implementation remains available in the `cement` branch as a reference for these features. They will be reintroduced incrementally in future releases.

### Fixed

- Multi-table distribution table selection sampling from [@melekderman]

[Unreleased]: https://github.com/mcdc-project/mcdc/tree/dev
[0.15.2]: https://github.com/mcdc-project/mcdc/releases/tag/v0.15.2
[0.15.1]: https://github.com/mcdc-project/mcdc/releases/tag/v0.15.1
[0.15.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.15.0
[0.14.2]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.2
[0.14.1]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.1
[0.14.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.0
[0.13.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.13.0
[@ilhamv]: https://github.com/ilhamv
[@melekderman]: https://github.com/melekderman
[@massimolarsen]: https://github.com/massimolarsen
[@nglaser3]: https://github.com/nglaser3
[@gunnarrl]: https://github.com/gunnarrl
[@Talen-Ayers]: https://github.com/Talen-Ayers
[@steps-re]: https://github.com/steps-re
[@braxtoncuneo]: https://github.com/braxtoncuneo
