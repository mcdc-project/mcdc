# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
  - Combined `object_` and `transport` tests for more efficient fixture reuse
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
[0.14.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.0
[0.13.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.13.0
[@ilhamv]: https://github.com/ilhamv
[@melekderman]: https://github.com/melekderman
[@massimolarsen]: https://github.com/massimolarsen
[@nglaser3]: https://github.com/nglaser3
[@gunnarrl]: https://github.com/gunnarrl
[@Talen-Ayers]: https://github.com/Talen-Ayers
