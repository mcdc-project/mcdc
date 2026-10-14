# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as a guide.

## [Unreleased]

### Added

- Add photon transport with coherent and incoherent scattering, photoelectric absorption with atomic relaxation, and pair production, plus an energy-independent constant-cross-section treatment, from [@DouglasHouser]
- Add cone surfaces with a half-angle interface in degrees, specialized transport kernels for all cone orientations, and a specialized arbitrary-axis cylinder kernel, from [@ilhamv]
- Add proton transport with nuclear reactions and secondary-particle production, continuous slowing down, energy-loss straggling, and multiple Coulomb scattering, from [@ethan-lame]
- Add developer documentation on particle transport architecture, covering particle steps, event handling, interaction data, and tally scoring triggers, from [@ilhamv]
- Add time, polar-cosine, and azimuthal dependence to weight windows; consolidate lower, target, and upper weights into one array and extend flattened-data accessors to eight dimensions, from [@nglaser3]
- Add a fissionable material and an outlet detector to the pulsed Kobayashi example, from [@ilhamv]
- Add CI check requiring generated Numba support to match the rebuild script, from [@ilhamv]
- Add piece-wise linear spatial distribution for source definition, from [@ilhamv]
- Add overriding option N_active, from [@ilhamv]
- Add MC/DC-VVP project documentation with verification case narratives, published results, and VVP result-generation and publication steps in the release checklist, from [@ilhamv]
- Add GPU-compatible functions for 3D cross products and vector normalization, from [@braxtoncuneo]

### Changed

- Report per-file progress while loading native neutron, proton, and electron data, and include data-loading time in runtime reports and HDF5 output, from [@ilhamv]
- Reduce compilation time for data-heavy models by assigning polymorphic subtype IDs with constant-time counters, from [@ilhamv]
- Correctly interpret empty-interpolation to all-linear default in neutron transport data generator, from [@ilhamv]
- Represent stopping power with `DataTable` and use the shared data evaluator with optional endpoint clamping, from [@ilhamv]
- Centralize cross-species production in `physics.produce_cross_species`, respecting particle transport activation and providing a shared foundation for other incident particle types beyond protons, from [@ilhamv]
- Update Lockwood electron transport regression test answer (also adjust to lower N_particle), from [@ilhamv]
- Select among multiple particle sources using a simulation-finalized cumulative distribution and binary search instead of a per-particle linear scan, from [@ilhamv]
- Rename collision tally/data to interaction tally/data to cover both discrete collisions and condensed interactions; describe tally types by their transport scoring triggers rather than as estimators, from [@ilhamv]
- Use the full incident-particle state for collision tally filtering, from [@ilhamv]
- Make lower-energy-first transport rule configurable per particle, from [@ilhamv]
- Introduce per-species transport settings with automatic activation of source particle types and nested settings in output files, replacing `set_transported_particles`, from [@ilhamv]
- Generate the electron data library in eV, store the elastic transport cross section under MT-526, from [@melekderman]
- Transport the lower-energy electron from ionization first for more effective bank usage, from [@melekderman]
- Reuse electron cross-section energy-grid indices and cumulative subshell cross sections during collision sampling, from [@melekderman]
- Unify polar–azimuthal basis construction across angle conversion, source direction sampling, and scattering, with robust Z-pole handling; the shared convention changes seeded particle trajectories, from [@nglaser3] and [@ilhamv]
- Organize the Kobayashi examples under `examples/kobayashi-dogleg/` as `steady_state`, `pulsed`, and `pulsed_with_fission`; use the fission variant in the pulsed tutorial and clarify its relation to the original PNE benchmark and the Zenodo transient adaptation, from [@ilhamv]
- Add standard `performance/` output metrics and replace `--runtime_output` with `--no-tally_output` to omit tally results, from [@ilhamv]
- Filter out empty numba support accessors from creation, from [@ilhamv]
- Update GPU transport support for the current MC/DC data model and Harmonize runtime, including GPU-compatible state access, particle-bank operations, array accessors, and torus intersections, from [@braxtoncuneo]
- Show previously published documentation versions in the documentation version switcher, from [@ilhamv]
- Optimize tally moments memory allocation — only allocate to non-master rank if necessary, from [@ilhamv]

### Deprecated

### Removed

- Caching of the `souce_loop` function is removed for GPU execution, from [@braxtoncuneo]
- Caching of the `bank_active_particle` function, from [@braxtoncuneo]
- Remove empty mcdc_get and mcdc_set members, from [@ilhamv]

### Fixed

- Preserve very low particle energies when converting from speed by avoiding floating-point cancellation in the relativistic kinetic-energy calculation, from [@ilhamv]
- Prevent particles from becoming trapped at zero-distance crossings of quadratic surfaces, from [@ilhamv]
- Fix tallies not applying their particle-type filter during scoring, from [@melekderman]
- Fix collision tally energy filtering to use the incident energy, from [@melekderman]
- Fix electron-ionization energy sampling, change the policy from [@melekderman]
- Fix track-length tallies skipping tracks shorter than the time coincidence tolerance, which zeroed electron flux tallies, from [@melekderman]
- Restore the Lockwood regression case to the 1 MeV benchmark and regenerate its answer, from [@melekderman]
- Fix tabulated distribution sampling at the first CDF point, from [@melekderman]
- Fix multi-table distribution sampling when the incident energy equals the first grid point, which used the last grid point as the lower bound and could fail or return wrong values, from [@melekderman]
- Fix element densities when collapsing a nuclide composition into elements, which counted nuclides with a shared symbol prefix (e.g., `Cr52` as carbon, `He4` as hydrogen), from [@melekderman]
- Correct ACEtk electron data loading units, elastic cross-section assignments, and CDF dataset names, from [@massimolarsen]
- Improve tally variance accuracy with stable online statistics and parallel moment merging; require multiple batches for fixed-source time-census and GPU transport, from [@ilhamv]
- Fix UCX transport errors in MPI runs on the unit-test and Numba-support CI workflows by restricting `UCX_TLS` to `self,sm,tcp`, from [@melekderman]
- Fix the generation of dispatch function handles when compiling for GPU, from [@braxtoncuneo]
- Fix use of default parameter values for `evaluate_data` and `evaluate_table`, from [@braxtoncuneo]
- Fix use of `numba-hip`-incompatible np.log10 calls in proton condensed interactions, from [@braxtoncuneo]

### Security

## [0.15.3] - 2026-09-27

### Fixed

- Correct Zenodo release metadata and update software authors in `CITATION.cff`, from [@melekderman] and [@ilhamv]

### Added

- Include `cffconvert` in the development dependencies for release citation validation, from [@ilhamv]

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
[0.15.3]: https://github.com/mcdc-project/mcdc/releases/tag/v0.15.3
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
[@ethan-lame]: https://github.com/ethan-lame
[@DouglasHouser]: https://github.com/DouglasHouser
