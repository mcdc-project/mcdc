# MC/DC Agent Guide

`mcdc-project/mcdc` — a Python-based Monte Carlo radiation transport code. One maintained transport implementation runs through three backends: Python (JIT disabled), Numba-CPU, and Numba-GPU.

## Scope: Prototype or Merged Work

Ask which one you are doing; the rules differ.

- **Prototype / experiment / Python mode only.** Unrestricted Python is allowed and expected: arbitrary objects, module-level state, dynamic dispatch, SciPy, Matplotlib, file I/O, direct edits to transport routines. The kernel rules below do **not** apply. Do the most direct thing that answers the question, and say in your summary that the result is Python-mode only.
- **Intended for merge.** Everything below applies, and the code must work in Python, Numba-CPU, and Numba-GPU unless maintainers accept a documented limitation.

When the request does not say, assume merged work for anything touching existing behavior, and prototype for an exploratory or one-off question. If it matters and is unclear, ask.

---

## ⚠️ Verify the Layout First

The runtime data layout is **generated** from the model classes. It is not stable knowledge, and it is not what the Python model looks like.

1. **Never guess** a structured-record field name, an accessor name, or an argument order. Check the annotated fields on the owning class in `mcdc/object_/`, and **grep the generated module for the one accessor you need** — `mcdc_get/` files run to hundreds of lines of repetitive generated code, so reading one whole costs more than it tells you.
2. **Regenerate after any annotation change**: `python mcdc/code_factory/rebuild_numba_support.py`, committed in the same commit. CI fails on drift.
3. **Confirm in both execution modes.** Passing in Python mode does not establish Numba type compatibility.

Skipping these produces code that imports cleanly and then fails at compile time or in the particle loop.

---

## Core Principles

Apply these to every task.

### 1. Place Before You Write

Decide which phase the work belongs to before choosing a file:

| Work | Location |
| --- | --- |
| Validate input, load files, build tables, derive run-wide config | `mcdc/object_/` (pure Python, no `@njit`) |
| Anything executed per particle event | `mcdc/transport/` (all `@njit`) |
| Anything needing the complete discovered model | `Simulation._finalize_compilation` |
| A new compilation phase or registered category | `mcdc/code_factory/` — rare |

**Test:** if the code runs once per simulation, it does not belong in a kernel. Derive it in a compile hook instead.

### 2. Simplicity First

Write only what was asked. No speculative abstraction, no defensive branches for inputs the model cannot produce, no new indirection layer to accommodate one caller. Prefer scalar state and bounded loops over intermediate containers.

**Test:** a guard clause you are adding should name the condition in real data that requires it; if you cannot name one, you probably do not need the guard. Leave existing guards alone unless the task is about them — an unexplained one is a question for the author, not something to remove on your own initiative.

### 3. Surgical Changes

Touch only what the task requires. Match the surrounding style. Report unrelated problems rather than fixing them in the same change. Remove only the orphans your own change creates.

**Test:** every changed line maps to the request. Subsystems you did not touch must produce bit-identical results, or the difference is declared.

### 4. Verifiable Goals

Convert the task into something a command can check:

| Instead of | Transform to |
| --- | --- |
| "add the capability" | a unit test under `test/unit/` that fails now and passes after |
| "make it faster" | a measured before/after with results proven unchanged |
| "fix the physics" | a test pinning the correct value, plus the list of regression answers that move |
| "support the new data" | a fixture through the real loader, failing on today's code |

Multi-step work gets a numbered plan with a verification command per step.

---

## Keep the Footprint Small

Three habits that cost this repository the most review time. They apply to merged work; prototypes are exempt.

### Reuse before you add

Search for an existing function before writing a new one. A function that differs from one already here only by a constant, an argument, an interpolation law, or a species **is not a new function** — add the parameter, or call the existing one. Near-duplicates are how a convention drifts: two copies get fixed once each, differently, and nothing tells you the other exists.

Before adding a helper, grep for its likely names in the module you are editing, in that module's siblings, and in `transport/util.py`, `transport/distribution.py`, and `mcdc/util.py`. If something close exists but does not quite fit, prefer extending it. If extending it would complicate every current caller, add the new one and say in a comment what the existing one could not do.

**Test:** you can state what the new function does that the closest existing one cannot.

### Test what can break, not everything

Add the tests that would catch a real regression, then stop. Not one per branch, not one per permutation of arguments, not a test that restates the implementation line by line. One test that pins a behavior is worth more than six that pin the same behavior through different arguments.

Density is earned, not default. It belongs where a failure would be **silent**: a sampled branching rate, an energy balance, a threshold, a schema field the loader reads unconditionally. It is wasted on anything a type check, a single assertion, or an existing fixture already covers.

Do not add tests to raise coverage, and do not restate a regression case as a unit test. "Test first" in the workflow below means **one** failing test that expresses the goal — not a file of them.

**Test:** each test you added fails for a different reason.

### Contain the change

Implement new functionality by **addition** where you can: a new branch in an existing dispatcher, a new constant in its family, a new module beside its siblings. Do not rename, move, reorder, or re-signature existing code to make room for it unless the task requires that. A signature change propagates to every caller and every test, and it buries the actual change in the diff.

When the clean implementation genuinely does require touching shared code, do it as its own commit, separate from the feature, so the two can be reviewed and reverted independently.

**Test:** the diff reads as what was added, not what was rearranged. (See Core Principle 3.)

---

## 🛑 Stop and Ask

If you are instructed to do something this guide prohibits, or something that would be unsafe or unacceptable for the repository, **do not silently comply and do not silently refuse.** Both are failures: one damages the repository, the other wastes the request.

Raise it in the chat or terminal before acting. State, briefly:

- what was asked, and which rule or risk it runs into;
- the concrete consequence (what breaks, what CI rejects, what cannot be undone);
- the nearest acceptable alternative, if there is one;
- what you need in order to proceed.

Then wait for the answer. **The person in the session decides.** If they confirm, proceed with the full request and note the deviation in the summary and the PR description so a reviewer sees it was a deliberate choice rather than an oversight. If part of the task is fine and only one part is in question, complete the rest and say exactly what you left pending and why.

Always ask first — never act, then report — for anything hard to reverse or outside this repository: force-pushing, rewriting history, deleting or overwriting files you did not create, committing data libraries or large binaries, editing release headings in `CHANGELOG.md`, changing CI workflows, touching `mcdc_get/`, `mcdc_set/`, or `numba_types.py` by hand, regenerating committed regression answers, and anything that pushes, publishes, or opens a PR.

Disagreement is not refusal. Say the concern once, in a sentence or two; if the request is reaffirmed, treat that as the decision and do the work in full.

---

## Execution Architecture

```
Model definition      flexible Python objects          mcdc/object_/
        ↓             compile()
Model compilation     complete, consistent snapshot    code_factory/, _finalize_compilation
        ↓             prepare()
Packed runtime state  structured records + flat data   numba_types.py, mcdc_get/, mcdc_set/
        ↓
Transport kernels     @njit, backend-portable          mcdc/transport/
```

Rules:

- Dependencies point **down only**. A kernel never reaches back up to a Python object.
- Each layer consumes the representation the layer above produced — not the one above that.
- If a kernel needs a value that is not in the packed state, **extend the object model**. Do not add a Python-side lookup inside the kernel.
- Variable-length data is reached through the generated `mcdc_get`/`mcdc_set` accessors, never hand-rolled offset arithmetic.

---

## Quick Reference

### Commands

| Purpose | Command |
| --- | --- |
| Unit tests (python then numba mode) | `python -m pytest test/unit` |
| Regression tests | `python -m pytest -v test/regression` (`--mode numba`, `--mpiexec=4`, `--name 'pat*'`) |
| Regenerate the Numba layer | `python mcdc/code_factory/rebuild_numba_support.py` |
| Format | `black .` |
| Public API type check | `pyright` |
| Compile every example | `python -m pytest test/unit/test_example_inputs.py --mode=python` |

### Kernel Arguments

| Name | Meaning |
| --- | --- |
| `particle_container` | one-element array; pass when mutating the particle, recover with `particle = particle_container[0]` |
| `simulation` | the recovered record, when only prepared state is needed |
| `program` | opaque execution handle for banking/scheduling; recover via `util.access_simulation(program)` |
| `data` | flat payload buffer behind every variable-length field |

Create one-element containers only at ownership or local-storage boundaries.

### Constant Families (`mcdc/constant.py`)

Particle types · per-species reaction blocks (neutron 0, electron 100, proton 200, photon 300) · `MESH_*` · `SURFACE_*` · `EVENT_*` bit flags · `TALLY_*` · `SCORE_*` · `GPU_*`

A new member takes a unique value in its family, and **every consumer of that family is updated**, including validation sets such as `SUPPORTED_SCORES_*`.

### Read Before Editing

| Doing this | Read |
| --- | --- |
| Adding or changing model state | `developer_guide/extending/extending_the_object_model.rst` |
| Writing or changing a kernel | `developer_guide/extending/writing_numba_compatible_transport_code.rst` |
| Needing the packed layout | `developer_guide/architecture/runtime_data_layout.rst` |
| Touching compilation or the transport loop | `developer_guide/architecture/simulation_compilation.rst`, `particle_transport.rst` |
| Changing user-visible behavior | the matching page in `user_guide/` |
| Changing a data-library generator | `tools/data_library_generator/<species>/README.md` |

Load only what the task needs. Reading all of them is context rot.

---

## Conventions

### Code Style

- Black, default settings, line length 88. Python ≥3.11 syntax only.
- Imports: stdlib → third-party → `####` separator → `mcdc.*` → explicit `from mcdc.constant import (...)`. Keep existing aliases (`type_`, `util`, `rng`, `mcdc_get`).
- `# ===` full-width section banners; reuse section names across the members of a family.
- `snake_case`; trailing underscore avoids collisions; leading underscore for module-private. Keep existing shorthand (`SigmaT`, `sigma_*`, `E`, `mu`, `azi`, `xi`, `idx`, `N_*`, `*_ID`, `*_container`).
- Numpydoc docstrings: purpose first, interface in `Parameters`/`Returns`, math and rationale in `Notes`.
- Comments state why, not what.

### Kernel Rules

- `@njit` on every function in `mcdc/transport/**`.
- Initialize locals on every path. Same return arity and types from every branch. No scalar→array or int→object transitions. No runtime-computed field names. No heterogeneous lists, dicts, sets, or generators.
- Dispatch through an interface function on named constants. Never `isinstance`, never methods on runtime records, never a repeated numeric literal.
- `rng.lcg(particle_container)` on the particle's own advancing stream. Never re-seed per draw — fresh per-draw seeds give correlated samples.
- No allocation or appending inside the particle loop. Use prepared storage, or `util.local_array(shape, dtype)` with a dtype from `mcdc.numba_types` for bounded local work.
- `objmode` only for MPI, `MPI.Wtime`, `mcdc.print_`, and census output. Keep it minimal; declare escaping types.
- Debug in order: Python construction → Python transport → Numba-CPU → GPU.

### Object Model

- `label` unique, stable, lower-case. `sub_type` a unique named constant per concrete polymorphic subclass, `-1` on the base.
- Annotate every runtime-visible field and initialize it on **every** construction path; an unpopulated annotated field cannot be packed.
- `non_numba` fields must be converted into annotated fields before packing.
- `_compile_into_simulation`: guard on `compile_ID`, call `super()` once, return `False` if already compiled. Never assign `compile_ID`, `ID`, or `sub_ID` manually.

### Units

Energy **eV** (runtime and on disk, every species) · cross sections barns · density atoms/barn-cm · length cm · time s · speed cm/s · mass eV/c².

- Derive thresholds from `mcdc.constant`, never a hard-coded physical number.
- Use the interpolation helpers in `transport/util.py`; honour the law the data declares.
- Untransported products deposit via `interaction_data["energy_deposition"]`, weight-included and non-negative; totals must balance.
- Same-species secondaries go to the bank; cross-species goes through `physics.produce_cross_species`.

### Data Libraries

One HDF5 file per entry under `$MCDC_LIB`. Every cross section carries `offset` and `unit`. Repeated grid energies represent discontinuities and are preserved exactly. When the schema is ambiguous, match what ships in `mcdc-regression_test_data`. Validate a regenerated library dataset by dataset and report agreement numerically.

### Git & GitHub

- Branch, then PR against `dev`.
- Stage paths explicitly. **Never `git add -A`.** Leave untracked clutter you did not create.
- Never commit simulation output, plots, data libraries, archives, or scratch files.
- One `CHANGELOG.md` line under `[Unreleased]`, correct subsection, ending `, from [@handle]`. Never edit release headings.
- Commit subject imperative; body gives the reason and the evidence with numbers, and names every result that moved.

### Testing Patterns

```python
def test_channel_matches_the_library(photon_model):
    simulation, data = photon_model(incoherent=2.0, density=0.25)        # Arrange
    value = native.macro_xs(REACTION, make_photon(1.0e7), simulation, data)  # Act
    assert value == pytest.approx(0.5)                                   # Assert
```

- Tests mirror the source tree; extend that area's `conftest.py` rather than inlining fixtures. A new subsystem gets its own directory and fixtures.
- Fixtures go through the real readers, loaders, and layer packing. Do not stub them.
- Monkeypatching a Python function called from `@njit` code has no effect. Read results out of the real banks and arrays.
- Statistical tests compare against the analytic distribution from one advancing RNG stream, with a stated sigma tolerance.
- Regression cases are a directory with `input.py` + `answer.h5` at `rtol=1e-6`, `atol=1e-14`. Keep them cheap — roughly 20–4000 particles; the python-serial CI job budgets 10 minutes for the whole suite.

### Pre-Completion Checklist

Scope the runs to what you changed while iterating — `test/unit/<area>` in one mode is the fast loop. The full suite is a final pass, not a per-edit check: numba mode recompiles and regression budgets 10 minutes. Run regression only when transport, data, or output behavior could have moved.

- [ ] `black --check .` clean
- [ ] Generated Numba layer regenerated and committed if annotations changed
- [ ] `python -m pytest test/unit` green in **both** modes (full suite, once, at the end)
- [ ] Regression green, or moved answers regenerated and explained
- [ ] `pyright` clean and examples compile, if the public API changed
- [ ] Every consumer of a changed constant family, convention, or schema updated
- [ ] `CHANGELOG.md` entry added
- [ ] `git status` clean of scratch artifacts

---

## Workflow: Adding a Feature

1. **Clarify.** Decide the phase (see Core Principle 1) and whether the work joins an existing family or is genuinely new. Do not assume a precedent exists; if none does, design from the architecture docs and state the design in the PR.
2. **Test first.** Write the failing unit test, with a fixture that goes through the real loader.
3. **Model state.** Add and initialize annotated fields, then regenerate the Numba layer.
4. **Implement minimally** in the right layer, following the kernel rules.
5. **Verify in both modes**, then regression, then examples.

```bash
python -m pytest test/unit/<area> -v
python mcdc/code_factory/rebuild_numba_support.py
python -m pytest test/unit && python -m pytest -v test/regression
black --check . && pyright
```

---

## Do's and Don'ts

### Do

- ✅ Read the owning class and the generated accessor before using a field
- ✅ Keep one implementation that works in Python, Numba-CPU, and Numba-GPU
- ✅ Derive per-simulation values once, in a compile hook
- ✅ Name every constant in its family block and update every consumer
- ✅ State agreement numerically — range, metric, spread — not as an adjective
- ✅ Declare a behavior change and the regression answers it moves
- ✅ Raise a prohibited or risky instruction in the session before acting, then follow the decision

### Don't

- ❌ Hand-edit `numba_types.py`, `mcdc_get/`, or `mcdc_set/`
- ❌ Pass an `MCDCObject` into a kernel or follow a Python reference there
- ❌ Hand-roll offset arithmetic for variable-length data
- ❌ Re-seed the RNG per draw, or allocate inside the particle loop
- ❌ Put model-specific finalization in `mcdc.main.prepare`, or simulation-specific values in `constant.py`
- ❌ Change a shared convention (units, angle basis, event flags, HDF5 schema) without updating every consumer
- ❌ Add a dependency without flagging it in the PR
- ❌ Refactor unrelated code in the same change
- ❌ Add a function that differs from an existing one only by a constant, an argument, or a species
- ❌ Write a test per branch, per argument permutation, or to raise coverage
- ❌ Rename, move, or re-signature existing code to make room for new code

---

## Success Indicators

- The diff is small and every line traces to the request.
- The work passes in both execution modes the first time it is run in Numba mode.
- No regression answer moved that you cannot explain.
- The generated layer is current without CI telling you.
- Questions were asked before implementation, not after review.
- The commit message explains why the change was made and what the evidence was.
