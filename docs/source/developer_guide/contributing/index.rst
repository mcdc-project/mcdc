.. _contributing:

============
Contributing
============

This guide describes the repository workflow and quality checks for contributing to MC/DC.
It is intended for both occasional contributors and project maintainers.

Start with the setup steps below.
Use :doc:`continuous_integration` to understand automated checks and :doc:`container_development` when developing in the project container.
Use :doc:`example_validation` when changing the public API or example problems.
Read :doc:`pull_requests` before preparing a contribution.
For software architecture and documentation practices, see the :doc:`../index`.

For implementation guidance specific to compiled transport functions, see :doc:`../extending/writing_numba_compatible_transport_code`.

Contributions target the ``dev`` branch.
Prepare a development checkout with the following steps:

#. Fork ``mcdc-project/mcdc`` to your GitHub account.
#. ``git clone git@github.com:<YOUR_GITHUB>/mcdc.git``
#. ``git switch dev``
#. Create and activate a Python 3.14 environment for contributor tooling.
#. ``python -m pip install -e ".[dev]"``

Development Workflow
--------------------

.. toctree::
   :maxdepth: 1

   continuous_integration
   container_development
   example_validation
   pull_requests

MC/DC documentation is an important part of the project and evolves alongside the codebase.
The :doc:`../documentation/index` guide describes the documentation philosophy, writing guidelines, and the tools used to build and maintain the documentation.

Please note our `code of conduct <https://github.com/mcdc-project/mcdc/blob/dev/CODE_OF_CONDUCT.md>`_, which we take seriously.

------------
Code Styling
------------

MC/DC uses the `Black code style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_.
Run Black with Python 3.14 from the repository root before submitting a contribution:

.. code-block:: sh

    black .

Black is included in the ``dev`` optional dependency group installed during development setup.
Black formats for every supported Python version listed in ``pyproject.toml``.

Public API Typing
-----------------

MC/DC ships inline type information for its public Python interface.
Run Pyright from the repository root after changing a public class, annotation, or export:

.. code-block:: sh

    pyright

The strict public API checks are defined in ``test/typecheck/public_api.py``.
Pyright checks the public API against Python 3.14.

---------
Debugging
---------

MCDC includes options to debug the Numba JIT code.
It does this by toggling Numba options using the numba.config submodule.
This will result in less performant code and longer compile times but will allow for better error messages from Numba and other packages.
`See Numba documentation of a list of all possible debug and compiler options. <https://numba.readthedocs.io/en/stable/reference/envvars.html#debugging>`_
The most useful set of debug options for MC/DC can be enabled with

.. code-block:: python3

    python input.py --mode=numba_debug

Which will toggle the following debug and compiler options in Numba:

* ``DISABLE_JIT=False`` turns on the jitter
* ``NUMBA_OPT=0`` forces the compilers to form unoptimized code, while values ``1``, ``2``, and ``3`` enable increasing optimization.
  Change this option when an error appears only at higher optimization levels.
* ``DEBUG=False`` controls all debugging options.
  MC/DC leaves this disabled in ``numba_debug`` because it produces extensive terminal output.
* ``NUMBA_FULL_TRACEBACKS=1`` allows errors from sub-packages to be printed (i.e. Numpy)
* ``NUMBA_BOUNDSCHECK=1`` makes Numba check vectors for bounds errors.
  Without this check, a bounds error can result in a segmentation fault.
  Together with full tracebacks, this option identifies the location of a bounds error in NumPy operations.
* ``NUMBA_DEBUG_NRT=1`` enables the `Numba runtime statistics counter <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html>`_.
  This counter helps diagnose memory leaks.
* ``NUMBA_DEBUG_TYPEINFER= 1`` print out debugging information about type inferences that numba might need to make if a function is ill-defined
* ``NUMBA_ENABLE_PROFILING=1`` enables profiler use
* ``NUMBA_DUMP_CFG=1`` prints out a control flow diagram

If extra debug options or alteration to these options are required they can be toggled and passed under the ``mode==numba_debug`` option tree in ``mcdc/config.py``.

-------
Caching
-------

MC/DC is a just-in-time (JIT) compiled code.
This is sometimes disadvantageous, especially for users who might run many versions of the same simulation with slightly different parameters.
As the JIT compilation scheme will only compile functions that are actually used in a given simulation, it is not a grantee that any one function will be compiled.

Developers should be very cautious about using caching features.
Numba has a few documented errors around caching.
The most critical of which is that functions in other files that are called by cached functions will not force a recompile, even if there are changes in those sub-functions.
In this case caching should be disabled.

In MC/DC the simulation functions (in ``mcdc/transport/simulation.py``) can be configured to use caching.
Caching behavior is controlled via the ``--caching`` and ``--clear_cache`` command-line flags.

To disable caching, omit the ``--caching`` flag (the default).
Python manages its own ``__pycache__`` directories, and MC/DC does not delete
them during startup. This allows independent batch launches to safely import
MC/DC from the same installation. If manual cache removal is necessary, ensure
that no running job is using the affected cache (`see more about clearing the
Numba cache <https://numba.readthedocs.io/en/stable/developer/caching.html>`_).


MC/DC may eventually enable `Numba's ahead-of-time compilation capabilities <https://numba.readthedocs.io/en/stable/user/pycc.html>`_.
The core development team is waiting for planned `upgrades to Numba's AOT functionality <https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-numba-pycc>`_.
However if absolutely required by users numba does allow for some `cache sharing <https://numba.readthedocs.io/en/stable/developer/caching.html>`_.

------------------
Adding a New Input
------------------

For architectural guidance on adding a model field, embedded configuration, registered object category, or polymorphic subtype, see :doc:`../extending/extending_the_object_model`.
Public model classes and configuration are primarily defined in ``mcdc/object_/``.
Common input-related locations include:

#. ``mcdc/object_/settings.py`` — simulation settings and k-eigenvalue parameters
#. ``mcdc/object_/material.py`` — material and composition definition
#. ``mcdc/object_/transport_model_data.py`` — particle-specific transport data
#. ``mcdc/object_/surface.py`` — surface geometry (``Surface`` class methods)
#. ``mcdc/object_/cell.py`` — cell definitions (``Cell``)
#. ``mcdc/object_/source.py`` — source specifications (``Source``)
#. ``mcdc/object_/tally.py`` — tally objects (``Tally``)
#. ``mcdc/object_/technique.py`` — variance reduction techniques
#. ``mcdc/config.py`` — command-line argument definitions

Changes to runtime-visible fields in the object model also require rebuilding
the generated Numba support. Follow :ref:`rebuilding_numba_support` for the
command, edit-test shortcut, and concurrency constraint.

-------
Testing
-------

See :doc:`continuous_integration` for more information on how we run these tests automatically.

MC/DC has unit and regression test suites that contributions must pass before they are accepted.
Unit tests exercise focused behavior in both Python and Numba modes.
Regression tests compare representative simulations against saved reference results.
GitHub Actions runs the CPU suites on Linux, and a self-hosted runner provides GPU regression coverage.

To run the default fast unit-test suite locally, run,

.. code-block:: sh

    python -m pytest

To run the full unit-test suite in both Python and Numba mode, run,

.. code-block:: sh

    python -m pytest test/unit

To run the regression tests locally, run,

.. code-block:: sh

    python -m pytest test/regression <OPTION_FLAG(s)>


The command runs all regression tests.
The following options control test selection and execution:

* Run a specific test (with wildcard ``*`` support): ``--name=<test_name>``
* Skip a specific test (with wildcard ``*`` support): ``--skip=<test_name>``
* Run in Numba mode: ``--mode=numba``
* Run against the GPU target: ``--target=gpu``
* Run in multiple MPI ranks (currently support ``mpiexec`` and ``srun``): ``--mpiexec=<number of ranks>``
* Run with Slurm ``srun`` instead of ``mpiexec``: ``--srun=<number of ranks>``

The flags can be combined.
Add a new test with the following steps:

#. Create a folder whose name identifies the test.
#. Add the input file as ``input.py``.
#. Add the answer key as ``answer.h5``.
#. Make sure that the number of particles run is large enough for a good test.
#. If the test runs longer than 5 seconds, consider decreasing the number of particles.

When adding a new hardware backend a new instantiation of the test suit should be made.
This is done with github actions. 
See the (``.github/workflows``) for examples.

If a new simulation type is added (e.g. quasi montecarlo w/ davidson's method, residual monte carlo, intrusive uq) more regression tests should be added with your PR.
If you are wondering accommodations.

--------------------
Adding Documentation
--------------------

Documentation is a core part of MC/DC.
Contributions that introduce new features, modify existing behavior, or change developer workflows should update the relevant documentation accordingly.

See the :doc:`../documentation/index` guide for documentation philosophy, writing guidelines, and instructions for contributing to the documentation.
