.. _contribution:

==================
Contribution Guide
==================

Thank you for looking to contribute to MC/DC! 
We are really excited to see what you bring to this exciting open source project!
Whether you are here to make a single PR and never return, or want to become a maintainer we are pumped to work with you.
We have regular developers meetings for any and all who are interested to discuss contributions to this code base.

This describes the processes of contributing to MC/DC for both internal (CEMeNT) and external developers.
We make contributions to the ``dev`` branch of MC/DC.
To get started making alterations in a cloned repo

#. fork ``CEMeNT-PSAAP/MCDC`` to your github account
#. ``git clone git@github.com:<YOUR_GITHUB>/MCDC.git``
#. ``git switch dev``
#. run install script which will install MC/DC as an editable package from this directory

Push some particles around!!!!

Development Guidelines
----------------------

.. toctree::
   :maxdepth: 1
   
   documentation.rst
   ci.rst

Check out the :doc:`documentation` guide for info on how to contribute to these docs.
We understand that documenting code is often a lower priority than the code itself, but it goes a long way towards usability and maintainability.

Please note our `code of conduct <https://github.com/CEMeNT-PSAAP/MCDC/blob/main/CODE_OF_CONDUCT.md>`_, which we take seriously.

------------
Code Styling
------------

Our code is auto-linted for the `Black code style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_.
Your contributions will not be merged unless you follow this code style.
It's pretty easy to do this locally, just run,

.. code-block:: sh


    pip install black
    black .


in the top level MC/DC directory and all necessary changes will be automatically made for you.

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
* ``NUMBA_OPT=0`` Forces the compilers to form un-optimized code (other options for this are ``1``, ``2``, and ``3`` with ``3`` being the most optimized). This option might need to be changed if errors only result from more optimization.
* ``DEBUG=False`` turns on all debugging options. This is still disabled in ``mcdc numba_debug`` as it will print ALOT of info on your terminal screen
* ``NUMBA_FULL_TRACEBACKS=1`` allows errors from sub-packages to be printed (i.e. Numpy)
* ``NUMBA_BOUNDSCHECK=1`` numba will check vectors for bounds errors. If this is disabled it bound errors will result in a ``seg_fault``. This in consort with the previous option allows for the exact location of a bound error to be printed from Numpy subroutines
* ``NUMBA_DEBUG_NRT=1`` enables the `Numba run time (NRT) statistics counter <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html>`_ This helps with debugging memory leaks.
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
Alternatively a developer could delete the ``__pycache__`` directory or other cache directory which is system dependent (`see more about clearing the numba cache <https://numba.readthedocs.io/en/stable/developer/caching.html>`_)


At some point MC/DC will enable `Numba's Ahead of Time compilation abilities <https://numba.readthedocs.io/en/stable/user/pycc.html>`_. But the core development team is holding off until scheduled `upgrades to AOT functionality in Numba are implemented <https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-numba-pycc>`_.
However if absolutely required by users numba does allow for some `cache sharing <https://numba.readthedocs.io/en/stable/developer/caching.html>`_.

------------------
Adding a New Input
------------------

To add a new keyword argument such that a user can interface with it in an input deck 
there are a few different places a dev will need to make alterations.
The input objects are defined as dataclasses in the ``mcdc/object_/`` directory:

#. ``mcdc/object_/settings.py`` — simulation settings and k-eigenvalue parameters
#. ``mcdc/object_/material.py`` — material definitions (``Material``, ``MaterialMG``)
#. ``mcdc/object_/surface.py`` — surface geometry (``Surface`` class methods)
#. ``mcdc/object_/cell.py`` — cell definitions (``Cell``)
#. ``mcdc/object_/source.py`` — source specifications (``Source``)
#. ``mcdc/object_/tally.py`` — tally objects (``TallyGlobal``, ``TallyCell``, ``TallySurface``, ``TallyMesh``)
#. ``mcdc/object_/technique.py`` — variance reduction techniques
#. ``mcdc/config.py`` — command-line argument definitions

-------
Testing
-------

Check out the :doc:`ci` for more info on how we run these tests automatically 

MC/DC has a robust testing suite that your changes must be able to pass before a PR is accepted.
Unit tests for functions that have them are ran in a pure python from.
Mostly this is for ensuring input operability
A regression test suite (including models with analytical and experimental solutions) is provided to ensure accuracy and precision of MC/DC.

Our test suite runs on every PR, and Push.
Our github based CI runs for, 

* linux-64 (x86)
* osx-64 (x86, intel based macs)

while we do not have continuous integration we have validated MC/DC on other systems.

To run the regression tests locally, navigate to ``MCDC/test/regression`` and run,

.. code-block:: sh


    python run.py <OPTION_FLAG(s)>


and all the tests will run. Various option ``OPTION_FLAG`` are accepted to control the tests ran,

* Run a specific test (with wildcard ``*`` support): ``--name=<test_name>`` 
* Run in Numba mode: ``--mode=numba``
* Run in multiple MPI ranks (currently support ``mpiexec`` and ``srun``): ``--mpiexec=<number of ranks>``

Note that flags can be combined. To add a new test:

#. Create a folder. The name of the folder will be the test name.
#. Add the input file. Name it`input.py`.
#. Add the answer key file. Name it `answer.h5`.
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


It's not everything it needs to be but we are trying!
If your contribution changes the behavior of the input deck, instillation process, or testing infrastructure your contribution must include alteration to this documentation.
That can be done by editing the RST files in ``/MCDC/docs/source/<FILENAME>.rst``.

To add a new page to the documentation,

#. Add a new file for example ``<FILE_NAME>.rst``
#. Add the necessary file header (for example this file is: ``.. _contributions:``)
#. Add ``<FILE_NAME>`` (without file extension to the ``.. toctree::`` section of ``index.rst``)
#. Write your contributions using ``.rst`` format (see this `cheat sheet <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_)

To build the docs changes you have made locally before committing,

#. Install dependencies (we recommend: ``conda install sphinx`` and ``pip install furo sphinx_toolbox``).
   Note that these dependencies are not installed as part of base MC/DC.
#. From the ``MCDC/docs/`` directory, run ``make html`` to compile.
#. Launch ``build/html/index.html`` with your browser of choice.


-------------
Pull Requests
-------------


MC/DC works off of a fork workflow in which contributors fork our repo, make alterations, and submit a pull requests.
You should only submit a pull request once your code passes all tests, is properly linted, you have edited documentation (if necessary), and added any new tests (if needed).
Open a PR to the ``dev`` branch in Github.
MC/DC's main branch is only updated for version releases at which time a PR from dev to main is opened, tagged, archived, and published automatically.

Within your pull request documentation please list:

#. Type of PR (e.g. enhancement, bugfix, etc);
#. Link to any theory to understand what you are doing;
#. Link to any open/closed issues if applicable;
#. New functionalities implemented
#. Depreciated functionalities
#. New dependencies needed (we don't add these lightly)
#. Anything else we need to give you the thorough code review you deserve!

If these things aren't listed we will ask for clarifying questions!
