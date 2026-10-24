.. _user_guide:

==========
User Guide
==========

The User Guide provides task-oriented instructions for building, running, and
analyzing MC/DC simulations. It begins with an introduction for new users and
then covers modeling, execution, examples, and troubleshooting.

For exact class and method signatures, use the :doc:`../reference/index`.

Getting Started
---------------

New to MC/DC? Begin here to understand the project, install the package, and
complete your first transport calculation.

.. toctree::
   :maxdepth: 2

   getting_started/index

Data Preparation
----------------

Use the projectile guides to prepare transport data from source libraries.

.. toctree::
   :maxdepth: 2

   data_preparation/index

Modeling and Results
--------------------

Follow :doc:`simulation_lifecycle` for the complete construct, compile,
visualize, execute, and post-process workflow. Use
:doc:`iterative_simulations` when several runs reuse most of a model while
changing selected inputs.

.. toctree::
   :maxdepth: 1

   simulation_lifecycle
   materials_and_multigroup
   sources
   iterative_simulations
   tallies

Execution
---------

.. toctree::
   :maxdepth: 1

   execution/index

Learning by Example
-------------------

Use :doc:`examples/index` to learn from complete, runnable input decks
that progress from basic models to advanced benchmarks.

.. toctree::
   :maxdepth: 2

   examples/index

Help & Support
--------------

.. toctree::
   :maxdepth: 1

   container
   faq
   troubleshooting
