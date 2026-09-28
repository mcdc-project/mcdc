.. _project_vvp:

=========================================
Verification, Validation, and Performance
=========================================

`MC/DC Verification, Validation, and Performance (MC/DC-VVP) <https://github.com/mcdc-project/mcdc-vvp>`_ provides the evidence used to assess the correctness, physical credibility, and computational behavior of MC/DC.
The campaigns are maintained alongside MC/DC so that repeatable systematic problems exercise major code changes.
VVP focuses on full-transport problems that test complete simulations, complementing the component-level verification provided separately by :doc:`unit tests <../../developer_guide/contributing/unit_testing>`.


Three kinds of evidence
-----------------------

There are three kinds of evidence: verification, validation, and performance.

1. **Verification** asks whether MC/DC solves its stated mathematical problems correctly.

   a. **Analytical verification** compares MC/DC with an analytical, semi-analytical, or manufactured reference solution and checks for the expected statistical convergence rate.

   b. **Code-to-code verification** compares independently implemented transport codes and checks whether their relative differences decrease at the expected Monte Carlo rate.
      Agreement between codes increases confidence but does not establish correctness because the codes may share assumptions or biases.

2. **Validation** asks whether an MC/DC model reproduces experimental observations within the relevant measurement, model, and nuclear-data uncertainties.
   Validation assesses the combined physical model, data, and implementation against experimental evidence.

3. **Performance** studies measure quantities such as runtime, throughput, memory use, and parallel scalability.
   They characterize computational efficiency separately from numerical correctness and physical fidelity.

The development campaign currently reports neutron-transport verification.
Validation and performance results will be added as their suites mature.

Organization
------------

MC/DC-VVP uses **suite**, **case**, and **task** as its standard hierarchy.
`Maestro Workflow Conductor <https://github.com/LLNL/maestrowf>`_ orchestrates these task sequences for local workstations and HPC platforms while each suite retains its own launch and processing workflow.

.. code-block:: text

   suite
   └── case
       └── task

A **suite** is a related collection of VVP problems that shares a launch and processing workflow.
A **case** is one physical or mathematical problem definition together with its model, reference, and result-processing logic.
A **task** is one execution of a case at one sampling level, such as a selected source-particle count or number of active eigenvalue cycles.

A suite answers a broad verification, validation, or performance question, a case exercises a particular problem and capability set, and repeated tasks expose how its result changes with sampling effort.

For example, the analytical neutron fixed-source verification suite contains the AZURV1 case, which is executed as a sequence of source-particle-count tasks:

.. code-block:: text

   analytical neutron fixed-source suite
   └── AZURV1 case
       ├── task: N_particle = 100,000
       ├── task: N_particle = 215,443
       ├── ...
       └── task: N_particle = 10,000,000

This example describes verification, where repeated sampling levels reveal an error-convergence trend.
The same hierarchy applies to validation and performance, but their tasks will likely represent different variations, such as experimental configurations for validation or hardware and scaling points for performance.

Current status and next steps
-----------------------------

The :doc:`verification program <verification/index>` currently covers analytical and code-to-code neutron transport.
Each case page presents its problem definition, reference solution, convergence study, and published results as one narrative.
The figures and animations are loaded from the rolling `VVP results release <https://github.com/mcdc-project/mcdc/releases/tag/vvp-results>`_, which is updated when a new MC/DC campaign is published.
The initial result set was generated from the development branches in preparation for MC/DC v0.16.0.

Future campaigns will extend the program with :doc:`validation <validation/index>` against experimental observations and :doc:`performance <performance/index>` studies of computational efficiency and scalability.

.. toctree::
   :hidden:

   verification/index
   validation/index
   performance/index
