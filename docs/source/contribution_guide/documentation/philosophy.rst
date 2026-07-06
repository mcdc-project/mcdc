.. _documentation_philosophy:

========================
Documentation Philosophy
========================

Vision
------

The MC/DC documentation should serve the diverse community that develops and
uses the project. As MC/DC continues to grow, its documentation should be as
scalable and maintainable as its software architecture.

MC/DC adopts a layered documentation philosophy that balances usability,
technical depth, and long-term maintainability across the entire project.

This philosophy applies to all forms of MC/DC documentation, including the
README, User Guide, Theory Guide, Python API documentation, examples,
tutorials, and API docstrings.

The Layered Documentation Philosophy
------------------------------------

MC/DC documentation is written for three complementary audiences. Rather than
maintaining separate documentation for each audience, individual documents
should progressively layer information from high-level usage to mathematical
concepts and implementation details. Readers can naturally stop at the level of
detail appropriate for their needs.

Users
^^^^^

Users build geometry, define materials and sources, configure simulations,
execute transport calculations, and analyze results.

Documentation for users should emphasize:

- What MC/DC provides.
- How to use the public API.
- Tutorials, examples, and recommended workflows.
- Best practices for building transport models.

Method Developers
^^^^^^^^^^^^^^^^^

Method Developers use MC/DC as a platform for developing and evaluating new
transport methods and computational algorithms.

Documentation for Method Developers should explain:

- Mathematical formulations.
- Numerical algorithms.
- Data representations.
- Design rationale.
- Extensibility points.
- Relationships between the public API and transport algorithms.

Framework Developers
^^^^^^^^^^^^^^^^^^^^

Framework Developers extend and maintain the MC/DC software framework itself.

Documentation for Framework Developers should describe:

- Software architecture.
- Internal APIs.
- Preparation pipeline.
- Memory layout.
- Compilation workflow.
- Parallel execution.
- Performance considerations.
- Implementation and design decisions.

Guiding Principles
------------------

Documentation should naturally progress from high-level concepts toward
implementation details.

A typical progression is:

#. Overview
#. Usage
#. Examples
#. Mathematical concepts
#. Implementation notes

Not every document requires every section. However, documentation should
generally present information in this order so that each audience can stop
reading once they have reached the level of detail they need.

Public behavior should be described before mathematical representation, and
mathematical representation should be described before implementation details.

API Docstrings
--------------

API docstrings should follow the same layered philosophy. In general:

- The opening description should explain the public purpose of the object,
  function, or module.
- Parameters, return values, attributes, and examples should focus on the
  public interface.
- Mathematical representations, algorithms, and design rationale should be
  documented in the ``Notes`` section when they help Method Developers.
- Framework-specific implementation details should be documented separately as
  implementation notes when appropriate.

Not every API requires all of these sections. The goal is to provide each
audience with the information it needs while keeping the documentation clear,
progressive, and easy to navigate.
