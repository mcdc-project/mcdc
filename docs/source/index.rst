:html_theme.sidebar_secondary.remove: true

======================================
MC/DC: Monte Carlo Dynamic Code
======================================

MC/DC is an open-source, Python-based Monte Carlo radiation transport software
package for rapid methods development and scalable execution on CPUs, GPUs,
and modern high-performance computing systems. New to the project? Begin with
:doc:`What is MC/DC? <user_guide/getting_started/what_is_mcdc>`.

-------------
Documentation
-------------

Choose the path that best matches what you want to accomplish.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`book;2em` User Guide
      :link: user_guide/index
      :link-type: doc
      :link-alt: Get started and learn how to use MC/DC
      :class-card: sd-card-hover
      :text-align: center

      Learn what MC/DC is, install it, run your first simulation, and follow
      task-oriented guidance for everyday use.

      +++
      Start here :octicon:`arrow-right`

   .. grid-item-card:: :octicon:`beaker;2em` Theory and Methods
      :link: theory/index
      :link-type: doc
      :link-alt: Study the transport theory and numerical methods in MC/DC
      :class-card: sd-card-hover
      :text-align: center

      Study the transport theory, numerical algorithms, and acceleration
      methods implemented in MC/DC.

      +++
      Explore the theory :octicon:`arrow-right`

   .. grid-item-card:: :octicon:`code-square;2em` API Reference
      :link: reference/python_api/index
      :link-type: doc
      :link-alt: Look up MC/DC Python classes and methods
      :class-card: sd-card-hover
      :text-align: center

      Look up the classes, methods, arguments, and attributes available through
      MC/DC's Python interface.

      +++
      Browse the API :octicon:`arrow-right`

   .. grid-item-card:: :octicon:`tools;2em` Developer Guide
      :link: developer_guide/index
      :link-type: doc
      :link-alt: Understand and contribute to MC/DC development
      :class-card: sd-card-hover
      :text-align: center

      Understand MC/DC's architecture, extend its implementation, and prepare
      contributions to the project.

      +++
      Develop MC/DC :octicon:`arrow-right`

More resources
--------------

- Learn from complete input decks in :doc:`user_guide/examples/index`.
- Review the :doc:`MC/DC verification, validation, and performance (VVP)<project/vvp/index>` evidence.
- Explore the ongoing :doc:`CARRE research program <project/carre>`, its collaboration opportunities, and the :doc:`MC/DC publication record <project/publications>`.
- Follow the contribution workflow in :doc:`developer_guide/contributing/index`.

.. admonition:: Recommended citation
   :class: tip

   Morgan, Joanna Piper, et al. "Monte Carlo/Dynamic Code (MC/DC): An accelerated
   Python package for fully transient neutron transport and rapid methods development."
   *Journal of Open Source Software* 9.96 (2024): 6415.
   https://doi.org/10.21105/joss.06415

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/index
   theory/index
   reference/index
   developer_guide/index
   project/index
