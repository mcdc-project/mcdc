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

   .. grid-item-card:: :octicon:`code-square;2em` Reference
      :link: reference/index
      :link-type: doc
      :link-alt: Look up MC/DC Python interfaces and data formats
      :class-card: sd-card-hover
      :text-align: center

      Look up MC/DC's Python interfaces and the HDF5 transport data format.

      +++
      Browse the reference :octicon:`arrow-right`

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

   .. grid-item-card:: :octicon:`checklist;2em` Verification, Validation, and Performance
      :link: project/vvp/index
      :link-type: doc
      :link-alt: Explore MC/DC verification, validation, and performance evidence
      :class-card: sd-card-hover
      :text-align: center

      Explore benchmark problems and results used to assess MC/DC's correctness,
      physical credibility, and computational performance.

      +++
      Explore the evidence :octicon:`arrow-right`

   .. grid-item-card:: :octicon:`telescope;2em` CARRE Project
      :link: project/carre
      :link-type: doc
      :link-alt: Explore the CARRE research program and collaboration opportunities
      :class-card: sd-card-hover
      :text-align: center

      Explore the research program advancing MC/DC, its goals, and
      opportunities for collaboration.

      +++
      Explore CARRE :octicon:`arrow-right`

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
