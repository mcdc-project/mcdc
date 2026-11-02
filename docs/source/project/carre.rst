.. _project_carre:

=====
CARRE
=====

The `Center for Advancing the Radiation Resilience of Electronics (CARRE) <https://carre-psaapiv.org/>`_ is a Predictive Simulation Center in the `Predictive Science Academic Alliance Program IV (PSAAP-IV) <https://psaap.llnl.gov/>`_.
CARRE is developing predictive simulation capabilities for radiation effects in electronic systems by bringing together multiscale and multiparticle physics, advanced computational methods, and exascale computing.

MC/DC in CARRE
--------------

MC/DC serves as a radiation-transport component of the CARRE simulation program.
Within CARRE, MC/DC is being extended from its neutron-transport foundation into a comprehensive multiparticle transport code.
The goal is a common simulation framework in which particles and their secondary products can be transported together across multiple physics fidelities and modern CPU and GPU systems.

Multiparticle Transport
-----------------------

The ongoing physics development targets coupled transport of neutrons, photons, electrons, protons, and other charged particles.
Coupling these particles within MC/DC will allow a simulation to follow primary radiation and the secondary particles it produces through a consistent model and execution framework.
As these capabilities mature, users will be able to apply MC/DC to coupled multiparticle radiation-transport problems relevant to radiation effects in electronics and other applications.

Methods for Predictive Science
------------------------------

CARRE's mission requires more than additional particle-interaction physics.
The broader MC/DC methods program includes uncertainty quantification, variance reduction, hybrid methods, verification and validation (V&V), and other advances needed for predictive radiation-transport simulation.
These methods must work with the evolving multiparticle physics while remaining usable across the :doc:`Python, Numba-CPU, and Numba-GPU execution paths <../developer_guide/architecture/transport_execution>`.

Users and Collaborators
-----------------------

The CARRE work creates opportunities for users and collaborators interested in particle-interaction models, nuclear and atomic data, multiparticle coupling, uncertainty quantification, variance reduction, hybrid methods, Monte Carlo algorithms, V&V benchmarks, and high-performance computing.
Prospective users can help shape the developing capabilities by sharing intended applications, workflow requirements, benchmark problems, and validation needs.
Researchers and developers are invited to explore the ongoing work and contribute through the :doc:`MC/DC development process <../developer_guide/contributing/index>`.

Foundation in CEMeNT
--------------------

The CARRE effort builds on MC/DC's origins in the `Center for Exascale Monte Carlo Neutron Transport (CEMeNT) <https://cement-psaap.github.io/>`_, a PSAAP-III project.
CEMeNT established MC/DC's :doc:`core software architecture <../developer_guide/architecture/index>`, :doc:`Numba-based compilation framework <../developer_guide/architecture/transport_execution>`, and :doc:`Python-first design and development philosophy <../developer_guide/architecture/python_first_numba_accelerated_design>`.
CEMeNT also developed transient Monte Carlo neutron-transport methods and scalable execution capabilities for exascale computing platforms.
CARRE leverages these software, methods-development, neutron-transport, and performance-portability foundations as MC/DC expands into its broader multiparticle and predictive-science role.

Ongoing Updates
---------------

This page will evolve with the CARRE program to document major capability milestones, research directions, user-facing readiness, V&V progress, and collaboration opportunities.
