.. MC/DC documentation master file

======================================
MC/DC: Monte Carlo Dynamic Code
======================================

MC/DC is an open-source, Python-based Monte Carlo radiation transport software
package that combines rapid methods development with scalable execution on modern
high-performance computing systems. It supports execution across CPUs and GPUs
while providing a flexible environment for developing and testing new transport
algorithms.

MC/DC is intended for researchers developing new Monte Carlo transport methods,
including variance reduction techniques, sensitivity and uncertainty quantification
methods, as well as high-performance computing algorithms. It also provides
an accessible platform for students learning Monte Carlo radiation transport methods
and modern code development.

MC/DC supports continuous-energy and multi-group neutron transport calculations,
including fixed-source and eigenvalue simulations on constructive solid geometry
(CSG) models. For continuous-energy transport, MC/DC converts
`ACE <https://nucleardata.lanl.gov/ace/>`_-format nuclear data libraries into its native
`HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ format. Photon, electron, proton,
and other charged-particle transport capabilities are currently under development as
part of the ongoing expansion of MC/DC into a comprehensive multi-particle radiation
transport software package.

MC/DC's Python interface enables rapid prototyping and iterative development,
while its `Numba <https://numba.pydata.org>`_-based compilation framework delivers high
performance without sacrificing portability.
`Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ provides a GPU execution
framework, and `MPI4Py <https://mpi4py.readthedocs.io/en/stable/>`_ enables
distributed-memory parallelism across large HPC systems.
In addition to desktop and workstation systems, MC/DC has been demonstrated on
large heterogeneous supercomputers, including
`Lassen <https://asc.llnl.gov/computers/historicdecommissioned-machines/lassen>`_
(IBM POWER9 and NVIDIA Volta V100) and 
`Tuolumne <https://warpx.readthedocs.io/en/latest/install/hpc/tuolumne.html>`_ (AMD
MI300A APU).

MC/DC was initiated by the Center for Exascale Monte Carlo Neutron Transport
(`CEMeNT <https://cement-psaap.github.io>`_), a Focused Investigatory Center of
the Predictive Science Academic Alliance Program–III
(`PSAAP-III <https://psaap.llnl.gov>`_). Development is now led by the Center for
Advancing the Radiation Resilience of Electronics
(`CARRE <https://carre-psaapiv.org>`_), a Predictive Simulation Center of
`PSAAP-IV <https://psaap.llnl.gov>`_.

MC/DC is released under the
`BSD 3-Clause <https://github.com/CEMeNT-PSAAP/MCDC/blob/main/LICENSE>`_
license and welcomes community contributions through
`GitHub <https://github.com/CEMeNT-PSAAP/MCDC>`_.



.. admonition:: Recommended citation
   :class: tip

   Morgan, Joanna Piper, et al. "Monte Carlo/Dynamic Code (MC/DC): An accelerated 
   Python package for fully transient neutron transport and rapid methods development." 
   Journal of Open Source Software 9.96 (2024): 6415. 
   https://joss.theoj.org/papers/10.21105/joss.06415

------------------------------
Contents
------------------------------

.. toctree::
   :maxdepth: 1
   :caption: User Documentation

   install
   user/index
   pythonapi/index
   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   contribution/index
   theory/index

.. toctree::
   :maxdepth: 1
   :caption: References

   publications

.. sidebar-links::
   :caption: External Links
   :pypi: mcdc
   :github:

   CARRE <https://carre-psaapiv.org/>
   CEMeNT <https://cement-psaap.github.io>
