.. MC/DC documentation master file

======================================
MC/DC: Monte Carlo Dynamic Code
======================================

MC/DC is a performant, scalable, and portable Python-based Monte Carlo radiation 
transport software package. It is purpose-built as a rapid methods development 
platform capable of leveraging modern high-performance computing systems, supporting 
both CPUs and GPUs.

MC/DC supports continuous-energy and multi-group neutron transport calculations. It is 
capable of running fixed-source and eigenvalue transport simulations on models built 
from constructive solid geometry. For continuous-energy neutron transport, 
MC/DC translates `ACE <https://nucleardata.lanl.gov/ace/>`_ nuclear data libraries into 
its native `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ format. Photon, electron, 
and charged-particle transport are currently under development, with the goal of making 
MC/DC a multi-radiation/particle transport software package.

While MC/DC's Python environment promotes rapid iterative testing of ideas, its 
Numba-based compilation framework improves runtime performance and enables portability. 
`Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ serves as the GPU execution 
framework, optimizing device utilization within stochastic simulations; and 
`MPI4Py <https://mpi4py.readthedocs.io/en/stable/>`_ is used to achieve parallel 
scalability across nodes in large computer clusters. In addition to running on commonly 
used desktops and workstations, MC/DC has been tested on large heterogeneous 
high-performance systems, including 
`Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen—decommissioned>`_ 
(IBM POWER9 and NVIDIA Volta V100) and 
`Tuolumne <https://hpc.llnl.gov/hardware/compute-platforms/tuolumne>`_ (AMD MI300A APU).

MC/DC development was initiated by the Center for Exascale Monte Carlo Neutron 
Transport (`CEMeNT <https://cement-psaap.github.io>`_), a Focused Investigatory Center 
of the Predictive Science Academic Alliance Program–III 
(`PSAAP-III <https://psaap.llnl.gov>`_). MC/DC is currently under active development 
by the Center for Advancing the Radiation Resilience of Electronics 
(`CARRE <https://carre-psaapiv.org>`_), a Predictive Simulation Center of 
`PSAAP-IV <https://psaap.llnl.gov>`_. MC/DC is open source 
(`BSD 3-Clause <https://github.com/CEMeNT-PSAAP/MCDC/blob/main/LICENSE>`_) and 
welcomes external contributions via `GitHub <https://github.com/CEMeNT-PSAAP/MCDC>`_.

------------------------------
Contents
------------------------------

.. toctree::
   :maxdepth: 1
   :caption: User Documentation

   install
   user/index
   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   contribution/index
   theory/index
   pythonapi/index

.. toctree::
   :maxdepth: 1
   :caption: References

   publications
   citation

.. sidebar-links::
   :caption: External Links
   :pypi: mcdc
   :github:

   CARRE <https://carre-psaapiv.org/>
   CEMeNT <https://cement-psaap.github.io>
