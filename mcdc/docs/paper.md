---
title: 'Monte Carlo / Dynamic Code (MC/DC): An accelerated Python package for fully transient neutron transport and rapid methods development'
tags:
  - Python
  - Monte Carlo
  - nuclear engineering
  - neutron transport
  - reactor analysis
  - numba
  - HPC
  - mpi4py
  - GPU
authors: # x=reviewed
  - name: Joanna Piper Morgan #x
    orcid: 0000-0003-1379-5431
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Ilham Variansyah
    orcid: 0000-0003-3426-7160
    affiliation: "1, 2"
    corresponding: true
  - name: Samuel L. Pasmann
    orcid: 0000-0003-1391-1471
    affiliation: "1, 3"
  - name: Kayla B. Clements
    orcid: 0000-0003-3358-5618
    affiliation: "1, 2"
  - name: Braxton Cuneo
    orcid: 0000-0002-6493-0990
    affiliation: "1, 5"
  - name: Alexander Mote 
    orcid: 0000-0001-5099-0223
    affiliation: "1, 2"
  - name: Charles Goodman
    affiliation: "1, 4"
  - name: Caleb Shaw 
    affiliation: "1, 4"
  - name: Jordan Northrop
    orcid: 0000-0003-0420-9699
    affiliation: "1, 2"
  - name: Rohan Pankaj
    orcid:  0009-0005-0445-9323
    affiliation: "1, 6"
  - name: Ethan Lame
    orcid: 0000-0001-7686-9755
    affiliation: "1, 2"
  - name:  Benjamin Whewell
    orcid: 0000-0001-7826-5525
    affiliation: "1, 3"
  - name: Ryan G. McClarren #advisors in order of authors except Niemeyer
    orcid: 0000-0002-8342-6132
    affiliation: "1, 3"
  - name: Todd S. Palmer
    orcid: 0000-0003-3310-5258
    affiliation: "1, 2"
  - name: Lizhong Chen 
    orcid: 0000-0001-5890-7121
    affiliation: "1, 2"
  - name: Dmitriy Y. Anistratov
    affiliation: "1, 4"
  - name: C. T. Kelley
    affiliation: "1, 4"
  - name: Camille J. Palmer
    orcid: 0000-0002-7573-4215
    affiliation: "1, 2"
  - name: Kyle E. Niemeyer
    orcid: 0000-0003-4425-7097
    affiliation: "1, 2"


affiliations:
 - name: Center for Exascale Monte Carlo Neutron Transport
   index: 1
 - name: Oregon State University, Corvallis, OR, USA
   index: 2
 - name: University of Notre Dame, South Bend, IN, USA
   index: 3
 - name: North Carolina State University, Raleigh, NC, USA
   index: 4
 - name: Seattle University, Seattle, WA, USA
   index: 5
 - name: Brown University, Providence, RI, USA
   index: 6
date: 28 Jan 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Predicting how neutrons move through space and time, and change speed and direction of travel, are important considerations when modeling inertial confinement fusion systems, pulsed neutron sources, and nuclear criticality safety experiments, among other systems.
This can be modeled with a Monte Carlo simulation, where particles with statistical importance are created and transported to produce a particle history [@lewis_computational_1984].
A particle's path and the specific set of events that occur within its history are governed by pseudo-random numbers, known probabilities (e.g., from material data), and known geometries.
Information about how particles move and/or interact with the system is tallied to construct a histogram solution of parameters of interest with an associated statistical error from the Monte Carlo process. 
Simulating dynamic systems that vary in time requires novel numerical methods to compute a solution performantly.
We designed Monte Carlo / Dynamic Code (`MC/DC`) to explore such novel numerical methods on modern high-performance computing systems.
We avoid the need for a compiled or domain-specific language by using the Numba compiler for Python to accelerate and abstract our compute kernels to near compiled code speeds.
We have implemented novel algorithms using this scheme and, in some verification tests, have approached the performance of industry-standard codes at the scale of tens of thousands of processors.

# Statement of need

`MC/DC` is a performant software platform for rapidly developing and applying novel, dynamic, neutron-transport algorithms on modern high-performance computing systems.
It uses the Numba compiler for Python to compile compute kernels to a desired hardware target, including support for graphics processing units (GPUs) [@lam_numba_2015].
`MC/DC` uses `mpi4py` for distributed-memory parallelism [@mpi4py_2021] and has run at the scale of tens of thousands of processors [@variansyah_mc23_mcdc].
These acceleration and abstraction techniques allow `MC/DC` developers to remain in a pure Python development environment without needing to support compiled or domain-specific languages.
This has allowed `MC/DC` to grow from its initialization less than two years ago into a codebase that supports full performant neutron transport and investigation of novel transport algorithms, with development mostly from relative novices.

Many traditionally developed neutron-transport codes are export-controlled (e.g. `MCNP` [@mcnp], `Shift` [@shift], and `MCATK` [@mcatk]) and some are known to be difficult to install, use, and develop in.
`MC/DC` is open-source, and thus, similar to other open-source Monte Carlo neutron-transport codes (e.g., `OpenMC` [@openmc]), it promotes knowledge sharing, collaboration, and inclusive, community-driven development.
What makes `MC/DC` unique is that its code base is exclusively written in Python, making it a good method exploration tool and an excellent entry point for students.
Furthermore, `MC/DC` is wrapped as a Python package that can be conveniently installed via the `pip` distribution, and its development is assisted by a suite of unit, regression, verification, and performance tests, which are mostly run using continuous integration via GitHub Actions.
This all together makes `MC/DC` ideal for use in an academic environment for both research and education.

`MC/DC` has support for continuous and multi-group energy neutron transport physics with constructive solid geometry modeling.
It can solve k-eigenvalue problems (used to determine neutron population growth rates in reactors) as well as fully dynamic simulations.
It also supports some simple domain decomposition, with more complex algorithms currently being implemented.
In an initial code-to-code performance comparison, `MC/DC` was found to run about 2.5 times slower than the Shift Monte Carlo code for a simple problem and showed similar scaling on some systems [@variansyah_mc23_mcdc].

`MC/DC`-enabled explorations into dynamic neutron transport algorithms have been successful, including quasi-Monte Carlo techniques [@mcdc:qmc], hybrid iterative techniques for k-eigenvalue simulations [@mcdc:qmcabs], population control techniques [@mcdc:variansyah_nse22_pct; @mcdc:variansyah_physor22_pct], continuous geometry movement techniques that model transient elements [@variansyah_mc23_moving_object] (e.g., control rods or pulsed neutron experiments) more accurately than step functions typically used by other codes, initial condition sampling technique for typical reactor transients [@variansyah_mc23_ic], hash-based random number generation [@mcdc:cuneo2024alternative], uncertainty and global sensitivity analysis [@mcdc:clements_mc23; @mcdc:clements_variance_2024], residual Monte Carlo methods, and machine learning techniques for dynamic node scheduling, among others.

# Future Work

The main `MC/DC` branch currently only supports CPU architectures enabled by Numba (`x86-64`, `arm64`, and `ppc64`) but we are rapidly extending support to GPUs.
We currently have operability on Nvidia GPUs (supported via Numba), and work is ongoing to enable compilation for AMD GPUs.
On GPUs, `MC/DC` will use the `harmonize` asynchronous GPU scheduler to increase performance [@brax2023].
`harmonize` works by batching jobs during execution such that similar operations get executed simultaneously, reducing the divergence between parallel threads running on the GPU.

We will continue to explore novel methods for dynamic neutron transport and will keep pushing to make `MC/DC` not only a proven platform for rapidly exploring neutron-transport methods, but also a fully-fledged simulation code for academic and industrial use.

# Acknowledgements

This work was supported by the Center for Exascale Monte-Carlo Neutron Transport (CEMeNT) a PSAAP-III project funded by the Department of Energy, grant number: DE-NA003967.

# References
