.. _ana:

============================
Acceleration and Abstraction
============================

MC/DC employs a layered compilation and abstraction strategy that allows the same Python source code to target CPUs (pure Python or Numba JIT) and GPUs (via Harmonize) without modification to the transport algorithms.

Execution Modes
---------------

MC/DC supports three execution modes, selected at runtime with the ``--mode`` flag:

- **Python mode** (``--mode=python``): Transport kernels run as interpreted Python. Useful for debugging and rapid prototyping.
- **Numba mode** (``--mode=numba``): Transport kernels are just-in-time compiled to native machine code using `Numba <https://numba.readthedocs.io/>`_. This provides significant speedup (often 100x or more) at the cost of an initial compilation overhead of 15–80 seconds.
- **Numba debug mode** (``--mode=numba_debug``): JIT compilation with extra debug instrumentation (bounds checking, full tracebacks, type inference logging). Slower, but produces actionable error messages.

The ``--target`` flag selects the hardware target: ``cpu`` (default) or ``gpu``.

Numba Object Generation
-----------------------

MC/DC's simulation state (materials, surfaces, cells, tallies, settings, particle banks, etc.) is defined as annotated Python classes in ``mcdc/object_/``.
At startup, the **Numba object generator** (``mcdc/code_factory/numba_objects_generator.py``) converts these class hierarchies into NumPy structured array dtypes:

#. Class annotations (type hints) are read and mapped to NumPy dtypes.
#. Polymorphic objects (e.g., different surface types) are represented using ``parent_ID``/``child_ID`` fields that index into typed sub-arrays.
#. All simulation data is flattened into a single contiguous NumPy buffer (``data``), enabling efficient access from JIT-compiled code.
#. Getter and setter access functions (in ``mcdc/mcdc_get/`` and ``mcdc/mcdc_set/``) are auto-generated so that JIT-compiled transport kernels can read and write simulation state without Python object overhead.

This approach allows the transport code to be written in natural, object-oriented Python while still achieving the performance of flat array access in compiled mode.

GPU Portability
---------------

When targeting GPUs, MC/DC uses the `Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ library as its GPU runtime.
The GPU program builder (``mcdc/code_factory/gpu/program_builder.py``) constructs a Harmonize ``RuntimeSpec`` that includes:

- **Global state**: the simulation structured array and the flat data buffer.
- **Device functions**: the MC/DC transport kernels, compiled from Python to device code via Numba.
- **Scheduling strategy**: either event-based (``--gpu_strategy=event``) or asynchronous (``--gpu_strategy=async``, Nvidia only).

The compilation pipeline differs by vendor:

- **Nvidia**: Python → PTX (via ``numba.cuda``) → relocatable device code (via ``nvcc``) → linked shared library.
- **AMD**: Python → LLVM-IR (via a `Numba-HIP patch <https://github.com/ROCm/numba-hip>`_) → relocatable device code (via ``hipcc`` / ``clang``) → linked shared library.

For a detailed walkthrough of the compilation flow, see the :ref:`theory_gpu` section.

For more details, see:

- J. P. Morgan, I. Variansyah, B. Cuneo, T. S. Palmer, and K. E. Niemeyer. "Performance Portable Monte Carlo Neutron Transport in MCDC via Numba." Preprint DOI 10.48550/arXiv.2306.07847.
- B. Cuneo and M. Bailey. "Divergence Reduction in Monte Carlo Neutron Transport with On-GPU Asynchronous Scheduling." *ACM TOMACS* (2023). DOI 10.1145/3626957.
