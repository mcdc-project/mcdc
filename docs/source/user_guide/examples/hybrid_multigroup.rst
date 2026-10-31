.. _example_hybrid_multigroup:

===========================
Hybrid Multigroup Transport
===========================

This fixed-source problem combines native H-1 composition with one-group neutron multigroup data on the same material.
The 1 eV source lies inside the multigroup interval from 0.1 to 10 eV, while the 1 MeV source uses native H-1 physics outside that interval.
Multigroup scattering represents the outgoing group at its 5.05 eV midpoint.
The tally uses physical energy boundaries because particle energy remains in eV during hybrid transport.
The example requires ``MCDC_LIB`` to identify a native-data library containing ``neutron-H1-293.6K.h5``.

Full Input
----------

.. literalinclude:: ../../../../examples/hybrid_multigroup/input.py
   :language: python
   :linenos:

Post-processing
---------------

.. literalinclude:: ../../../../examples/hybrid_multigroup/process-output.py
   :language: python
   :linenos:

How to Run
----------

From inside ``examples/hybrid_multigroup``:

.. code-block:: sh

   python input.py
   python process-output.py

The transport calculation writes ``hybrid_multigroup.h5`` and the post-processing script writes ``hybrid_multigroup_flux.png``.
