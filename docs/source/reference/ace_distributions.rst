.. _ace-distribution-support:

========================
ACE Distribution Support
========================

This page maps ACE neutron distributions to their ``mcdc-data`` representations, Python objects, and transport samplers.
The source law definitions are given in the `ACE format reference <https://github.com/NuclearData/ACEFormat/blob/master/src/ContinuousEnergyNeutron.tex>`_.
ACE LAW numbers and ENDF LAW numbers identify different representations and are not interchangeable.

The :doc:`data format reference <data_format/representations>` defines the stored fields and interpolation rules.
The :doc:`neutron preparation guide <../user_guide/data_preparation/neutron>` describes library generation and reaction coverage.

Energy-angle laws
-----------------

The table lists the energy laws accepted by ``tools/data_library_generator/common/ace_converter.py``.
Object classes are defined in ``mcdc.object_.distribution`` and samplers in ``mcdc.transport.distribution``.
The ``mcdc-data`` column gives the Distribution's ``type`` attribute.

.. list-table:: Supported ACE energy-angle laws
   :header-rows: 1
   :widths: 5 15 14 22 22 22

   * - ACE LAW
     - Physical model
     - ``mcdc-data``
     - Object class
     - Transport sampler
     - Sampling and constraints
   * - 3
     - Level scattering
     - ``level-scattering``
     - ``DistributionLevelScattering``
     - ``sample_level_scattering``
     - Separate angular Distribution
   * - 4
     - Tabulated outgoing energy
     - ``multi-table``
     - ``DistributionMultiTable``
     - ``_sample_multi_table``
     - Continuous, discrete, or mixed; corresponding discrete lines; linear incident-energy interpolation; separate angular Distribution
   * - 7
     - Simple Maxwellian fission spectrum
     - ``maxwellian``
     - ``DistributionMaxwellian``
     - ``sample_maxwellian``
     - Separate angular Distribution
   * - 9
     - Evaporation spectrum
     - ``evaporation``
     - ``DistributionEvaporation``
     - ``sample_evaporation``
     - Separate angular Distribution
   * - 44
     - Kalbach–Mann energy-angle model
     - ``kalbach-mann``
     - ``DistributionKalbachMann``
     - ``sample_kalbach_mann``
     - Continuous energy PDFs; linear incident-energy interpolation; joint angle sampling
   * - 61
     - Tabulated energy with conditional angle
     - ``energy-angle-table``
     - ``DistributionTableEnergyAngle``
     - ``sample_table_energy_angle``
     - Continuous energy and cosine PDFs; linear incident-energy interpolation; joint angle sampling
   * - 66
     - N-body phase space
     - ``n-body``
     - ``DistributionNBody``
     - ``_sample_correlated_distribution`` (N-body branch)
     - COM frame; dimensionless energy-fraction PDF; isotropic COM emission

Laws 1, 2, 5, 11, 22, 24, 33, and 67 are not accepted by the converter.
LAW 4's discrete-line support does not imply conversion support for LAW 2.

.. _ace-incident-energy-sampling:

Incident-energy table sampling
------------------------------

Laws 4, 44, and 61 use conditional tables indexed by incident energy.
For E between neighboring incident energies E0 and E1, define :math:`t=(E-E_0)/(E_1-E_0)`.
For continuous-only tables, select the lower or upper table with probabilities 1 - t and t, then sample its outgoing-energy PDF using that table's density interpolation.
LAW 4 separates discrete-line selection from continuous sampling as described below.

The sampled energy is mapped into the interpolated support using unit-base scaling.
If the selected table has support [a, b] and supplies a sample y, the outgoing energy is

.. math::

   E'=a_t+\frac{y-a}{b-a}(b_t-a_t),
   \qquad
   a_t=(1-t)a_0+ta_1,
   \qquad
   b_t=(1-t)b_0+tb_1.

Here [a0, b0] and [a1, b1] are the lower and upper incident tables' energy supports.
Outside the incident-energy grid, use the nearest endpoint table without support scaling.

Separate angular MultiTables use the same incident-table selection but leave the cosine coordinate unchanged.
For a MultiTable with histogram incident-energy interpolation, use the lower table without support scaling.
The ACE converter requires linear incident-energy interpolation for laws 4, 44, and 61.

Sampling by law
---------------

The descriptions below use incident energy E, outgoing energy E', and scattering cosine :math:`\mu`.
Energies are in eV, and the reaction declares the sampling reference frame.
Laws 3, 4, 7, and 9 use a separate angular Distribution; laws 44, 61, and 66 supply both energy and angle.

.. _ace-law-3:

LAW 3: Level scattering
^^^^^^^^^^^^^^^^^^^^^^^^^

The outgoing energy is deterministic at a given incident energy:

.. math::

   E'=C_2(E-C_1).

The converter stores C1 in eV and leaves C2 dimensionless.
``sample_level_scattering`` evaluates this expression without an energy random draw.
The reaction samples its separate angular Distribution and transforms the result to the laboratory frame when required.

.. _ace-law-4:

LAW 4: Tabulated outgoing energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each incident energy supplies a continuous PDF, a discrete PMF, or both.
The converter separates ACE's leading discrete entries from the continuous background and packs both components into ``multi-table``.
Line probabilities remain dimensionless, while line energies and continuous coordinates are converted to eV and continuous densities to 1/eV.
The PMF sum and PDF integral jointly equal one at each incident energy.
The continuous component retains its own density interpolation rules.

Corresponding discrete lines have the same index in neighboring incident tables.
For line j, interpolate its probability and energy:

.. math::

   p_j(E)=(1-t)p_{0,j}+tp_{1,j},
   \qquad
   e_j(E)=(1-t)e_{0,j}+te_{1,j}.

Use a uniform random number to select a line from the cumulative interpolated probabilities.
Return its interpolated energy directly, without continuous support scaling.
All incident tables in a converted LAW 4 distribution must retain the same number and ordering of corresponding lines; zero probabilities preserve lines that vanish at an endpoint.

If the draw reaches the continuous component, rebase it onto that component's cumulative probability.
Let q0 and q1 be the endpoint continuous probabilities, and :math:`q(E)=(1-t)q_0+tq_1`.
Select the lower or upper continuous table with conditional probabilities :math:`(1-t)q_0/q(E)` and :math:`tq_1/q(E)`.
These weights preserve the interpolated continuous contribution when the discrete probabilities change with incident energy.
Invert the selected continuous CDF and apply :ref:`unit-base scaling <ace-incident-energy-sampling>` using the continuous support only.
If a continuous component is absent at one endpoint, retain the other endpoint's support.
Histogram and linear density segments have analytic inverses; logarithmic segments use bounded inversion.
The separate angular Distribution is sampled independently of outgoing energy at the given incident energy.

The separate treatment of line probabilities, line energies, and the continuous background follows the `MCNP LAW 4 description (pages 2-42–2-43) <https://mcnpx.lanl.gov/pdf_files/TechReport_2003_LANL_LA-UR-03-1987Revised212008_SweezyBoothEtAl.pdf#page=76>`_.

.. _ace-law-7:

LAW 7: Simple Maxwellian fission spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The outgoing-energy density has the form

.. math::

   p(E'\mid E)\propto\sqrt{E'}\exp[-E'/T(E)],
   \qquad 0\le E'\le E-U.

T(E) is the positive spectral temperature in eV, and U is ``restriction-energy`` in eV.
Spectral temperature is distinct from the target's temperature in kelvin.
The converter stores T(E) as Data with its source interpolation; the runtime also accepts polynomial Data.
Transport evaluates the temperature at E, clamps a table outside its grid, and uses rejection sampling to enforce the energy restriction.
The spectrum is normalized over its permitted interval, and its angle is supplied separately.

.. _ace-law-9:

LAW 9: Evaporation spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The outgoing-energy density has the form

.. math::

   p(E'\mid E)\propto E'\exp[-E'/T(E)],
   \qquad 0\le E'\le E-U.

T(E) and U have the same Data representation, units, and interpretation as in :ref:`ace-law-7`.
The extra factor E' distinguishes this spectrum from the Maxwellian spectrum's square-root factor.
``sample_evaporation`` evaluates the temperature and uses rejection sampling within the restricted energy interval.
Both analytic samplers return zero energy if the temperature is nonpositive or E is no greater than U.
The angle is sampled from a separate angular Distribution.

.. _ace-law-44:

LAW 44: Kalbach–Mann
^^^^^^^^^^^^^^^^^^^^^^

This joint distribution combines an outgoing-energy PDF with an analytic conditional cosine density:

.. math::

   p(\mu\mid E,E')=\frac{a}{2\sinh a}
   \left[\cosh(a\mu)+r\sinh(a\mu)\right],
   \qquad -1\le\mu\le 1.

The precompound fraction r lies in [0, 1], and the angular slope a is nonnegative.
The a = 0 limit is isotropic.

The outgoing-energy PDFs are packed under ``tables/outgoing-energy``.
Each incident table also has ``precompound-factor`` and ``angular-slope`` Data records.
ACE supplies these parameters on its outgoing-energy grid; ``mcdc-data`` also permits independent parameter grids and polynomial Data.

Transport selects an incident table and samples its energy PDF as described in :ref:`ace-incident-energy-sampling`.
It evaluates r and a at the sampled energy before support scaling, then samples the conditional cosine using those parameters.
Parameter tables clamp to their endpoint values outside their own grids.
The energy and angle come from the same selected incident table; no separate angular Distribution is sampled.

.. _ace-law-61:

LAW 61: Tabulated energy-angle distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This joint distribution replaces the analytic Kalbach–Mann cosine law with tabulated conditional cosine PDFs.
There is one cosine table for each outgoing-energy point in each incident table.

After incident-table selection, energy sampling identifies an interval j and a cumulative probability u within that table.
For histogram density interpolation, use the cosine table at the lower outgoing-energy point j.
For linear density interpolation, select point j or j + 1 according to which CDF endpoint, C[j] or C[j + 1], is closer to u; ties select j.
The comparison uses the selected table's CDF before unit-base energy scaling.
Sample the chosen cosine PDF using its own interpolation and an independent random number.

These histogram and linear rules follow the `MCNP LAW 61 sampling description (page 2-47) <https://mcnpx.lanl.gov/pdf_files/TechReport_2003_LANL_LA-UR-03-1987Revised212008_SweezyBoothEtAl.pdf#page=81>`_.
MC/DC applies the same CDF-distance rule to the logarithmic density interpolation laws supported by ``mcdc-data``.

.. _ace-law-66:

LAW 66: N-body phase space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For N final-state bodies, the normalized energy fraction x has density

.. math::

   p(x)\propto\sqrt{x}(1-x)^{3N/2-4},
   \qquad 0\le x\le 1.

The converter preserves the source's dimensionless fraction PDF, final-state particle count N, and total-mass ratio R.
R is the total final-state mass divided by the emitted particle mass.
The source tabulation follows the `NJOY phase-space model <https://github.com/njoy/NJOY2016-manual/blob/master/acex.tex>`_.

Transport samples x using the density's interpolation, then obtains the outgoing COM energy from

.. math::

   E'=xE'_{\max},
   \qquad
   E'_{\max}=\frac{R-1}{R}\left(\frac{A}{A+1}E+Q\right).

Here A is the target-to-projectile mass ratio and Q is the reaction Q-value in eV.
The energy scale is zero when no positive energy is available.
The maximum-energy relation is defined by `ACE LAW 66 <https://github.com/NuclearData/ACEFormat/blob/master/src/ContinuousEnergyNeutron.tex#L867>`_.

Emission is isotropic in COM and independent of outgoing COM energy, so transport samples :math:`\mu=2\xi-1` with an independent uniform random number.
The reaction requires ``reference-frame = COM`` and has no separate angular-cosine Distribution.
Emission in the laboratory frame is generally anisotropic after the kinematic transformation.
The dimensionless density supports all five Interpolation laws and multiple regions in ``mcdc-data``; ACE supplies histogram or linear interpolation.

Angular representations
-----------------------

Separate angular data come from ACE's LAND/AND blocks rather than the numbered energy laws above.

.. list-table:: Angular representation mapping
   :header-rows: 1
   :widths: 25 20 25 30

   * - ACE representation
     - ``mcdc-data``
     - Object class
     - Transport handling
   * - Fully isotropic angular distribution
     - ``isotropic``
     - ``DistributionIsotropic``
     - ``sample_isotropic_cosine``
   * - Incident-energy-dependent tabulated cosine PDFs
     - ``multi-table``
     - ``DistributionMultiTable``
     - ``_sample_multi_table`` without outgoing-energy support scaling
   * - Angles supplied by a joint energy-angle law
     - Angular field omitted
     - ``DistributionNone`` for the separate angular field
     - Angle sampled together with energy by law 44, 61, or 66

Separate incident-energy-dependent angular records must contain tabulated PDFs at every incident energy.
The converter does not accept 32-equiprobable-bin angular records or isotropic entries embedded within such a table sequence.
An omitted angular field denotes joint angle sampling, not isotropy.

Composition and runtime
-----------------------

For inelastic reactions with multiple spectra, each component uses one of the energy representations above.
The reaction's ``spectrum-probability-grid`` and ``spectrum-probability`` fields select the component; no separate mixture Distribution is required.
Selection probabilities must be constant or histogram tables on a common incident-energy grid.
Prompt fission accepts one energy-spectrum component, and delayed-neutron spectra use law 4.

``mcdc.object_.hdf5_reader`` constructs the corresponding Distribution and Data objects from the stored groups.
Compilation places their numeric fields in typed runtime records and flat arrays; transport samples those records without accessing HDF5.
Scalar energy and angular distributions use ``sample_distribution_with_scale`` and ``sample_distribution``, respectively.
Joint energy-angle distributions use ``sample_correlated_distribution_with_scale``.
Reaction transport applies the declared reference-frame transformation after sampling.
