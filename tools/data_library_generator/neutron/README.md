# MC/DC Data Library Generator

Converts ACE-format nuclear data files into MC/DC's per-nuclide HDF5 format
for continuous-energy neutron transport.

## Prerequisites

```bash
pip install h5py numpy tqdm
```

## Install ACEtk from source

[ACEtk](https://github.com/njoy/ACEtk)

You need a collection of ACE files (e.g., from NJOY or an ENDF/B distribution).

## Environment Variables

| Variable      | Description                                           |
|---------------|-------------------------------------------------------|
| `MCDC_ACELIB` | Path to the directory containing your ACE files.      |
| `MCDC_LIB`    | Path to the output directory for MC/DC HDF5 files.    |

## Usage

```bash
export MCDC_ACELIB=/path/to/ace/files
export MCDC_LIB=/path/to/mcdc/library

python generate.py              # Convert only missing nuclides
python generate.py --rewrite    # Regenerate all files
python generate.py --verbose    # Print detailed per-nuclide info
```

## What it Does

For each ACE file in `$MCDC_ACELIB`, the generator:

1. Reads the ACE header to identify the nuclide (Z, A, isomeric state) and temperature.
2. Sorts reactions into groups and extracts their pointwise cross sections and the energy grid:
   elastic is MT-2; fission is MT-18 (or MT-19/20/21/38); capture is any other reaction with
   zero multiplicity; inelastic is any other reaction with non-zero multiplicity. MT-1, 3, 4, 10
   and MT > 117 are ignored.
3. Extracts angular distributions (tabulated cosine PDFs) and energy distributions
   (level scattering, evaporation, Maxwellian, Kalbach-Mann, N-body, tabulated) per reaction channel.
4. For fissionable nuclides, extracts prompt/delayed ν(E), precursor fractions, decay constants, and energy spectra.
5. Writes a single HDF5 file per nuclide-temperature combination (e.g., `U235-293.6K.h5`).

## Output HDF5 Schema

```
<Nuclide>-<Temperature>K.h5          file attrs: source_title, source_version, source_date,
│                                                source_comments (if present in the ACE header)
├── nuclide_name                     (string)
├── excitation_level                 (int, 0 = ground state)
├── temperature                      (float, K)
├── atomic_number                    (int)
├── mass_number                      (int)
├── atomic_weight_ratio              (float)
├── fissionable                      (bool)
└── neutron_reactions/
    ├── xs_energy_grid               (1-D array, MeV; shared by every reaction)
    ├── elastic_scattering/
    │   └── MT-002/                  (attr: MT)
    │       ├── xs                   (1-D array, barns; attr: offset = 0)
    │       ├── Q-value              (float, MeV; always 0)
    │       ├── reference_frame      (string, always "COM")
    │       └── angular_cosine_distribution/
    ├── capture/                     (reactions with zero multiplicity)
    │   └── MT-xxx/                  (attr: MT)
    │       ├── xs                   (1-D array, barns; attr: offset)
    │       ├── Q-value              (float, MeV)
    │       └── reference_frame      (string, "COM" or "LAB")
    ├── inelastic_scattering/        (omitted if the nuclide has none)
    │   └── MT-xxx/                  (attr: MT)
    │       ├── xs                   (1-D array, barns; attr: offset)
    │       ├── Q-value              (float, MeV)
    │       ├── reference_frame      (string, "COM" or "LAB")
    │       ├── multiplicity         (int, number of emitted neutrons)
    │       ├── angular_cosine_distribution/
    │       ├── spectrum_probability_grid   (1-D array, MeV)
    │       ├── spectrum_probability        (2-D array, [N_grid-1, N_spectra])
    │       └── energy_spectrum-N/          (N = 1, 2, ...; one per spectrum)
    └── fission/                     (omitted if not fissionable)
        ├── MT-018/                  (or MT-019/020/021/038 if MT-018 is not given)
        │   ├── xs                   (1-D array, barns; attr: offset)
        │   ├── Q-value              (float, MeV)
        │   ├── reference_frame      (string, "COM" or "LAB")
        │   ├── angular_cosine_distribution/
        │   ├── spectrum_probability_grid, spectrum_probability   (as in inelastic)
        │   └── energy_spectrum-1/   (prompt fission spectrum; only one is supported)
        ├── prompt_multiplicity/     (attr: type = "polynomial" | "tabulated")
        ├── delayed_multiplicity/    (same layout; only if the ACE file has delayed data)
        └── delayed_neutron_precursors/
            ├── fractions            (1-D array, one per precursor group)
            ├── decay_rates          (1-D array, 1/s)
            └── energy_spectrum-N/   (N = 1..number of groups; delayed spectrum)
```

Where the attribute `xs:offset` is the index of the first non-zero cross section value for a given reaction. This avoids storing zero cross-sections.

Note that all energy values are recorded in MeV (to be converted to eV at simulation time).

### `angular_cosine_distribution/`

Group attribute `type` selects the layout:

| `type`              | Datasets                                                                      |
|---------------------|-------------------------------------------------------------------------------|
| `isotropic`         | none                                                                          |
| `energy-correlated` | none (the cosine is determined by the sampled energy distribution)            |
| `tabulated`         | `energy` (MeV), `offset` (int), `value` (cosines), `pdf`                      |

For For `tabulated` distributions, `energy` holds the incident-energy grid. `value` and `pdf` contain the table of outgoing angle and probability,
for each incident energy, joined together into 1-D arrays. For incident energy of index `i`, the value `offset[i]` is the index in `value`/`pdf` where
the table for that incident energy starts. Note that this is different than the cross-section attribute with the same name.
Interpolation is linear.

### `energy_spectrum-N/`

Group attribute `type` selects the layout. "Table" types concatenate one outgoing-energy table per
incident energy, with `offset[i]` giving the start index of table `i`.

| `type`                  | Datasets                                                                                                      |
|-------------------------|---------------------------------------------------------------------------------------------------------------|
| `level-scattering`      | `C1` (MeV), `C2`                                                                                              |
| `evaporation`           | `temperature_energy_grid` (MeV), `temperature` (MeV), `restriction_energy` (MeV), `temperature_interpolations`, `interpolation_boundaries` |
| `maxwellian`            | same as `evaporation`                                                                                         |
| `tabulated`             | `energy` (MeV), `offset`, `value` (outgoing energy, MeV), `pdf`                                               |
| `kalbach-mann`          | `energy` (MeV), `offset`, `energy_out` (MeV), `pdf`, `precompound_factor`, `angular_slope`                    |
| `energy-angle-tabulated`| `energy` (MeV), `offset`, `energy_out` (MeV), `pdf`, `cosine_offset`, `cosine`, `cosine_pdf`                  |
| `N-body`                | `value` (MeV), `pdf`                                                                                          |

### Fission multiplicity (`prompt_multiplicity/`, `delayed_multiplicity/`)

| `type`       | Datasets                                                                                      |
|--------------|-----------------------------------------------------------------------------------------------|
| `polynomial` | `coefficient` (attr `unit-base` = MeV: ν = Σ c_l E^l with E in MeV)                           |
| `tabulated`  | `energy` (MeV), `value`; linear interpolation                                                 |

## See Also

- [Continuous Energy Theory Guide](../../../docs/source/theory/continuous_energy.rst)
- [Installation — CE Library Configuration](../../../docs/source/user_guide/getting_started/installation.rst)
