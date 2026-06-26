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
2. Extracts pointwise cross sections (elastic, capture, inelastic, fission) and the energy grid.
3. Extracts angular distributions (tabulated cosine PDFs) and energy distributions
   (level scattering, evaporation, Maxwellian, Kalbach-Mann, N-body, tabulated) per reaction channel.
4. For fissionable nuclides, extracts prompt/delayed ν(E), precursor fractions, decay constants, and energy spectra.
5. Writes a single HDF5 file per nuclide-temperature combination (e.g., `U235-293.6K.h5`).

## Output HDF5 Schema

```
<Nuclide>-<Temperature>K.h5
├── nuclide_name              (string)
├── temperature               (float, K)
├── atomic_weight_ratio       (float)
├── fissionable               (bool)
└── neutron_reactions/
    ├── xs_energy_grid         (1-D array, MeV)
    ├── elastic_scattering/
    │   └── MT-002/
    │       ├── xs             (1-D array, barns)
    │       ├── cosine/
    │       └── energy/
    ├── capture/
    │   └── MT-102/ ...
    ├── inelastic_scattering/
    │   └── MT-051/ ...
    └── fission/
        └── MT-018/
            ├── xs
            ├── cosine/
            ├── energy/
            ├── nu_total/
            ├── nu_prompt/
            ├── nu_delayed/
            └── delayed_neutron/ ...
```

## See Also

- [Continuous Energy Theory Guide](../../docs/source/theory/cont_energy.rst)
- [Installation Guide — CE Library Configuration](../../docs/source/install.rst)
