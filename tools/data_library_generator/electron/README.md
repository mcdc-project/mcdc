# MC/DC Electron Data Library Generator
Converts EPRDATA14 ACE-format electron/photon/relaxation data into MC/DC's
per-element HDF5 format for continuous-energy electron transport.

## Prerequisites
```bash
pip install ACEtk h5py numpy tqdm
```
You need the EPRDATA14 library (available from [LANL Nuclear Data](https://nucleardata.lanl.gov/ace/eprdata14)).

## Environment Variables
| Variable                     | Description                                        |
|------------------------------|----------------------------------------------------|
| `MCDC_ACELIB_ELECTRON`       | Path to the EPRDATA14 data file.                   |
| `MCDC_ACELIB_ELECTRON_XSDIR` | Path to the EPRDATA14 `xsdir` file.                |
| `MCDC_LIB_ELECTRON`          | Path to the output directory for MC/DC HDF5 files. |

## Usage
```bash
export MCDC_ACELIB_ELECTRON=/path/to/eprdata14/eprdata14
export MCDC_ACELIB_ELECTRON_XSDIR=/path/to/eprdata14/xsdir
export MCDC_LIB_ELECTRON=/path/to/mcdc/electron/library

python generate_electron.py              # Convert only missing elements
python generate_electron.py --rewrite    # Regenerate all files
python generate_electron.py --verbose    # Print detailed per-element info
```

## What it Does
For each element (Z=1 to Z=100) in the EPRDATA14 library, the generator:
1. Reads the xsdir to locate each elemental table within the single EPRDATA14 file.
2. Extracts the principal cross section energy grid and pointwise cross sections
   (elastic, bremsstrahlung, excitation, electroionization) for all reaction channels.
3. Extracts tabulated elastic angular distributions (cosine PDFs) per incident energy (MT-528).
4. Extracts excitation energy loss as a function of incident energy (MT-527).
5. Extracts bremsstrahlung outgoing photon energy distributions per incident energy (MT-526).
6. Extracts per-subshell electroionization cross sections and knock-on electron
   energy distributions (MT-534 for K-shell, MT-535+ for higher shells).
7. Extracts atomic relaxation (fluorescence and Auger) transition data per subshell.
8. Writes a single HDF5 file per element (e.g., `Al.h5`).

## Output HDF5 Schema
```
<Symbol>.h5
├── element_symbol                              (string)
├── atomic_number                               (int)
├── atomic_weight_ratio                         (float)
├── electron_reactions/
│   ├── xs_energy_grid                          (1-D array, MeV)
│   ├── elastic_scattering/
│   │   └── MT-528/
│   │       ├── xs                              (1-D array, barns)
│   │       └── angular_cosine_distribution/
│   │           ├── energy                      (1-D array, MeV)
│   │           ├── offset                      (1-D array, int)
│   │           ├── cosine                      (1-D array)
│   │           ├── pdf                         (1-D array)
│   │           └── cdf                         (1-D array)
│   ├── excitation/
│   │   └── MT-527/
│   │       ├── xs                              (1-D array, barns)
│   │       └── energy_loss/
│   │           ├── energy                      (1-D array, MeV)
│   │           └── energy_loss                 (1-D array, MeV)
│   ├── bremsstrahlung/
│   │   └── MT-526/
│   │       ├── xs                              (1-D array, barns)
│   │       └── energy_distribution/
│   │           ├── energy                      (1-D array, MeV)
│   │           ├── offset                      (1-D array, int)
│   │           ├── energy_out                  (1-D array, MeV)
│   │           └── pdf                         (1-D array)
│   └── electroionization/
│       └── MT-534/  (K-shell; MT-535, MT-536, ... for higher shells)
│           ├── xs                              (1-D array, barns)
│           ├── binding_energy                  (float, MeV)
│           └── energy_distribution/
│               ├── energy                      (1-D array, MeV)
│               ├── offset                      (1-D array, int)
│               ├── energy_out                  (1-D array, MeV)
│               └── pdf                         (1-D array)
└── atomic_relaxation/
    └── MT-534/  (K-shell; MT-535, MT-536, ... for higher shells)
        ├── number_of_transitions               (int)
        ├── primary_subshell                    (1-D array, int)
        ├── secondary_subshell                  (1-D array, int)
        ├── energy                              (1-D array, MeV)
        └── probability                         (1-D array)
```

## See Also
- [Continuous Energy Theory Guide](../../docs/source/theory/cont_energy.rst)
- [Installation Guide — CE Library Configuration](../../docs/source/install.rst)
