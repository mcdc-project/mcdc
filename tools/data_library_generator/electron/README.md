# MC/DC Electron Data Library Generator
Converts EPRDATA14 ACE-format electron/photon/relaxation data into MC/DC's
per-element HDF5 format for continuous-energy electron transport.

## Prerequisites

- Installing ACEtk from source: [link](https://github.com/njoy/ACEtk)
- Dependencies: `pip install h5py numpy tqdm`
- You need the EPRDATA14 library (available from [LANL Nuclear Data](https://nucleardata.lanl.gov/ace/eprdata14)).

## Environment Variables
| Variable               | Description                                        |
|------------------------|----------------------------------------------------|
| `MCDC_ACELIB_ELECTRON` | Path to the EPRDATA14 data file.                   |
| `MCDC_LIB_ELECTRON`    | Path to the output directory for MC/DC HDF5 files. |

## Usage
```bash
export MCDC_ACELIB_ELECTRON=/path/to/eprdata14/eprdata14/eprdata14
export MCDC_LIB_ELECTRON=/path/to/mcdc/electron/library

python generate.py              # Convert only missing elements
python generate.py --rewrite    # Regenerate all files
python generate.py --verbose    # Print detailed per-element info
```

## What it Does
For each element (Z=1 to Z=100) in the EPRDATA14 library, the generator:
1. Loads all elemental tables from the single concatenated EPRDATA14 file.
2. Extracts the shared cross section energy grid and pointwise cross sections
   (elastic, excitation, bremsstrahlung, and total ionization summed over subshells).
3. Stores the total elastic cross section (small and large angles) under MT-526,
   the large-angle cross section under `large_angle/xs`, and the tabulated
   large-angle scattering cosine CDFs.
4. Extracts the excitation average energy loss as a function of incident energy (MT-528).
5. Extracts the bremsstrahlung average energy loss and the photon energy CDFs as a
   function of incident energy (MT-527).
6. Extracts per-subshell ionization cross sections, binding energies, and knock-on
   electron energy CDFs, grouped under the total ionization reaction (MT-522).
7. Extracts atomic relaxation (fluorescence and Auger) transition data per subshell.
8. Writes a single HDF5 file per element (e.g., `Al.h5`).

Existing HDF5 files are skipped by default. Use `--rewrite` to regenerate them
after changing the schema or energy units.

## Output HDF5 Schema
```
<Symbol>.h5
├── element_symbol                              (string)
├── atomic_number                               (int)
├── atomic_weight_ratio                         (float)
├── electron_reactions/
│   ├── xs_energy_grid                          (1-D array, eV)
│   ├── elastic_scattering/
│   │   └── MT-526/                             (attr: MT)
│   │       ├── reference_frame                 (string: "LAB")
│   │       ├── xs                              (1-D array, barns; attr: offset; total elastic)
│   │       ├── transport                       (1-D array, barns)
│   │       └── large_angle/                     (attr: MT = 525)
│   │           ├── xs_energy                   (1-D array, eV)
│   │           ├── xs                          (1-D array, barns; large-angle elastic)
│   │           └── scattering_cosine/
│   │               ├── energy_grid             (1-D array, eV)
│   │               ├── energy_offset           (1-D array, int)
│   │               ├── value                   (1-D array, cosine)
│   │               └── CDF                     (1-D array)
│   ├── excitation/
│   │   └── MT-528/                             (attr: MT)
│   │       ├── reference_frame                 (string: "LAB")
│   │       ├── xs                              (1-D array, barns; attr: offset)
│   │       └── energy_loss/
│   │           ├── energy                      (1-D array, eV)
│   │           └── value                       (1-D array, eV)
│   ├── bremsstrahlung/
│   │   └── MT-527/                             (attr: MT)
│   │       ├── reference_frame                 (string: "LAB")
│   │       ├── xs                              (1-D array, barns; attr: offset)
│   │       ├── energy_loss/
│   │       │   ├── energy                      (1-D array, eV)
│   │       │   └── value                       (1-D array, eV; mean photon energy)
│   │       └── photon_energy/
│   │           ├── energy_grid                 (1-D array, eV)
│   │           ├── energy_offset               (1-D array, int)
│   │           ├── value                       (1-D array, eV)
│   │           └── CDF                         (1-D array)
│   └── ionization/
│       └── MT-522/                             (attr: MT; all subshells grouped here)
│           ├── reference_frame                 (string: "LAB")
│           ├── xs                              (1-D array, barns; attr: offset; sum over subshells)
│           └── subshells/
│               └── MT-NNN/                     (attrs: MT = ENDF subshell MT, 534+; subshell)
│                   ├── energy_grid             (1-D array, eV)
│                   ├── xs                      (1-D array, barns)
│                   ├── binding_energy          (float, eV)
│                   └── product/
│                       ├── energy_grid         (1-D array, eV)
│                       ├── energy_offset       (1-D array, int)
│                       ├── value               (1-D array, eV)
│                       └── CDF                 (1-D array)
└── atomic_relaxation/
    └── MT-NNN/                                 (attrs: MT; subshell; one per ionization subshell, 534+)
        ├── number_of_transitions               (int)
        ├── primary_designator                  (1-D array, int)   [if number_of_transitions > 0]
        ├── secondary_designator                (1-D array, int)   [if number_of_transitions > 0]
        ├── energy                              (1-D array, eV)    [if number_of_transitions > 0]
        └── probability                         (1-D array)        [if number_of_transitions > 0]
```

## See Also
- [Continuous Energy Theory Guide](../../../docs/source/theory/continuous_energy.rst)
- [Installation — CE Library Configuration](../../../docs/source/user_guide/getting_started/installation.rst)
