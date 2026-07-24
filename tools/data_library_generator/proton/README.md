# MC/DC Proton Data Library Generator
Converts ACE-format proton cross section data from TENDL2021 into MC/DC's
per-nuclide HDF5 format for proton transport. Also uses the NIST PSTAR
database to add stopping power data to the HDF5 files.

## Prerequisites

- Installing ACEtk from source: [link](https://github.com/njoy/ACEtk)
- Dependencies: `pip install h5py numpy tqdm`
- For any nuclide that you want stopping power for, download the stopping power data from NIST'S PSTAR database: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
    - When downloading the stopping power data, select only the total stopping power. This generator script uses the load_pstar_file function to extract the data assuming two columns of data.
- You need the TENDL2021 ACE files for protons. Avaiable at this link: https://tendl.imperial.ac.uk/tendl_2021/tar.html
    - Or, download the tar file directly: https://tendl.imperial.ac.uk/tendl_2021/tar_files/TENDL-ACE-p.tgz

## Environment Variables
| Variable               | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| `MCDC_ACELIB_PROTON`   | Path to the directory containing the TENDL2021 ACE files.              |
| `MCDC_LIB_PROTON`      | Path to the output directory for MC/DC HDF5 files.                     |
| `MCDC_PSTAR_LIB`       | Path to the directory containing the PSTAR stopping power table files. |

## Usage
```bash
export MCDC_ACELIB_PROTON=/path/to/tendl2021/acefiles
export MCDC_LIB_PROTON=/path/to/mcdc/proton/library

python generate.py              # Convert only missing elements
python generate.py --rewrite    # Regenerate all files
python generate.py --verbose    # Print detailed per-element info
```

## What it Does
For each element (Z=1 to Z=103) in the TENDL2021 ACE file library, the generator:
1. Loads the data from the ACE table, and writes basic data (name, temperature, mass, etc.) to the HDF5 file.
2. Extracts the stopping power data (if present in $MCDC_PSTAR_LIB).
3. Extracts the MT numbers for elastic scattering reactions and their cross sections.
4. Extracts the MT numbers for capture reactions and their cross sections.
5. Extracts the MT numbers for inelastic scattering reactions and their cross sections.
6. Extracts the energy & angular distributions for scattering reactions.
7. Creates a data block in the HDF5 file to handle the secondary particle products & energies.
8. If there are isotopes present with PSTAR stopping power data, but without an ACE file from TENDL, it creates
    an HDF5 file that contains the stopping power data.

## Output HDF5 Schema
```
File attrs: source_title, source_version, source_date, source_comments (if present)

<Nuclide>-<T>K.h5
├── nuclide_name                                (string)
├── excitation_level                            (int)
├── temperature                                 (float; attr: unit="K")
├── atomic_number                               (int)
├── mass_number                                 (int)
├── atomic_weight_ratio                         (float)
├── radiation_length                            (float; attr: unit="g/cm2")
├── fissionable                                 (bool)
├── stopping_power/                             (present only if a matching PSTAR file was found)
│   ├── energy                                  (1-D array; attr: unit="MeV")
│   └── total_stopping_power                    (1-D array; attr: unit="MeV cm2/g")
├── proton_reactions/
│   ├── xs_energy_grid                          (1-D array; attr: unit="MeV")
│   ├── elastic_scattering/
│   │   └── MT-002/                             (attr: MT=2)
│   │       ├── xs                              (1-D array, barns; attr: offset=0)
│   │       ├── Q-value                         (float=0.0; attr: unit="MeV")
│   │       ├── reference_frame                 (string: "COM")
│   │       └── angular_cosine_distribution/    (attr: type="energy-correlated"; see [A] below)
│   ├── capture/                                (one group per capture MT, i.e. multiplicity=0)
│   │   └── MT-NNN/                             (attr: MT)
│   │       ├── xs                              (1-D array, barns; attr: offset)
│   │       ├── Q-value                         (float; attr: unit="MeV")
│   │       └── reference_frame                 (string: "LAB" or "COM")
│   ├── inelastic_reaction/                     (present only if any inelastic MTs exist)
│   │   └── MT-NNN/                             (attr: MT; one per inelastic MT)
│   │       ├── xs                              (1-D array, barns; attr: offset)
│   │       ├── Q-value                         (float; attr: unit="MeV")
│   │       ├── reference_frame                 (string: "LAB" or "COM")
│   │       ├── multiplicity                    (int)
│   │       ├── angular_cosine_distribution/    (see [A] below)
│   │       ├── spectrum_probability_grid       (1-D array; attr: unit="MeV")
│   │       ├── spectrum_probability            (2-D array [grid x n_dist])
│   │       └── energy_spectrum-N/              (one per outgoing-energy law; see [B] below)
│   └── fission/                                (present only if fissionable)
│       ├── MT-NNN/                             (attr: MT; MT-018 or the fission-chance MTs 19/20/21/38)
│       │   ├── xs                              (1-D array, barns; attr: offset)
│       │   ├── Q-value                         (float; attr: unit="MeV")
│       │   ├── reference_frame                 (string: "LAB" or "COM")
│       │   ├── angular_cosine_distribution/    (see [A] below)
│       │   ├── spectrum_probability_grid       (1-D array; attr: unit="MeV")
│       │   ├── spectrum_probability            (2-D array [grid x n_dist])
│       │   └── energy_spectrum-N/              (see [B] below)
│       ├── prompt_multiplicity/                (see [C] below)
│       ├── delayed_multiplicity/               (optional; see [C] below)
│       └── delayed_neutron_precursors/         (optional)
│           ├── fractions                       (1-D array, one per precursor group)
│           ├── decay_rates                     (1-D array; attr: unit="/s")
│           └── energy_spectrum-N/              (one per precursor group; see [B] below)
└── secondary_particles/                        (present only if ACE table has secondary-particle data)
    └── ZAP_<zap>/                              (attrs: ZAP, particle_name; one per secondary particle type)
        └── MT-NNN/                             (attrs: MT, multiplicity, reference_frame)
            ├── production_xs                   (1-D array, barns; attr: offset)
            ├── kalbach_mann/                    (attr: type="kalbach-mann")
            │   ├── energy                      (1-D array; attr: unit="MeV")
            │   ├── offset                      (1-D int array, one entry per incident energy)
            │   ├── energy_out                  (1-D array; attr: unit="MeV")
            │   ├── pdf                         (1-D array)
            │   ├── cdf                         (1-D array)
            │   ├── precompound_factor          (1-D array)
            │   └── angular_slope               (1-D array)
            └── angular_cosine_distribution/    (see [A] below)


[A] angular_cosine_distribution/ schema (load_cosine_distribution):
  attr: type="given_in_energy_distribution"     (angular data embedded in the energy-distribution block — no other content)
  — or —
  attr: type="tabulated"; attr: unit="MeV"
  ├── incident_energies                         (1-D array)
  └── E_in_i/                                   (one group per incident energy)
      attr: type="tabulated" →  cosines, pdf, cdf   (1-D arrays)
      attr: type="isotropic" →  (no datasets)

[B] energy_spectrum-N/ schema (load_energy_distribution), attr: law = ENDF law number:
  law=44 (Kalbach-Mann):
      attr: type="kalbach-mann"
      ├── energy               (1-D array; attr: unit="MeV")
      ├── offset               (1-D int array)
      ├── energy_out           (1-D array; attr: unit="MeV")
      ├── pdf, cdf             (1-D arrays)
      ├── precompound_factor   (1-D array)
      └── angular_slope        (1-D array)
  law=4 (tabulated outgoing energy):
      ├── incident_energies    (1-D array)
      └── E_in_k/ → outgoing_energies, pdf, cdf   (1-D arrays)
  law=3 (level scattering):
      ├── C1                   (float)
      └── C2                   (float)
  law=1 (equiprobable bins):
      ├── incident_energies    (1-D array)
      └── E_in_k/ → energies   (1-D array)
  law=-1 (unrecognized type):
      attr: type_name=<ACEtk class name>
      └── xss_array            (1-D array; only if extraction succeeds)

[C] prompt_multiplicity/ and delayed_multiplicity/ schema (load_fission_multiplicity):
  attr: type="tabulated"  → energies, multiplicities   (1-D arrays)
  attr: type="polynomial" → coefficients                (1-D array)
  attr: type="unknown"    → attr: type_name=<ACEtk class name>  (no data)


── Stopping-power-only fallback (process_pstar_only_file; H-1, H-2, He-3, He-4 when no ACE file exists) ──

<Symbol><A>-<T>K.h5
├── nuclide_name                                (string)
├── excitation_level                            (int = 0)
├── temperature                                 (float; attr: unit="K")
├── atomic_number                               (int)
├── mass_number                                 (int)
├── atomic_weight_ratio                         (float)
├── radiation_length                            (float; attr: unit="g/cm2")
├── fissionable                                 (bool = False)
└── stopping_power/
    ├── energy                                  (1-D array; attr: unit="MeV")
    └── total_stopping_power                    (1-D array; attr: unit="MeV cm2/g")
```

## See Also
- [TENDL2021](https://tendl.imperial.ac.uk/tendl_2021/tendl2021.html)
