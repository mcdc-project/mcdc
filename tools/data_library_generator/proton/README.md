# MC/DC Proton Data Library Generator

Converts ACE-format proton cross sections from TENDL-2021 into MC/DC's
per-nuclide HDF5 format for proton transport. The generator also reads NIST
PSTAR stopping-power tables and merges them into the output files when they are
available.

## Prerequisites

- Install ACEtk from source: [ACEtk](https://github.com/njoy/ACEtk)
- Install Python dependencies:

```bash
pip install h5py numpy tqdm
```

- Download the TENDL-2021 proton ACE library, for example from:
  - https://tendl.imperial.ac.uk/tendl_2021/tar.html
  - https://tendl.imperial.ac.uk/tendl_2021/tar_files/TENDL-ACE-p.tgz
- For stopping power, download the total stopping-power files from NIST PSTAR:
  - https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
  - The script expects one file per element, named like `Al.txt`, with two columns: energy and stopping power.

## Environment Variables

| Variable | Description |
| --- | --- |
| `MCDC_ACELIB_PROTON` | Directory containing the TENDL-2021 proton ACE files (`*.ace`). |
| `MCDC_PSTAR_LIB` | Directory containing the PSTAR stopping-power tables (`<Symbol>.txt`). |
| `MCDC_LIB_PROTON` | Output directory for the generated MC/DC HDF5 files. |

## Usage

```bash
export MCDC_ACELIB_PROTON=/path/to/tendl2021/acefiles
export MCDC_PSTAR_LIB=/path/to/pstar/tables
export MCDC_LIB_PROTON=/path/to/mcdc/proton/library

python generate.py                 # Convert only files that do not already exist
python generate.py --rewrite       # Regenerate everything
python generate.py --verbose       # Print per-file progress and data details
python generate.py --only p-Al027.ace  # Convert only selected ACE files
```

## What it Does

For each ACE file in `MCDC_ACELIB_PROTON`, the generator:

1. Reads the ACE header to determine the nuclide and excitation state.
2. Writes the basic HDF5 metadata (nuclide name, temperature, mass, atomic weight, radiation length, and fission flag).
3. Reads the principal cross-section block and stores the universal energy grid.
4. Writes the elastic scattering reaction (`MT-002`) from the principal block.
5. Classifies the remaining reactions into capture and inelastic-scattering groups, ignoring redundant sum MTs and unsupported proton-fission entries. (Capture is assumed to be any nuclear reaction which has an incident proton and no outgoing proton; inelastic scattering reactions are assumed to have an incident proton and any nonzero number of outgoing protons.)
6. Writes per-reaction cross sections, `Q-value`, reference frame, multiplicity, angular distributions, and outgoing-proton spectra.
7. Writes any secondary-particle channels present in the ACE file (neutrons, deuterons, tritons, He-3, alphas, and photons), including their yield tables, energy spectra, and angular distributions.
8. If `MCDC_PSTAR_LIB` contains a matching file for an element, adds `stopping_power/energy` and `stopping_power/total_stopping_power` to the HDF5 file.
9. For H-1, H-2, He-3, and He-4, creates a minimal stopping-power-only HDF5 file when no matching ACE file exists but a PSTAR table is present.

## Output HDF5 Schema

```text
<Nuclide>-0K.h5
├── source_title                              (HDF5 file attribute: string)
├── source_version                            (HDF5 file attribute: string)
├── source_date                               (HDF5 file attribute: string)
├── source_comments                           (optional HDF5 file attribute: string)
├── nuclide_name                              (string)
├── excitation_level                          (int)
├── temperature                               (float; attr: unit="K")
├── atomic_number                             (int)
├── mass_number                               (int)
├── atomic_weight_ratio                       (float)
├── radiation_length                          (float; attr: unit="g/cm2")
├── fissionable                               (bool)
├── stopping_power/                           (present only when a PSTAR file is available)
│   ├── energy                                (1-D array; unit: MeV)
│   └── total_stopping_power                  (1-D array; unit: MeV cm2/g)
├── proton_reactions/
│   ├── xs_energy_grid                        (1-D array; unit: MeV)
│   ├── elastic_scattering/
│   │   └── MT-002/
│   │       ├── xs                            (1-D array, barns; attr: offset=0)
│   │       ├── Q-value                       (float; unit: MeV)
│   │       ├── reference_frame               (string: "COM")
│   │       └── angular_cosine_distribution/  (tabulated, isotropic, or given in the energy block)
│   ├── capture/
│   │   └── MT-NNN/
│   │       ├── xs                            (1-D array, barns; attr: offset)
│   │       ├── Q-value                       (float; unit: MeV)
│   │       └── reference_frame               (string: "LAB" or "COM")
│   └── inelastic_scattering/
│       └── MT-NNN/
│           ├── xs                            (1-D array, barns; attr: offset)
│           ├── Q-value                       (float; unit: MeV)
│           ├── reference_frame               (string: "LAB" or "COM")
│           ├── multiplicity                  (int)
│           ├── proton_yield/                 (optional; used when the yield is energy-dependent)
│           │   ├── energy                    (1-D array; unit: MeV)
│           │   └── value                     (1-D array)
│           ├── angular_cosine_distribution/  (tabulated, isotropic, or given in the energy block)
│           ├── spectrum_probability_grid     (1-D array; unit: MeV)
│           ├── spectrum_probability          (2-D array [grid x n_dist])
│           └── energy_spectrum-1/            (or energy_spectrum-N; one group per distribution)
│               └── ...                      (layout depends on the ACE law: tabulated, kalbach-mann, or level-scattering)
│
│       └── secondary_products/              (created under each reaction group when non-primary products are present)
│           ├── neutron/
│           │   ├── ZAP                      (attribute: int)
│           │   ├── particle_name            (attribute: string)
│           │   ├── ace_particle_id          (attribute: int)
│           │   ├── reference_frame          (attribute: string: "LAB" or "COM")
│           │   ├── yield/
│           │   │   ├── energy              (1-D array; unit: MeV)
│           │   │   └── value               (1-D array)
│           │   ├── kalbach_mann/           (or energy_spectrum/; one distribution type only)
│           │   │   └── ...
│           │   └── angular_cosine_distribution/
│           │       └── ...
│           ├── deuteron/
│           │   └── ...
│           ├── triton/
│           │   └── ...
│           ├── alpha/
│           │   └── ...
│           └── photon/
│               └── ...
```

The angular and energy distribution layout follows the actual ACE conversion logic in `generate.py`:

- `angular_cosine_distribution` stores either a tabulated angular PDF/CDF, an isotropic distribution, or a marker that the angle information is given elsewhere in the ACE energy block.
- `energy_spectrum-N` groups store the outgoing-energy law for each distribution, including `tabulated`, `kalbach-mann`, and `level-scattering` layouts.
- `secondary_particles` contains the product-specific yield and spectrum information for each non-primary particle species produced by a proton reaction.

## See Also

- [Continuous Energy Theory Guide](../../../docs/source/theory/continuous_energy.rst)
- [Installation — CE Library Configuration](../../../docs/source/user_guide/getting_started/installation.rst)
