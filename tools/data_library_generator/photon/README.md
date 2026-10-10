# MC/DC Photon Data Library Generator

Converts the Evaluated Photon Data Library (EPDL) and the Evaluated Atomic Data
Library (EADL) into the HDF5 format MC/DC reads for photon transport.

## Prerequisites

- An EPDL distribution in ENDF-6 format, one file per element, named
  `EPDL.ZA<ZZZ>000.endf`.
- An EADL distribution in the same layout, named `EADL.ZA<ZZZ>000.endf`, for
  atomic relaxation. Optional: without it the library still loads, and
  photoelectric absorption simply emits no fluorescence.
- No ACE reader is needed. EPDL and EADL are plain text and are parsed directly,
  so unlike the electron generator this one has no ACEtk dependency.

## Environment Variables
| Variable          | Description                                        |
|-------------------|----------------------------------------------------|
| `MCDC_EPDL_LIB`   | Path to the directory of EPDL ENDF files.          |
| `MCDC_EADL_LIB`   | Path to the directory of EADL ENDF files.          |
| `MCDC_LIB_PHOTON` | Path to the output directory for MC/DC HDF5 files. |

```bash
export MCDC_EPDL_LIB=/path/to/endf/epdl
export MCDC_EADL_LIB=/path/to/endf/eadl
export MCDC_LIB_PHOTON=/path/to/mcdc/photon/library
```

## Usage

```bash
python generate.py              # convert elements missing from the output directory
python generate.py --rewrite    # reconvert every element
python generate.py --verbose    # print each element instead of a progress bar
```

## What it Does

For each element found in `$MCDC_EPDL_LIB`:

1. **Takes MT-501's abscissa as the shared energy grid.** EPDL tabulates the
   total cross section on the unionized grid that every other photon channel is
   a subset of, so no union has to be constructed. The grid repeats each
   absorption-edge energy, which is how ENDF represents the discontinuity, and
   those repeats are preserved exactly.
2. **Projects each channel onto that grid**, honouring the interpolation law the
   file declares. Every EPDL photon section is INT=2 (lin-lin); mixed-law tables
   are handled per range. At a repeated edge energy the first copy takes the
   limit from below the edge and the second the limit from above, so the
   discontinuity survives the projection.
3. **Copies the sub-tabulated data verbatim** — the coherent form factor and the
   incoherent scattering function, both tabulated against momentum transfer in
   inverse angstroms, keep their own grids.
4. **Writes `offset` and `unit` attributes on every cross section.** Both are
   read unconditionally by `Element.set_photon_data`, so neither is optional.
5. **Reads atomic relaxation from EADL MF=28/MT=533**, one transition table per
   subshell. `secondary_designator` distinguishes the two de-excitation
   channels: zero is radiative, so a characteristic X-ray is emitted and
   transported, and non-zero is non-radiative (Auger), whose electron is not
   transported and therefore deposits locally.

Energies are written in **eV** and cross sections in **barns**, each declaring
its unit, which is what `read_energy` and the runtime expect.

## Output HDF5 Schema

```
<symbol>.h5
├── element_symbol, element_name, atomic_number, atomic_weight_ratio
│     atomic_number and atomic_weight_ratio are read by
│     Element._compile_into_simulation for every particle type and are not
│     optional, even in a photon-only library.
├── photon_reactions/
│   ├── xs_energy_grid                       attrs: unit="eV"
│   ├── total/xs                             attrs: unit="barns"
│   ├── coherent_scattering/MT502/           attrs: MT
│   │   ├── xs                               attrs: offset, unit="barns"
│   │   ├── form_factor/{momentum_transfer, value}
│   │   └── reference_frame
│   ├── incoherent_scattering/MT504/
│   │   ├── xs
│   │   ├── scattering_function/{momentum_transfer, value}
│   │   └── reference_frame
│   ├── photoelectric/MT522/
│   │   ├── xs
│   │   ├── reference_frame
│   │   └── subshells/MT-534 .. MT-5NN/
│   │       ├── binding_energy               attrs: unit="eV"
│   │       ├── energy_grid                  attrs: unit="eV"
│   │       └── xs                           attrs: unit="barns"
│   └── pair_production/
│       ├── MT515/{xs, reference_frame}      electron field
│       ├── MT517/{xs, reference_frame}      nuclear field
│       └── total/xs                         MT-516, their sum
└── atomic_relaxation/
    ├── n_subshells
    └── subshells/MT-534 .. MT-5NN/
        ├── binding_energy, designator, n_electrons
        └── transitions/{energy, probability,
                         origin_designator, secondary_designator}
```

Reaction groups are spelled `MTnnn` and photoelectric subshells `MT-nnn`. The
inconsistency is matched rather than corrected, so that a library written here
is interchangeable with the libraries already in circulation.

## Validation

The generated `Al.h5` was compared dataset by dataset against the `Al.h5` in
`mcdc-project/mcdc-regression_test_data`. Every cross section, form factor,
scattering function, photoelectric subshell and relaxation table agrees to
within 1e-9 relative, and the energy grid, form factor and scattering function
are bit-identical. Running the same deck against each library gives identical
tallies.

Two differences remain, both deliberate:

- `atomic_weight_ratio` differs by 3e-6 relative, because it is taken from
  EPDL's own HEAD record rather than from the ACE-derived value.
- `anomalous_scattering` is **not** written. EPDL carries it as MF=27/MT=505 and
  MT=506, but the values disagree with the libraries in circulation by orders of
  magnitude, the convention behind them is not documented here, and no MC/DC
  code reads them. Writing data that cannot be validated is worse than omitting
  it.

## See Also

- `tools/data_library_generator/electron/` — the per-element generator this one
  mirrors, which reads EPRDATA14 through ACEtk.
- `tools/data_library_generator/neutron/` — the per-nuclide generator.
