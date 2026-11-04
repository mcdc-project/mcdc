# MC/DC photon data preparation

Convert EPDL and EADL ENDF files to the [MC/DC data format](../../../docs/source/reference/data_format/index.rst), `mcdc-data`.
Use a Python environment with `h5py`, `numpy`, and `tqdm`.

From the repository root:

```bash
export MCDC_LIB="/path/to/output/mcdc-library"
export MCDC_EPICS_LIB="/path/to/input/epics2025"

python tools/data_library_generator/photon/generate.py --elements H Mg Si
```

Photon files are named `photon-{element}.h5`; shared EADL records are stored in `atoms.h5`.
Place `EPDL.ZA<ZZZ>000.endf` and matching `EADL.ZA<ZZZ>000.endf` files directly in `MCDC_EPICS_LIB`.
Omit `--elements` to process all EPDL inputs; use `--rewrite` to regenerate photon data or `--verbose` for reaction details.
See the [photon preparation guide](../../../docs/source/user_guide/data_preparation/photon.rst) for source libraries, conversion behavior, validation, and the shared `--keep-going` option.

`generate.py` coordinates element selection and shared-file updates using the [common modules](../common/README.md).
The [format reference](../../../docs/source/reference/data_format/index.rst) and [YAML illustrations](../../../docs/source/reference/data_format/illustrations.rst) describe the output structure.
