# MC/DC neutron data preparation

Convert neutron ACE files to the [MC/DC data format](../../../docs/source/reference/data_format/index.rst), `mcdc-data`.
Use a Python environment with [ACEtk](https://github.com/njoy/ACEtk), `h5py`, `numpy`, and `tqdm`.

From the repository root:

```bash
export MCDC_LIB="/path/to/output/mcdc-library"
export MCDC_ACE_LIB="/path/to/input/ace-library"

python tools/data_library_generator/neutron/generate.py --ace-files 1002.10c 88223.10c 92240.10c
```

Omit `--ace-files` to process all input files; use `--rewrite` to regenerate existing data or `--verbose` for reaction details.
See the [neutron preparation guide](../../../docs/source/user_guide/data_preparation/neutron.rst) for input-library conventions, temperature mapping, reaction selection, conversion behavior, validation, and the shared `--keep-going` option.

`generate.py` coordinates command-line options and output files using the [shared modules](../common/README.md).
The [format reference](../../../docs/source/reference/data_format/index.rst) and [YAML illustrations](../../../docs/source/reference/data_format/illustrations.rst) describe the output structure.
