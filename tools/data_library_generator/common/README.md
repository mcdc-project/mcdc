# Shared data preparation modules

Shared converters and HDF5 writers target the [MC/DC data format](../../../docs/source/reference/data_format/index.rst), `mcdc-data`.

- [ace_converter.py](ace_converter.py) interprets ACE records, converts units, and maps source data into MC/DC representations.
- [epics_converter.py](epics_converter.py) interprets ENDF-formatted EPICS evaluations, maps photon and atomic-relaxation data into MC/DC representations, and prepares shared EADL element records in `atoms.h5`.
- [hdf5_writer.py](hdf5_writer.py) writes datasets and interpolation metadata, manages file and record provenance, and publishes output files.
- [batch.py](batch.py) provides the shared `--keep-going` option and conversion failure reporting.

Projectile-specific `generate.py` scripts import these modules and handle command-line options, input selection, and output-file coordination; see the [data preparation guides](../../../docs/source/user_guide/data_preparation/index.rst).

`FORMAT` and `VERSION` in `hdf5_writer.py` identify the shared output specification.
See the [format conventions](../../../docs/source/reference/data_format/conventions.rst) for versioning, units, naming, and provenance, and the [YAML illustrations](../../../docs/source/reference/data_format/illustrations.rst) for file layouts.

