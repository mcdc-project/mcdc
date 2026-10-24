# Shared data preparation modules

Shared converters and HDF5 writers target the [MC/DC data format](../../../docs/source/reference/data_format/index.rst), `mcdc-data`.

- [ace_converter.py](ace_converter.py) interprets ACE records, converts units, and maps source data into MC/DC representations.
- [epics_converter.py](epics_converter.py) interprets ENDF-formatted EPICS evaluations and maps photon and atomic-relaxation data into MC/DC representations.
- [hdf5_writer.py](hdf5_writer.py) writes datasets and interpolation metadata, manages provenance and file merging, and publishes output files.

Projectile-specific `generate.py` scripts import these modules and handle command-line options, input selection, and output-file coordination; see the [data preparation guides](../../../docs/source/user_guide/data_preparation/index.rst).

`FORMAT` and `VERSION` in `hdf5_writer.py` identify the shared output specification.
See the [format conventions](../../../docs/source/reference/data_format/conventions.rst) for versioning, units, naming, and provenance, and the [YAML illustrations](../../../docs/source/reference/data_format/illustrations.rst) for file layouts.

