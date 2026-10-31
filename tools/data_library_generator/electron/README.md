# Electron data preparation

[generate.py](generate.py) prepares `electron-{element}.h5` from EPRDATA14 ACE tables using the shared [ACE converter](../common/ace_converter.py) and [HDF5 writer](../common/hdf5_writer.py).
It also fills missing element records in `atoms.h5`, retaining existing atomic evaluations.

See the [electron preparation guide](../../../docs/source/user_guide/data_preparation/electron.rst) for dependencies, source downloads, environment variables, and command-line options.
The [data-format reference](../../../docs/source/reference/data_format/index.rst) defines the output layout and reusable representations.
