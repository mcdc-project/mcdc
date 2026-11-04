# Proton data preparation

The proton generator converts proton ACE and optional NIST PSTAR data into `proton-{nuclide}.h5` files in the MC/DC `mcdc-data` format.

For requirements, environment variables, command-line options (including `--keep-going`), supported reactions, and output details, see the [Proton data preparation guide](../../../docs/source/user_guide/data_preparation/proton.rst).

Run the generator from the repository root:

```bash
python tools/data_library_generator/proton/generate.py
```
