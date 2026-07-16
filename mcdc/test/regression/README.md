# MC/DC - Regression Test

To run all tests:

```bash
python run.py
```

To run a specific test (with wildcard `*` support):

```bash
python run.py --name=<test_name>
```

To run in Numba mode:

```bash
python run.py --mode=numba
```

To run in multiple MPI ranks (currently support `mpiexec` and `srun`):

```bash
python run.py --mpiexec=<number of ranks>
```

To add a new test:

1. Create a folder. The name of the folder will be the test name.
2. Add the input file named `input.py`.
3. Add the answer key file named `answer.h5`.
4. Make sure that the number of particles run is large enough for a good test.
5. If the test runs longer than 10 seconds in serial Python mode, consider decreasing the number of particles.
