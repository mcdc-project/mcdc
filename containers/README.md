# MC/DC Container Images

This directory contains guidance for building and running MC/DC
using container technologies on HPC systems (LLNL, OSU).

## Container Tools by System

| System           | Institution | Tool   |
|------------------|-------------|--------|
| Local development| —           | Docker |
| Tuolumne         | LLNL        | Podman |
| Dane             | LLNL        | Podman |
| COE DGX H100/H200| OSU         | Docker |
| COE DGX-2        | OSU         | Docker |

## Building

### Local (Docker)
```bash
docker build -t mcdc:dev .
```

### LLNL systems (Podman)
```bash
# First time only - configure podman
enable-podman

# Build image
podman build -t mcdc:dev .
```

## Running

### Local (Docker)
```bash
docker run --rm -it mcdc:dev bash
docker run --rm mcdc:dev python input.py
```

### LLNL systems (Podman)
```bash
podman run --rm -it mcdc:dev bash
podman run --rm mcdc:dev python input.py
```

### MPI + SLURM (LLNL)
```bash
srun -n 32 podman run --rm mcdc:dev python input.py