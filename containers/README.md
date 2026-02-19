# MC/DC Container Images

This directory contains guidance for building and running MC/DC
using container technologies on HPC systems (LLNL, OSU).

## Tested Platforms

| System       | OS         | Arch   | Tool          | Status |
|--------------|------------|--------|---------------|--------|
| MacBook Pro  | macOS 26.3 | arm64  | Docker 29.2.0 | ✅      |
| Tuolumne     | RHEL 8.10  | x86_64 | Podman 4.9.4  | ✅      |
| Dane         | RHEL 8.10  | x86_64 | Podman 4.9.4  | ✅      |
| Tioga        | RHEL 8.10  | x86_64 | Podman 4.9.4  | ✅      |

All platforms produce identical containers: Debian 13, Python 3.11.14,
MPICH 4.2.1, MC/DC 0.12.0.

## Container Tools by System

| System             | Institution | Tool   |
|--------------------|-------------|--------|
| Local development  | —           | Docker |
| Tuolumne           | LLNL        | Podman |
| Tioga              | LLNL        | Podman |
| Dane               | LLNL        | Podman |
| COE DGX H100/H200  | OSU         | Docker |
| COE DGX-2          | OSU         | Docker |

## Building

### Local (Docker)
```bash
docker build -t mcdc:dev .
```

### LLNL Systems (Podman)

LLNL HPC systems use network filesystems (NFS/Lustre) which are
incompatible with Podman's default storage. You must redirect
Podman storage to a local filesystem:
```bash
podman --root /var/tmp/$USER/containers/storage build -t mcdc:dev .
```

> **Note:** The `--root` flag is required on every `podman` command.
> To avoid repeating it, create `~/.config/containers/storage.conf`:
> ```bash
> mkdir -p ~/.config/containers
> cat > ~/.config/containers/storage.conf << EOF
> [storage]
> driver = "overlay"
> graphroot = "/var/tmp/$USER/containers/storage"
>
> [storage.options.overlay]
> force_mask = "700"
> mount_program = "/usr/bin/fuse-overlayfs"
> EOF
> ```
> After this, you can use `podman` without the `--root` flag.
> This works on Dane but may not take effect on all systems
> (e.g., Tioga still requires `--root`).

## Running

### Local (Docker)
```bash
docker run --rm -it mcdc:dev
docker run --rm mcdc:dev python input.py
```

### LLNL Systems (Podman)
```bash
podman --root /var/tmp/$USER/containers/storage run --rm -it mcdc:dev
podman --root /var/tmp/$USER/containers/storage run --rm mcdc:dev python input.py
```

### Running with MPI (inside container)
```bash
# Interactive
podman --root /var/tmp/$USER/containers/storage run --rm -it mcdc:dev
mpirun -n 4 python input.py

# One-shot
podman --root /var/tmp/$USER/containers/storage run --rm mcdc:dev \
    mpirun -n 4 python input.py
```

## Troubleshooting

### `lsetxattr: operation not supported`
Podman storage is on a network filesystem. Use `--root /var/tmp/$USER/containers/storage`
to redirect storage to a local filesystem.

### `setgroups 65534 failed`
The Dockerfile already includes a fix for this (APT sandbox disable).
If you see this error, make sure you are using the latest Dockerfile.

### hwloc or UCX warnings
Messages like `hwloc received invalid information` or `unable to read somaxconn`
are harmless. The container cannot fully read the host's hardware topology
but MPI communication still works correctly via shared memory.