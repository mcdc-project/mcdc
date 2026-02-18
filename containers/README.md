# MC/DC Container Images

This directory contains guidance for building and running MC/DC
using container technologies on HPC systems (LLNL, OSU).

## Container Tools by System

| System            | Institution | Tool   |
|-------------------|-------------|--------|
| Local development | —           | Docker |
| Tuolumne          | LLNL        | Podman |
| Dane              | LLNL        | Podman |
| COE DGX H100/H200 | OSU         | Docker |
| COE DGX-2         | OSU         | Docker |

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
> cat > ~/.config/containers/storage.conf << 'EOF'
> [storage]
> driver = "overlay"
> graphroot = "/var/tmp/<username>/containers/storage"
>
> [storage.options.overlay]
> force_mask = "700"
> mount_program = "/usr/bin/fuse-overlayfs"
> EOF
> ```
> Replace `<username>` with your actual username. After this, you can
> use `podman` without the `--root` flag.

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

### MPI `attempt to run as root`
The Dockerfile creates a non-root user (`mcdc_user`) to avoid this.
If you see this error, rebuild the image with the latest Dockerfile.