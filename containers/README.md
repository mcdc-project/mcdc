# MC/DC Container Guide

## What Are Containers?

A container is a lightweight, portable package that bundles an application
together with everything it needs to run: code, libraries, system tools,
and settings. Think of it like a shipping container — no matter what ship
(computer) carries it, the contents inside stay the same.

**Why does this matter for MC/DC?**

Installing MC/DC requires Python, MPI, Numba, and many other dependencies.
Getting all of these working together — especially on HPC systems where you
don't have admin access — can be painful. A container solves this by giving
you a pre-built environment where everything is already installed and tested.

## Tested Platforms

| System      | OS         | Arch   | Container Tool     | Status |
|-------------|------------|--------|--------------------|--------|
| MacBook Pro | macOS 26.3 | arm64  | Docker 29.2.0      | ✅      |
| Tuolumne    | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✅      |
| Dane        | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✅      |
| Tioga       | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✅      |
| COE (OSU)   | Rocky 8.10 | x86_64 | Apptainer 1.4.5    | ✅      |

All platforms produce identical containers: Debian 13, Python 3.11,
MPICH 4.2.1, MC/DC 0.12.0.

---

# Part 1: Getting Started (New Users)

This section is for anyone who just wants to **run MC/DC** in a container.
No prior container experience needed.

## Step 1: Pull the Pre-Built Image

You don't need to build anything. A ready-to-use image is available on
the GitHub Container Registry.

### Local Machine (Docker)

First, [install Docker Desktop](https://www.docker.com/products/docker-desktop/)
if you haven't already. Then open a terminal and run:

```bash
docker pull ghcr.io/cement-psaap/mcdc:dev
docker run --rm -it ghcr.io/cement-psaap/mcdc:dev
```

You are now inside the container. Try:

```bash
python -c "import mcdc; print('MC/DC OK')"
```

Type `exit` to leave the container.

### LLNL Systems — Tuolumne, Tioga, Dane (Podman)

Podman is already installed on LLNL systems. It works just like Docker.

```bash
podman pull ghcr.io/cement-psaap/mcdc:dev
podman run --rm -it ghcr.io/cement-psaap/mcdc:dev
```

> **Getting an error?** If you see `lsetxattr: operation not supported`,
> see [LLNL Storage Setup](#llnl-storage-setup) in Part 2.

### OSU Systems — COE (Apptainer)

Apptainer is already installed on COE. It uses a slightly different syntax.

```bash
apptainer build --sandbox mcdc_sandbox docker://ghcr.io/cement-psaap/mcdc:dev
apptainer exec mcdc_sandbox python -c "import mcdc; print('MC/DC OK')"
```

> **Note:** If `apptainer pull mcdc.sif ...` fails with "Out of memory",
> use `--sandbox` as shown above.

## Step 2: Run Your Simulation

### Docker / Podman

```bash
# Run a script (mounts your current directory into the container)
docker run --rm -v $(pwd):/work -w /work mcdc:dev python input.py

# Run with MPI (4 processes)
docker run --rm mcdc:dev mpirun -n 4 python input.py
```

For Podman, replace `docker` with `podman`.

**What do the flags mean?**

- `--rm`: Automatically clean up the container when it exits.
- `-it`: Open an interactive terminal (for typing commands).
- `-v $(pwd):/work`: Share your current folder with the container.
- `-w /work`: Start inside the shared folder.

### Apptainer (OSU)

```bash
# Run a script
apptainer exec mcdc_sandbox python input.py

# Run with MPI (4 processes)
apptainer exec mcdc_sandbox mpirun -launcher fork -n 4 python input.py
```

> **Note:** Apptainer automatically shares your home directory, so your
> files are already visible inside the container.

## Step 3: Docker Compose (Optional Shortcut)

If you use Docker on your laptop, Docker Compose lets you skip the long
commands above. From the MC/DC repo root:

```bash
# Open a development shell
docker compose -f containers/docker-compose.yml run --rm dev bash

# Run the test suite
docker compose -f containers/docker-compose.yml run --rm test

# Run with MPI
docker compose -f containers/docker-compose.yml run --rm mpi mpirun -n 4 python input.py
```

---

# Part 2: Advanced Users

This section covers building images from source, HPC-specific setup,
and troubleshooting.

## Building from Source

If you need to test local code changes or customize the image, you can
build it yourself.

**Important:** Always run build commands from the **root directory** of
the MC/DC repository (not from inside `containers/`).

```bash
cd /path/to/MCDC
ls containers/Dockerfile    # This should exist
ls pyproject.toml            # This should exist
```

### Docker (Local Machine)

```bash
docker build -f containers/Dockerfile -t mcdc:dev .
```

### Docker Compose (Local Machine)

```bash
docker compose -f containers/docker-compose.yml build
```

### Podman (LLNL Systems)

```bash
podman build -f containers/Dockerfile -t mcdc:dev .
```

### Apptainer (OSU)

Apptainer cannot build directly from a Dockerfile. Two options:

**Option A: Pull from the registry** (easiest)

```bash
apptainer build --sandbox mcdc_sandbox docker://ghcr.io/cement-psaap/mcdc:dev
```

**Option B: Build on your laptop, transfer to COE**

```bash
# On your laptop
docker build -f containers/Dockerfile -t mcdc:dev .
docker save mcdc:dev -o mcdc.tar
scp mcdc.tar <username>@submit-a.hpc.engr.oregonstate.edu:~/

# On COE
apptainer build --sandbox mcdc_sandbox docker-archive://mcdc.tar
```

## LLNL Storage Setup

LLNL HPC systems use network filesystems (NFS/Lustre) that are
incompatible with Podman's default storage. You will see this error:

```
lsetxattr: operation not supported
```

**Option A: Use the `--root` flag on every command**

```bash
podman --root /var/tmp/$USER/containers/storage build -f containers/Dockerfile -t mcdc:dev .
podman --root /var/tmp/$USER/containers/storage run --rm -it mcdc:dev
```

**Option B: Create a configuration file (recommended)**

This tells Podman to always use local storage:

```bash
mkdir -p ~/.config/containers
cat > ~/.config/containers/storage.conf << EOF
[storage]
driver = "overlay"
graphroot = "/var/tmp/$USER/containers/storage"

[storage.options.overlay]
force_mask = "700"
mount_program = "/usr/bin/fuse-overlayfs"
EOF
```

Verify it works:

```bash
podman info | grep graphRoot
```

You should see `/var/tmp/<your-username>/containers/storage`.

> **Note:** On some systems (e.g., Tioga), the config file may not take
> effect. Fall back to Option A if needed.

## Troubleshooting

### `lsetxattr: operation not supported`

**Cause:** Podman storage is on a network filesystem.
**Fix:** See [LLNL Storage Setup](#llnl-storage-setup).

### `setgroups 65534 failed` or APT sandbox errors

**Cause:** Rootless Podman cannot map the `nobody` user.
**Fix:** Already handled in the Dockerfile. Make sure you're using the latest version.

### `permission denied` when running container (Podman)

**Cause:** Some LLNL filesystems block non-root access inside containers.
**Fix:** `podman run --rm -it --user root mcdc:dev`

### `Out of memory (cache_alloc)` or `Failed to create thread` (Apptainer)

**Cause:** `mksquashfs` needs more resources than available.
**Fix:** Use sandbox mode: `apptainer build --sandbox mcdc_sandbox docker://ghcr.io/cement-psaap/mcdc:dev`

### hwloc or UCX warnings

Messages like `hwloc received invalid information` or `unable to read somaxconn`
are **harmless**. MPI still works correctly.

### `HYDU_create_process: execvp error on file srun`

**Cause:** MPICH detects Slurm and tries to use `srun` inside the container.
**Fix:** `mpirun -launcher fork -n 4 python input.py`

---

# Part 3: For Developers

This section is for MC/DC developers who need to build and publish
container images to the GitHub Container Registry.

## Pushing to the Registry

### One-Time Setup

1. Go to https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Select the `write:packages` scope
4. Copy the token
5. Login:

```bash
echo "YOUR_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

### Building and Pushing

All HPC systems are x86_64 (amd64). If you are on an Apple Silicon Mac,
you must cross-compile:

```bash
docker build --platform linux/amd64 -f containers/Dockerfile -t mcdc:dev-amd64 .
docker tag mcdc:dev-amd64 ghcr.io/cement-psaap/mcdc:dev
docker tag mcdc:dev-amd64 ghcr.io/cement-psaap/mcdc:dev-$(date +%Y-%m-%d)
docker push ghcr.io/cement-psaap/mcdc:dev
docker push ghcr.io/cement-psaap/mcdc:dev-$(date +%Y-%m-%d)
```

### Making the Package Public

After the first push, the package is private by default. To make it
accessible to everyone:

1. Go to https://github.com/orgs/CEMeNT-PSAAP/packages
2. Find `mcdc` → Settings
3. Change visibility to **Public**
4. Under "Manage Actions access", add the `MCDC` repository

### Push Troubleshooting

If you get `manifest unknown` or `permission_denied`:
- Verify the package visibility is set to Public
- Verify your token has the `write:packages` scope
- Try pushing a dated tag first (this sometimes initializes the package)

## File Overview

```
containers/
├── Dockerfile           # Build instructions for the container image
├── docker-compose.yml   # Simplified commands for Docker users
└── README.md            # This file
```