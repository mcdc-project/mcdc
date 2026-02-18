# Use Python 3.11 as base image
# MC/DC requires Python >=3.10 and <3.12
FROM python:3.11-slim

# Image metadata - links this image to the MC/DC GitHub repository
LABEL org.opencontainers.image.source="https://github.com/CEMeNT-PSAAP/MCDC"
LABEL org.opencontainers.image.description="MC/DC: Monte Carlo Dynamic Code"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# Install system dependencies
# gcc, g++         : C/C++ compilers required for building Python extensions
# libopenmpi-dev   : OpenMPI headers required to build mpi4py from source
# openmpi-bin      : OpenMPI runtime binaries (mpirun, mpiexec, etc.)
# git              : Required for editable installs and version detection
# The final line removes apt cache to keep image size small
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenmpi-dev \
    openmpi-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
# All subsequent commands will run from this path
WORKDIR /opt/mcdc

# Copy only pyproject.toml first
# Docker caches each step - if pyproject.toml hasn't changed,
# the pip install step below will use cache (faster rebuilds)
COPY pyproject.toml .

# Install mpi4py separately before other dependencies
# This ensures mpi4py is built against the system OpenMPI (libopenmpi-dev)
# rather than downloading a pre-built binary that may not be compatible
RUN pip install --no-cache-dir mpi4py

# Copy the rest of the repository into the container
COPY . .

# Install MC/DC in editable mode with dev dependencies
# -e        : editable mode, changes to source are reflected immediately
# .[dev]    : installs optional dev dependencies (pytest, black, pre-commit)
#             defined in pyproject.toml under [project.optional-dependencies]
RUN pip install --no-cache-dir -e ".[dev]"

# Default command when container starts
# Can be overridden: docker run mcdc:dev python input.py
CMD ["bash"]